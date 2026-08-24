"""
assimilation/services/assimilation_service.py — Sequential Forecast-Assimilate Loop
=====================================================================================

Implements the complete EnKF data assimilation cycle:

    while not harvest:
        1. Find next observation date
        2. Forecast ensemble to that date
        3. Retrieve + QC-filter observations from DB
        4. Build observation vector y and error covariance R
        5. Apply EnKF update  → X_a
        6. Persist AssimilationState record
        7. Inject corrected states into ensemble members
        8. Continue

Design principles:
    - STATELESS: holds no mutable simulation state between calls.
    - DB-decoupled: the observation repo and state repo are injected.
    - No WOFOST imports: delegates to EnsembleManager / StateUpdater.
    - Partial observations: variables absent from y are left as NaN → EnKF skips them.
    - Outlier rejection: configurable z-score gate before building y.
    - Cloud/quality filtering: configurable thresholds on Observation metadata.
    - Irregular intervals: driven by actual observation timestamps — no fixed stride.
"""

from __future__ import annotations

import datetime
import logging
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.app.assimilation.ensemble.ensemble_manager import EnsembleManager
from backend.app.assimilation.filters.enkf import enkf_update
from backend.app.assimilation.forecast.forecast_step import forecast_until
from backend.app.assimilation.operators.observation_operator import (
    DirectObservationOperator,
    SurfaceSoilMoistureObservationOperator,
    UnsupportedObservationError,
    get_observation_operator,
)
from backend.app.assimilation.models.assimilation_state import AssimilationState
from backend.app.assimilation.models.observation import Observation, ObservationSource, ObservationStatus
from backend.app.assimilation.repositories.assimilation_state_repository import AssimilationStateRepository
from backend.app.assimilation.repositories.observation_repository import ObservationRepository
from backend.app.assimilation.covariance.observation_covariance import ObservationCovariance
from backend.app.assimilation.state.state_vector import STATE_VARIABLES, STATE_INDEX, STATE_DIM, StateVector
from backend.app.assimilation.updater.state_updater import StateUpdater, InjectionResult
from backend.app.models.assimilation_run import AssimilationRun
from backend.app.services.confidence_estimator import ConfidenceEstimator
from backend.app.services.multi_source_fusion_service import MultiSourceFusionService
from backend.app.services.quality_control_service import QualityControlService, QCConfig
from backend.app.api.schemas.fusion import FusionRequest, ConfidenceRequest, ObservationSource as FusionObservationSource

logger = logging.getLogger(__name__)

# Map observation variable_name (uppercase DB convention) → StateVector lowercase key
_OBS_VAR_TO_SV: dict[str, str] = {
    "LAI":                      "lai",
    "SM":                       "sm",
    "ROOT_ZONE_SOIL_MOISTURE":  "sm",
    "ROOT_ZONE_SM":             "sm",
    "SURFACE_SOIL_MOISTURE":    "sm",
    "SURFACE_SM":                "sm",
    "TAGP":                     "tagp",
    "TWSO":                     "twso",
    "RFTRA":                    "rftra",
    "TWLV":                     "twlv",
    "TWST":                     "twst",
    "TWRT":                     "twrt",
    "DVS":                      "dvs",
    "RD":                       "rd",
}


def _map_source_to_fusion_enum(obs: Observation) -> FusionObservationSource:
    src_val = obs.source.value if hasattr(obs.source, "value") else str(obs.source)
    raw_prov = getattr(obs, "provider_name", "") or ""
    prov = str(raw_prov).upper()
    if src_val == "SATELLITE":
        if "MODIS" in prov:
            return FusionObservationSource.MODIS
        elif "SENTINEL1" in prov or "SAR" in prov:
            return FusionObservationSource.SENTINEL1_SAR
        return FusionObservationSource.SENTINEL2
    elif src_val == "SENSOR":
        if "SAR" in prov or "SENTINEL1" in prov:
            return FusionObservationSource.SENTINEL1_SAR
        return FusionObservationSource.ERA5_LAND
    elif src_val == "MANUAL":
        return FusionObservationSource.SMARTPHONE_GRVI
    elif src_val == "WEATHER":
        return FusionObservationSource.ERA5_LAND
    elif src_val == "MODEL":
        return FusionObservationSource.FUSED
    try:
        return FusionObservationSource(src_val)
    except (ValueError, TypeError):
        return FusionObservationSource.FUSED


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class QCFilter:
    """Quality-control thresholds applied before building the observation vector.

    Observations failing any active threshold are silently excluded from the
    current assimilation cycle (they remain VALID in the DB — no status mutation).
    """
    min_quality_score: Optional[int]   = 60     # Skip obs with quality_score < this
    max_cloud_cover:   Optional[float] = 0.20   # Skip satellite obs with cloud_cover > this
    max_z_score:       float           = 3.0    # Outlier gate: skip if |z| > this vs ensemble


@dataclass
class AssimilationConfig:
    """Configuration for a full-season assimilation run."""
    # ── Observation sources to include
    include_sources: list[str] = field(
        default_factory=lambda: [s.value for s in ObservationSource]
    )
    # ── QC settings
    qc: QCFilter = field(default_factory=QCFilter)
    # ── Ensemble settings
    ensemble_size: int = 50
    # ── Aggregation: when multiple obs for same variable on same date, how to combine
    # "mean" averages value and propagates uncertainty; "best_quality" picks highest score
    aggregation: str = "mean"
    # ── Minimum observations to trigger an EnKF update (skip if fewer pass QC)
    min_obs_for_update: int = 1
    # ── StateUpdater flags
    inject_dvs: bool = False
    inject_rd:  bool = False


# ── Per-cycle result ──────────────────────────────────────────────────────────

@dataclass
class AssimilationCycleResult:
    """Result of a single forecast → assimilate → inject cycle."""
    cycle_date:          datetime.date
    obs_retrieved:       int                         # raw obs from DB
    obs_after_qc:        int                         # obs passing QC
    obs_assimilated:     int                         # obs with matching SV variable
    variables_updated:   list[str]                   # SV variables that received EnKF update
    ensemble_mean_prior: dict[str, Optional[float]]  # x_f mean as dict
    ensemble_mean_post:  dict[str, Optional[float]]  # x_a mean as dict
    innovation:          dict[str, Optional[float]]  # y - H*x_f per variable
    injection_results:   list[InjectionResult]       # per-member injection records
    persisted_state_id:  Optional[uuid.UUID]         # AssimilationState DB pk
    skipped:             bool = False                # True if min_obs not met
    skip_reason:         Optional[str] = None
    fusion_diagnostics:  dict = field(default_factory=dict)  # dynamic R & fusion diagnostics


# ── Full-season result ────────────────────────────────────────────────────────

@dataclass
class SeasonAssimilationResult:
    """Aggregated result for a complete season assimilation run."""
    field_id:            Optional[uuid.UUID]
    simulation_run_id:   Optional[uuid.UUID]
    sow_date:            datetime.date
    harvest_date:        datetime.date
    total_cycles:        int
    executed_cycles:     int
    skipped_cycles:      int
    cycle_results:       list[AssimilationCycleResult]

    @property
    def total_observations_assimilated(self) -> int:
        return sum(c.obs_assimilated for c in self.cycle_results)


# ── Service ───────────────────────────────────────────────────────────────────

class AssimilationService:
    """Sequential forecast-assimilate loop for EnKF crop state estimation.

    Usage:
        manager = EnsembleManager(...)
        manager.create_ensemble(n=50)

        service = AssimilationService(
            obs_repo=ObservationRepository(db),
            state_repo=AssimilationStateRepository(db),
        )
        result = service.run_season(
            manager=manager,
            harvest_date=date(2024, 7, 30),
            field_id=field_uuid,
        )
    """

    def __init__(
        self,
        obs_repo: ObservationRepository,
        state_repo: AssimilationStateRepository,
        config: Optional[AssimilationConfig] = None,
    ) -> None:
        self.obs_repo   = obs_repo
        self.state_repo = state_repo
        self.config     = config or AssimilationConfig()
        self._updater   = StateUpdater(
            inject_dvs=self.config.inject_dvs,
            inject_rd=self.config.inject_rd,
            verify=False,  # speed: skip read-back in production loop
        )
        self.confidence_estimator = ConfidenceEstimator()
        db_session = getattr(self.obs_repo, "db", None)
        self.fusion_service = MultiSourceFusionService(db_session=db_session) if db_session else None
        self.qc_service = QualityControlService()

    # ── Public API ────────────────────────────────────────────────────────

    def run_season(
        self,
        manager: EnsembleManager,
        harvest_date: datetime.date,
        *,
        field_id:          Optional[uuid.UUID] = None,
        simulation_run_id: Optional[uuid.UUID] = None,
        assimilation_run_id: Optional[uuid.UUID] = None,
    ) -> SeasonAssimilationResult:
        """Run the complete forecast-assimilate loop for a full crop season.

        Discovers observation dates automatically from the DB, then iterates
        the EnKF cycle for each date up to harvest_date.

        Args:
            manager:           Initialised EnsembleManager with N members created.
            harvest_date:      Stop criterion — loop ends when all members reach this date.
            field_id:          Optional field UUID for DB queries and persistence.
            simulation_run_id: Optional SimulationRun UUID for linking AssimilationState records.
            assimilation_run_id: Optional AssimilationRun UUID. If not provided but simulation_run_id is, a new run will be created.

        Returns:
            SeasonAssimilationResult with per-cycle diagnostics.
        """
        if not manager.members:
            raise ValueError("EnsembleManager has no members. Call create_ensemble() first.")

        sow_date      = manager.members[0].current_date
        obs_dates     = self._discover_observation_dates(field_id, sow_date, harvest_date)
        cycle_results: list[AssimilationCycleResult] = []

        logger.info(
            "AssimilationService.run_season: field=%s sow=%s harvest=%s obs_dates=%d",
            field_id, sow_date, harvest_date, len(obs_dates),
        )

        db_session = self.state_repo.session
        auto_run = None
        if simulation_run_id is not None and assimilation_run_id is None:
            try:
                auto_run = AssimilationRun(
                    simulation_id=simulation_run_id,
                    ensemble_size=len(manager.members),
                    status="RUNNING",
                    total_cycles=len(obs_dates),
                    executed_cycles=0,
                    skipped_cycles=0,
                    observations_used=0,
                )
                db_session.add(auto_run)
                db_session.commit()
                db_session.refresh(auto_run)
                assimilation_run_id = auto_run.id
            except Exception as e:
                logger.error("Failed to automatically create AssimilationRun record: %s", e)
                db_session.rollback()

        try:
            for obs_date in obs_dates:
                if obs_date >= harvest_date:
                    break
                # Check if all members have already terminated
                if all(m.wofost.flag_terminate for m in manager.members):
                    logger.info("All ensemble members terminated before harvest. Stopping.")
                    break

                result = self._run_cycle(
                    manager=manager,
                    obs_date=obs_date,
                    field_id=field_id,
                    simulation_run_id=simulation_run_id,
                    assimilation_run_id=assimilation_run_id,
                )
                cycle_results.append(result)

            executed = sum(1 for c in cycle_results if not c.skipped)
            skipped  = sum(1 for c in cycle_results if c.skipped)

            if auto_run is not None:
                auto_run.status = "COMPLETED"
                auto_run.completed_at = datetime.datetime.now(datetime.timezone.utc)
                auto_run.executed_cycles = executed
                auto_run.skipped_cycles = skipped
                auto_run.observations_used = sum(c.obs_assimilated for c in cycle_results)
                db_session.commit()
                db_session.refresh(auto_run)

        except Exception as e:
            if auto_run is not None:
                try:
                    auto_run.status = "FAILED"
                    auto_run.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    db_session.commit()
                except Exception as db_err:
                    logger.error("Failed to update auto_run to FAILED status: %s", db_err)
            raise e

        return SeasonAssimilationResult(
            field_id=field_id,
            simulation_run_id=simulation_run_id,
            sow_date=sow_date,
            harvest_date=harvest_date,
            total_cycles=len(cycle_results),
            executed_cycles=executed,
            skipped_cycles=skipped,
            cycle_results=cycle_results,
        )

    def run_single_cycle(
        self,
        manager: EnsembleManager,
        obs_date: datetime.date,
        *,
        field_id:          Optional[uuid.UUID] = None,
        simulation_run_id: Optional[uuid.UUID] = None,
        assimilation_run_id: Optional[uuid.UUID] = None,
    ) -> AssimilationCycleResult:
        """Run one forecast → assimilate → inject cycle for a given observation date.

        Useful for step-by-step control or replaying a single assimilation event.
        """
        return self._run_cycle(
            manager=manager,
            obs_date=obs_date,
            field_id=field_id,
            simulation_run_id=simulation_run_id,
            assimilation_run_id=assimilation_run_id,
        )

    # ── Internal loop ─────────────────────────────────────────────────────

    def _run_cycle(
        self,
        manager: EnsembleManager,
        obs_date: datetime.date,
        field_id: Optional[uuid.UUID],
        simulation_run_id: Optional[uuid.UUID],
        assimilation_run_id: Optional[uuid.UUID] = None,
    ) -> AssimilationCycleResult:
        """Execute one EnKF cycle: forecast → observe → update → inject → persist."""

        # ── 1. Forecast ensemble to observation date ──────────────────────
        logger.debug("Cycle %s: running forecast step", obs_date)
        X_f, x_mean_f = forecast_until(manager, obs_date)

        # ── 2. Retrieve observations from DB ─────────────────────────────
        raw_obs = self._fetch_observations(field_id, obs_date)
        logger.debug("Cycle %s: retrieved %d observations", obs_date, len(raw_obs))

        # ── 3. QC filtering ───────────────────────────────────────────────
        qc_obs = self._apply_qc(raw_obs, X_f, x_mean_f)
        logger.debug("Cycle %s: %d observations passed QC", obs_date, len(qc_obs))

        if len(qc_obs) < self.config.min_obs_for_update:
            return AssimilationCycleResult(
                cycle_date=obs_date,
                obs_retrieved=len(raw_obs),
                obs_after_qc=len(qc_obs),
                obs_assimilated=0,
                variables_updated=[],
                ensemble_mean_prior=self._vec_to_dict(x_mean_f),
                ensemble_mean_post=self._vec_to_dict(x_mean_f),
                innovation={v: None for v in STATE_VARIABLES},
                injection_results=[],
                persisted_state_id=None,
                skipped=True,
                skip_reason=f"Only {len(qc_obs)} obs passed QC (min={self.config.min_obs_for_update})",
            )

        # ── 4. Build y and R via Data Fusion Pipeline ──────────────────────────
        y, R, obs_assimilated, fusion_diag = self._build_observation_vector(
            qc_obs, field_id=field_id, obs_date=obs_date
        )

        if obs_assimilated == 0:
            return AssimilationCycleResult(
                cycle_date=obs_date,
                obs_retrieved=len(raw_obs),
                obs_after_qc=len(qc_obs),
                obs_assimilated=0,
                variables_updated=[],
                ensemble_mean_prior=self._vec_to_dict(x_mean_f),
                ensemble_mean_post=self._vec_to_dict(x_mean_f),
                innovation={v: None for v in STATE_VARIABLES},
                injection_results=[],
                persisted_state_id=None,
                skipped=True,
                skip_reason="No QC-passed obs mapped to a known StateVector variable",
                fusion_diagnostics=fusion_diag,
            )

        # ── 5. EnKF update with explicit observation operator ───────────────
        obs_idx = np.where(np.isfinite(y))[0]
        obs_operator = DirectObservationOperator(obs_idx, state_dim=STATE_DIM)
        X_a, d, K = enkf_update(X_f, y, R, H_operator=obs_operator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_mean_a = np.nanmean(X_a, axis=1)

        variables_updated = [
            STATE_VARIABLES[i] for i in range(STATE_DIM) if not np.isnan(d[i])
        ]
        logger.info("Cycle %s: EnKF updated %d variables: %s", obs_date, len(variables_updated), variables_updated)

        if "sm" in variables_updated:
            sm_idx = STATE_INDEX["sm"]
            prior_sm = float(x_mean_f[sm_idx])
            obs_sm = float(y[sm_idx]) if not np.isnan(y[sm_idx]) else None
            post_sm = float(x_mean_a[sm_idx])
            delta_sm = float(post_sm - prior_sm)

            rd_vals = []
            for m in manager.members:
                try:
                    if hasattr(m, "wofost") and hasattr(m.wofost, "get_variable"):
                        val = m.wofost.get_variable("RD")
                        if val is not None and float(val) > 0:
                            rd_vals.append(float(val))
                except Exception:
                    pass
            rd = float(np.mean(rd_vals)) if rd_vals else 10.0
            implied_delta_w = float(delta_sm * rd)
            unc = float(np.sqrt(R[sm_idx, sm_idx])) if not np.isnan(R[sm_idx, sm_idx]) else None

            fusion_diag["sm_diagnostics"] = {
                "prior_sm": round(prior_sm, 6),
                "observed_sm": round(obs_sm, 6) if obs_sm is not None else None,
                "posterior_sm": round(post_sm, 6),
                "delta_sm": round(delta_sm, 6),
                "rd": round(rd, 4),
                "implied_delta_w": round(implied_delta_w, 6),
                "uncertainty": round(unc, 6) if unc is not None else None,
            }

        # ── 6. Persist AssimilationState ──────────────────────────────────
        state_id = self._persist(
            X_f=X_f, X_a=X_a, y=y, d=d, K=K,
            n_members=len(manager.members),
            n_obs=obs_assimilated,
            obs_date=obs_date,
            field_id=field_id,
            simulation_run_id=simulation_run_id,
            assimilation_run_id=assimilation_run_id,
        )

        # ── 7. Inject corrected states ────────────────────────────────────
        analysis_states = [
            StateVector.from_numpy(X_a[:, i], date=obs_date)
            for i in range(X_a.shape[1])
        ]
        injection_results = self._updater.inject_ensemble(manager.members, analysis_states)

        return AssimilationCycleResult(
            cycle_date=obs_date,
            obs_retrieved=len(raw_obs),
            obs_after_qc=len(qc_obs),
            obs_assimilated=obs_assimilated,
            variables_updated=variables_updated,
            ensemble_mean_prior=self._vec_to_dict(x_mean_f),
            ensemble_mean_post=self._vec_to_dict(x_mean_a),
            innovation=self._vec_to_dict(d),
            injection_results=injection_results,
            persisted_state_id=state_id,
            fusion_diagnostics=fusion_diag,
        )

    # ── Observation helpers ───────────────────────────────────────────────

    def _discover_observation_dates(
        self,
        field_id: Optional[uuid.UUID],
        sow_date: datetime.date,
        harvest_date: datetime.date,
    ) -> list[datetime.date]:
        """Return sorted list of unique calendar dates with valid observations."""
        if field_id is None:
            return []

        start = datetime.datetime.combine(sow_date, datetime.time.min, tzinfo=datetime.timezone.utc)
        end   = datetime.datetime.combine(harvest_date, datetime.time.min, tzinfo=datetime.timezone.utc)

        obs = self.obs_repo.get_observations_between(
            field_id=field_id,
            start=start,
            end=end,
            status=ObservationStatus.VALID,
            limit=10000,
        )
        dates = sorted({o.timestamp.date() for o in obs})
        logger.info("Discovered %d unique observation dates for field=%s", len(dates), field_id)
        return dates

    def _fetch_observations(
        self,
        field_id: Optional[uuid.UUID],
        obs_date: datetime.date,
    ) -> list[Observation]:
        """Fetch all VALID observations for a field on a given calendar date."""
        if field_id is None:
            return []
        return self.obs_repo.get_by_date(
            field_id=field_id,
            date=obs_date,
            status=ObservationStatus.VALID,
        )

    def _apply_qc(
        self,
        observations: list[Observation],
        X_f: np.ndarray,
        x_mean_f: np.ndarray,
    ) -> list[Observation]:
        """Apply quality filters via QualityControlService; return observations that pass all checks."""
        qc_config = QCConfig(
            min_quality_score=self.config.qc.min_quality_score,
            max_cloud_cover=self.config.qc.max_cloud_cover,
            max_z_score=self.config.qc.max_z_score,
            include_sources=self.config.include_sources,
        )
        return self.qc_service.filter_observations(
            observations,
            X_f=X_f,
            x_mean_f=x_mean_f,
            config=qc_config,
        )

    def _build_observation_vector(
        self,
        observations: list[Observation],
        field_id: Optional[uuid.UUID] = None,
        obs_date: Optional[datetime.date] = None,
        explicit_R: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, int, dict]:
        """Aggregate & fuse observations into the EnKF y vector and dynamic R matrix.

        Processes raw QC-passed observations through ConfidenceEstimator and
        MultiSourceFusionService to generate dynamic observation error covariance (R)
        and fused observation vector (y).

        Supports optional explicit covariance matrix `explicit_R` supplied by a trusted
        component, falling back safely to diagonal independence if invalid or omitted.

        Returns:
            y: fused observation vector shape (STATE_DIM,), NaN for unobserved vars
            R: observation error covariance matrix (STATE_DIM, STATE_DIM)
            n_assimilated: number of distinct variables in y (non-NaN count)
            fusion_diagnostics: dictionary containing fusion metadata & dynamic R values
        """
        groups: dict[str, list[Observation]] = {}
        for obs in observations:
            sv_key = _OBS_VAR_TO_SV.get(obs.variable_name.upper())
            if sv_key is None:
                continue  # variable not in state vector
            groups.setdefault(sv_key, []).append(obs)

        y = np.full(STATE_DIM, np.nan)
        r_diag = np.full(STATE_DIM, np.nan)
        fusion_diagnostics: dict[str, dict] = {}

        for sv_key, obs_list in groups.items():
            idx = STATE_INDEX[sv_key]

            # 1. Compute dynamic confidence scores & R values
            prepared_obs = []
            for obs in obs_list:
                fusion_src = _map_source_to_fusion_enum(obs)
                ts = getattr(obs, "timestamp", None)
                obs_dt = ts.date() if isinstance(ts, datetime.datetime) else (ts if isinstance(ts, datetime.date) else None)
                c_date = obs_date or obs_dt or datetime.date.today()
                days_since = max(0, (datetime.date.today() - c_date).days) if isinstance(c_date, datetime.date) else 0

                raw_fid = field_id or getattr(obs, "field_id", None)
                f_id = raw_fid if isinstance(raw_fid, (uuid.UUID, str)) else None

                raw_cloud = getattr(obs, "cloud_cover", 0.0)
                cloud_cover = raw_cloud if isinstance(raw_cloud, (int, float)) else 0.0

                val = getattr(obs, "value", 0.0)
                obs_val = float(val) if isinstance(val, (int, float)) else 0.0

                conf_req = ConfidenceRequest(
                    source=fusion_src,
                    value=obs_val,
                    cloud_cover=float(cloud_cover),
                    viewing_angle=0.0,
                    sensor_health=1.0,
                    days_since_observation=int(days_since),
                    field_id=f_id,
                )
                conf_resp = self.confidence_estimator.compute_confidence(conf_req)

                unc = getattr(obs, "uncertainty", None)
                if unc is not None and isinstance(unc, (int, float)) and unc > 0:
                    r_std = float(unc)
                else:
                    r_std = float(conf_resp.observation_error_r)

                r_var = r_std ** 2

                # Select observation operator based on variable and source provenance
                src_str = getattr(obs.source, "value", str(obs.source))
                op = get_observation_operator(obs.variable_name, source=src_str, uncertainty=r_std)
                if isinstance(op, SurfaceSoilMoistureObservationOperator):
                    logger.warning(
                        "AssimilationService: Surface SM observation from %s (depth 0-5 cm, support=surface_skin) "
                        "cannot be directly mapped to WOFOST root-zone SM (0-100 cm) without vertical hydrology model — skipping direct SM update",
                        src_str,
                    )
                    continue

                prepared_obs.append((obs, fusion_src, conf_resp, r_std, r_var))

            if not prepared_obs:
                continue

            # 2. Fuse observations
            if self.config.aggregation == "best_quality":
                best_item = max(
                    prepared_obs,
                    key=lambda item: (item[0].quality_score or 0, item[2].confidence_score),
                )
                fused_val = best_item[0].value
                var_combined = best_item[4]
                sources_used = [best_item[1].value]
                conf_scores = [best_item[2].confidence_score]
            elif sv_key == "lai" and len(set(f_src for _, f_src, _, _, _ in prepared_obs)) > 1 and self.fusion_service:
                c_date = obs_date or (obs_list[0].timestamp.date() if getattr(obs_list[0], "timestamp", None) else datetime.date.today())
                max_cloud = max([getattr(o, "cloud_cover", 0.0) or 0.0 for o in obs_list])
                req_fid = field_id if isinstance(field_id, (uuid.UUID, str)) else uuid.uuid4()
                fusion_req = FusionRequest(
                    field_id=req_fid,
                    date=c_date,
                    observations=[
                        {
                            "source": f_src.value,
                            "value": obs.value,
                            "confidence": c_resp.confidence_score,
                            "observation_error_r": r_std,
                            "variance": r_var,
                        }
                        for obs, f_src, c_resp, r_std, r_var in prepared_obs
                    ],
                    cloud_cover=max_cloud,
                )
                fusion_resp = self.fusion_service.fuse_lai(fusion_req)
                fused_val = fusion_resp.fused_lai
                inv_vars = [1.0 / max(1e-6, r_var) for _, _, _, _, r_var in prepared_obs]
                var_combined = 1.0 / max(1e-6, sum(inv_vars))
                sources_used = fusion_resp.contributing_sources or [f_src.value for _, f_src, _, _, _ in prepared_obs]
                conf_scores = [c_resp.confidence_score for _, _, c_resp, _, _ in prepared_obs]
            else:
                vals = np.array([obs.value for obs, _, _, _, _ in prepared_obs])
                r_vars = np.array([r_var for _, _, _, _, r_var in prepared_obs])
                inv_vars = 1.0 / np.maximum(1e-6, r_vars)
                fused_val = float(np.average(vals, weights=inv_vars))
                var_combined = float(1.0 / np.sum(inv_vars))
                sources_used = [f_src.value for _, f_src, _, _, _ in prepared_obs]
                conf_scores = [c_resp.confidence_score for _, _, c_resp, _, _ in prepared_obs]

            y[idx] = fused_val
            r_diag[idx] = var_combined
            fusion_diagnostics[sv_key] = {
                "fused_value": round(float(fused_val), 4),
                "dynamic_r_variance": round(float(var_combined), 6),
                "dynamic_r_std": round(float(np.sqrt(var_combined)), 4),
                "obs_count": len(obs_list),
                "sources_used": sources_used,
                "confidence_scores": conf_scores,
            }

        # Build R via ObservationCovariance abstraction
        if explicit_R is not None:
            obs_cov = ObservationCovariance.from_matrix(
                explicit_R, fallback_variances=r_diag, expected_dim=STATE_DIM
            )
        else:
            obs_cov = ObservationCovariance.from_variances(r_diag, dim=STATE_DIM)

        R = obs_cov.matrix
        n_assimilated = int(np.sum(~np.isnan(y)))

        return y, R, n_assimilated, fusion_diagnostics

    # ── Persistence ───────────────────────────────────────────────────────

    def _persist(
        self,
        *,
        X_f: np.ndarray,
        X_a: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        K: np.ndarray,
        n_members: int,
        n_obs: int,
        obs_date: datetime.date,
        field_id: Optional[uuid.UUID],
        simulation_run_id: Optional[uuid.UUID],
        assimilation_run_id: Optional[uuid.UUID] = None,
    ) -> Optional[uuid.UUID]:
        """Persist an AssimilationState record. Returns the new record's UUID, or None on error."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_mean_f = np.nanmean(X_f, axis=1)
                x_mean_a = np.nanmean(X_a, axis=1)
                # Full covariance (stored as flat list for JSON)
                cov_f = np.cov(X_f).tolist()

            assimilation_time = datetime.datetime.combine(
                obs_date, datetime.time.min, tzinfo=datetime.timezone.utc
            )

            record = AssimilationState(
                field_id=field_id,
                simulation_run_id=simulation_run_id,
                assimilation_run_id=assimilation_run_id,
                assimilation_time=assimilation_time,
                ensemble_mean=self._vec_to_dict(x_mean_f),
                ensemble_covariance={"matrix": cov_f, "variables": list(STATE_VARIABLES)},
                observation_vector=self._vec_to_dict(y),
                innovation_vector=self._vec_to_dict(d),
                kalman_gain={"matrix": K.tolist(), "variables": list(STATE_VARIABLES)},
                updated_state_vector=self._vec_to_dict(x_mean_a),
                forecast_state_vector=self._vec_to_dict(x_mean_f),
                number_of_members=n_members,
                observation_count=n_obs,
            )
            saved = self.state_repo.save_state(record)
            logger.info("Persisted AssimilationState id=%s date=%s", saved.id, obs_date)
            return saved.id

        except Exception as exc:
            logger.error("Failed to persist AssimilationState: %s", exc, exc_info=True)
            return None

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _vec_to_dict(arr: np.ndarray) -> dict[str, Optional[float]]:
        """Convert a STATE_DIM numpy array to a {variable: value} dict. NaN → None."""
        result: dict[str, Optional[float]] = {}
        for i, var in enumerate(STATE_VARIABLES):
            v = float(arr[i]) if i < len(arr) else None
            result[var] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(v, 6)
        return result
