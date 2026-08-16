"""
backend/app/features/feature_engine.py
======================================

Leakage-Safe Feature Engine for AgriTwin
---------------------------------------
Computes state, growth rate, stress, EnKF diagnostic, and observation quality features
strictly using data available on or before the target forecast/assimilation timestamp (`as_of_date`).

Guarantees:
1. Temporal Leakage Safety: Filters out all observation, weather, daily output, and assimilation
   records with timestamps > as_of_date.
2. Reuses existing state vectors, ORM models, and diagnostic structures without modifying
   WOFOST or EnKF mathematical cores.
3. No ML model training or synthetic data generation inside the engine.
"""

import datetime
import logging
from typing import Any, Dict, List, Optional, Union

from backend.app.features.schemas import (
    AssimilationDiagnosticFeatures,
    FeatureVector,
    GrowthRateFeatures,
    ObservationQualityFeatures,
    ThermalStressFeatures,
    WaterStressFeatures,
)

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Extracts leakage-safe tabular feature vectors from multi-source digital twin state data."""

    def __init__(self, heat_threshold_c: float = 35.0, cold_threshold_c: float = 5.0):
        """Initialise FeatureEngine with configurable stress thresholds.

        Args:
            heat_threshold_c: Daily maximum temperature (°C) threshold for heat stress counting.
            cold_threshold_c: Daily minimum temperature (°C) threshold for cold stress counting.
        """
        self.heat_threshold_c = heat_threshold_c
        self.cold_threshold_c = cold_threshold_c

    def compute_features(
        self,
        as_of_date: datetime.date,
        daily_outputs: List[Any],
        assimilation_states: Optional[List[Any]] = None,
        observations: Optional[List[Any]] = None,
        weather_records: Optional[List[Dict[str, Any]]] = None,
    ) -> FeatureVector:
        """Compute leakage-safe feature vector as of a specific forecast/assimilation date.

        Args:
            as_of_date: Cutoff date. All input data after this date is strictly ignored.
            daily_outputs: List of DailyOutput ORM objects or dicts up to or including as_of_date.
            assimilation_states: List of AssimilationState ORM objects, CompactCycleDiagnostics, or dicts.
            observations: List of Observation ORM objects or dicts.
            weather_records: Optional list of weather daily dicts (keys: date, tmax, tmin, etc.).

        Returns:
            FeatureVector containing categorized features and a flattened dictionary.
        """
        # 1. Temporal Leakage Filtering (Strictly <= as_of_date)
        filtered_daily = self._filter_daily_outputs(daily_outputs, as_of_date)
        filtered_assim = self._filter_assimilation_states(assimilation_states or [], as_of_date)
        filtered_obs = self._filter_observations(observations or [], as_of_date)
        filtered_weather = self._filter_weather_records(weather_records or [], as_of_date)

        # 2. Extract Current State at as_of_date
        current_record = filtered_daily[-1] if filtered_daily else None
        current_dvs = self._get_attr(current_record, "dvs")
        current_lai = self._get_attr(current_record, "lai")
        current_tagp = self._get_attr(current_record, "tagp")
        current_sm = self._get_attr(current_record, "sm")

        # 3. Compute Feature Categories
        growth_rates = self._compute_growth_rates(filtered_daily)
        water_stress = self._compute_water_stress(filtered_daily)
        thermal_stress = self._compute_thermal_stress(filtered_daily, filtered_weather)
        assim_diagnostics = self._compute_assimilation_diagnostics(filtered_assim)
        obs_quality = self._compute_observation_quality(filtered_obs, as_of_date)

        # 4. Construct FeatureVector
        feature_vector = FeatureVector(
            as_of_date=as_of_date,
            current_dvs=current_dvs,
            current_lai=current_lai,
            current_tagp=current_tagp,
            current_sm=current_sm,
            growth_rates=growth_rates,
            water_stress=water_stress,
            thermal_stress=thermal_stress,
            assimilation_diagnostics=assim_diagnostics,
            observation_quality=obs_quality,
        )

        # 5. Build Flattened Numerical Dictionary
        feature_vector.feature_flat_dict = self._flatten_feature_vector(feature_vector)
        return feature_vector

    # ── Helpers for Filtering & Attribute Access ─────────────────────────────

    def _get_attr(self, item: Any, attr: str, default: Any = None) -> Any:
        """Safe getter for ORM objects, Pydantic objects, or dictionaries."""
        if item is None:
            return default
        if isinstance(item, dict):
            return item.get(attr, default)
        return getattr(item, attr, default)

    def _parse_date(self, val: Any) -> Optional[datetime.date]:
        """Normalize datetime, date, or ISO string to datetime.date."""
        if val is None:
            return None
        if isinstance(val, datetime.datetime):
            return val.date()
        if isinstance(val, datetime.date):
            return val
        if isinstance(val, str):
            try:
                return datetime.datetime.fromisoformat(val).date()
            except ValueError:
                return None
        return None

    def _filter_daily_outputs(self, outputs: List[Any], cutoff: datetime.date) -> List[Any]:
        """Filter and sort daily outputs up to cutoff date."""
        valid = []
        for o in outputs:
            d = self._parse_date(self._get_attr(o, "date"))
            if d and d <= cutoff:
                valid.append((d, o))
        valid.sort(key=lambda x: x[0])
        return [o for _, o in valid]

    def _filter_assimilation_states(self, states: List[Any], cutoff: datetime.date) -> List[Any]:
        """Filter and sort assimilation states up to cutoff date."""
        valid = []
        for s in states:
            d = self._parse_date(
                self._get_attr(s, "assimilation_time")
                or self._get_attr(s, "cycle_date")
                or self._get_attr(s, "date")
            )
            if d and d <= cutoff:
                valid.append((d, s))
        valid.sort(key=lambda x: x[0])
        return [s for _, s in valid]

    def _filter_observations(self, obs_list: List[Any], cutoff: datetime.date) -> List[Any]:
        """Filter observations up to cutoff date."""
        valid = []
        for o in obs_list:
            d = self._parse_date(
                self._get_attr(o, "obs_date")
                or self._get_attr(o, "observation_time")
                or self._get_attr(o, "date")
            )
            if d and d <= cutoff:
                valid.append((d, o))
        return [o for _, o in valid]

    def _filter_weather_records(self, weather_list: List[Dict[str, Any]], cutoff: datetime.date) -> List[Dict[str, Any]]:
        """Filter weather records up to cutoff date."""
        valid = []
        for w in weather_list:
            d = self._parse_date(self._get_attr(w, "date"))
            if d and d <= cutoff:
                valid.append((d, w))
        valid.sort(key=lambda x: x[0])
        return [w for _, w in valid]

    # ── Feature Calculators ───────────────────────────────────────────────────

    def _compute_growth_rates(self, sorted_daily: List[Any]) -> GrowthRateFeatures:
        """Compute ΔLAI/Δt and ΔTAGP/Δt over 1-day and 7-day windows."""
        if not sorted_daily:
            return GrowthRateFeatures()

        latest = sorted_daily[-1]
        latest_lai = self._get_attr(latest, "lai")
        latest_tagp = self._get_attr(latest, "tagp")

        delta_lai_1d = None
        delta_lai_7d = None
        delta_tagp_1d = None
        delta_tagp_7d = None

        if len(sorted_daily) >= 2:
            prev_1d = sorted_daily[-2]
            prev_lai_1d = self._get_attr(prev_1d, "lai")
            prev_tagp_1d = self._get_attr(prev_1d, "tagp")
            
            d_t_1d = (self._parse_date(self._get_attr(latest, "date")) - self._parse_date(self._get_attr(prev_1d, "date"))).days
            if d_t_1d > 0:
                if latest_lai is not None and prev_lai_1d is not None:
                    delta_lai_1d = (latest_lai - prev_lai_1d) / d_t_1d
                if latest_tagp is not None and prev_tagp_1d is not None:
                    delta_tagp_1d = (latest_tagp - prev_tagp_1d) / d_t_1d

        if len(sorted_daily) >= 8:
            prev_7d = sorted_daily[-8]
            prev_lai_7d = self._get_attr(prev_7d, "lai")
            prev_tagp_7d = self._get_attr(prev_7d, "tagp")
            
            d_t_7d = (self._parse_date(self._get_attr(latest, "date")) - self._parse_date(self._get_attr(prev_7d, "date"))).days
            if d_t_7d > 0:
                if latest_lai is not None and prev_lai_7d is not None:
                    delta_lai_7d = (latest_lai - prev_lai_7d) / d_t_7d
                if latest_tagp is not None and prev_tagp_7d is not None:
                    delta_tagp_7d = (latest_tagp - prev_tagp_7d) / d_t_7d
        elif len(sorted_daily) > 1:
            prev_first = sorted_daily[0]
            prev_lai_f = self._get_attr(prev_first, "lai")
            prev_tagp_f = self._get_attr(prev_first, "tagp")
            d_t_f = (self._parse_date(self._get_attr(latest, "date")) - self._parse_date(self._get_attr(prev_first, "date"))).days
            if d_t_f > 0:
                if latest_lai is not None and prev_lai_f is not None:
                    delta_lai_7d = (latest_lai - prev_lai_f) / d_t_f
                if latest_tagp is not None and prev_tagp_f is not None:
                    delta_tagp_7d = (latest_tagp - prev_tagp_f) / d_t_f

        return GrowthRateFeatures(
            delta_lai_1d=delta_lai_1d,
            delta_lai_7d=delta_lai_7d,
            delta_tagp_1d=delta_tagp_1d,
            delta_tagp_7d=delta_tagp_7d,
        )

    def _compute_water_stress(self, sorted_daily: List[Any]) -> WaterStressFeatures:
        """Compute cumulative water stress and recent window averages."""
        if not sorted_daily:
            return WaterStressFeatures()

        cum_rftra_deficit = 0.0
        rftra_vals = []

        for record in sorted_daily:
            rftra = self._get_attr(record, "rftra")
            if rftra is not None:
                deficit = max(0.0, 1.0 - float(rftra))
                cum_rftra_deficit += deficit
                rftra_vals.append(float(rftra))

        mean_rftra_7d = None
        mean_rftra_14d = None

        if rftra_vals:
            recent_7 = rftra_vals[-7:]
            mean_rftra_7d = float(sum(recent_7) / len(recent_7))

            recent_14 = rftra_vals[-14:]
            mean_rftra_14d = float(sum(recent_14) / len(recent_14))

        latest_sm = self._get_attr(sorted_daily[-1], "sm")
        current_sm_deficit = None
        if latest_sm is not None:
            # Assuming field capacity ~0.35 cm3/cm3
            current_sm_deficit = max(0.0, 0.35 - float(latest_sm))

        return WaterStressFeatures(
            cumulative_rftra_deficit=cum_rftra_deficit,
            mean_rftra_7d=mean_rftra_7d,
            mean_rftra_14d=mean_rftra_14d,
            current_sm_deficit=current_sm_deficit,
        )

    def _compute_thermal_stress(
        self, sorted_daily: List[Any], sorted_weather: List[Dict[str, Any]]
    ) -> ThermalStressFeatures:
        """Compute heat/cold stress days and temperature dynamics from daily records or weather."""
        heat_days = 0
        cold_days = 0
        temp_ranges_7d = []
        tmaxs_7d = []

        # If explicit weather records are supplied
        if sorted_weather:
            for w in sorted_weather:
                tmax = self._get_attr(w, "tmax")
                tmin = self._get_attr(w, "tmin")

                if tmax is not None and float(tmax) > self.heat_threshold_c:
                    heat_days += 1
                if tmin is not None and float(tmin) < self.cold_threshold_c:
                    cold_days += 1

            recent_w7 = sorted_weather[-7:]
            for w in recent_w7:
                tmax = self._get_attr(w, "tmax")
                tmin = self._get_attr(w, "tmin")
                if tmax is not None:
                    tmaxs_7d.append(float(tmax))
                if tmax is not None and tmin is not None:
                    temp_ranges_7d.append(float(tmax) - float(tmin))

        mean_temp_range = float(sum(temp_ranges_7d) / len(temp_ranges_7d)) if temp_ranges_7d else None
        max_tmax = max(tmaxs_7d) if tmaxs_7d else None

        return ThermalStressFeatures(
            cumulative_heat_days=heat_days,
            cumulative_cold_days=cold_days,
            mean_temp_range_7d=mean_temp_range,
            max_tmax_7d=max_tmax,
        )

    def _compute_assimilation_diagnostics(self, sorted_assim: List[Any]) -> AssimilationDiagnosticFeatures:
        """Compute EnKF innovation statistics and ensemble spread metrics."""
        if not sorted_assim:
            return AssimilationDiagnosticFeatures()

        cycle_count = len(sorted_assim)
        latest = sorted_assim[-1]

        # Latest cycle innovations and spreads
        innov_latest = self._get_attr(latest, "innovation") or self._get_attr(latest, "innovation_vector") or {}
        prior_spread = self._get_attr(latest, "ensemble_spread_prior") or self._get_attr(latest, "ensemble_covariance") or {}
        post_spread = self._get_attr(latest, "posterior_spread") or {}
        update_mag = self._get_attr(latest, "state_update_magnitude") or {}

        # Accumulate mean innovation across cycles
        all_innovs: Dict[str, List[float]] = {}
        for s in sorted_assim:
            inv = self._get_attr(s, "innovation") or self._get_attr(s, "innovation_vector") or {}
            for k, v in inv.items():
                if v is not None and isinstance(v, (int, float)):
                    all_innovs.setdefault(k, []).append(float(v))

        mean_innovs = {k: float(sum(vals) / len(vals)) for k, vals in all_innovs.items() if vals}

        # Normalize dictionary numeric values cleanly
        def clean_dict(d: Any) -> Dict[str, float]:
            if not isinstance(d, dict):
                return {}
            res = {}
            for k, v in d.items():
                if v is not None and isinstance(v, (int, float)):
                    res[str(k).lower()] = float(v)
            return res

        return AssimilationDiagnosticFeatures(
            assimilation_cycles_count=cycle_count,
            mean_innovation=mean_innovs,
            latest_innovation=clean_dict(innov_latest),
            prior_spread=clean_dict(prior_spread),
            posterior_spread=clean_dict(post_spread),
            state_update_magnitude=clean_dict(update_mag),
        )

    def _compute_observation_quality(self, filtered_obs: List[Any], cutoff: datetime.date) -> ObservationQualityFeatures:
        """Compute observation counts, quality scores, sources present, and observation age."""
        if not filtered_obs:
            return ObservationQualityFeatures()

        total_count = len(filtered_obs)
        valid_count = 0
        rejected_count = 0
        quality_scores = []
        sources = set()
        latest_valid_date: Optional[datetime.date] = None

        for o in filtered_obs:
            status = str(self._get_attr(o, "status") or self._get_attr(o, "quality_flag") or "VALID").upper()
            source = str(self._get_attr(o, "source") or "UNKNOWN")
            sources.add(source)

            d = self._parse_date(
                self._get_attr(o, "obs_date")
                or self._get_attr(o, "observation_time")
                or self._get_attr(o, "date")
            )

            if status in ("VALID", "PASSED", "CLEAN"):
                valid_count += 1
                if d and (latest_valid_date is None or d > latest_valid_date):
                    latest_valid_date = d

                score = self._get_attr(o, "quality_score") or self._get_attr(o, "confidence_score")
                if score is not None:
                    quality_scores.append(float(score))
            else:
                rejected_count += 1

        mean_quality = float(sum(quality_scores) / len(quality_scores)) if quality_scores else None

        latest_age_days = None
        if latest_valid_date is not None:
            latest_age_days = float((cutoff - latest_valid_date).days)

        return ObservationQualityFeatures(
            total_obs_count=total_count,
            valid_obs_count=valid_count,
            rejected_obs_count=rejected_count,
            mean_quality_score=mean_quality,
            latest_obs_age_days=latest_age_days,
            obs_sources_present=sorted(list(sources)),
        )

    def _flatten_feature_vector(self, fv: FeatureVector) -> Dict[str, float]:
        """Flatten FeatureVector into a 1D scalar dictionary for analytics/tabular ML pipelines."""
        flat: Dict[str, float] = {}

        if fv.current_dvs is not None:
            flat["current_dvs"] = float(fv.current_dvs)
        if fv.current_lai is not None:
            flat["current_lai"] = float(fv.current_lai)
        if fv.current_tagp is not None:
            flat["current_tagp"] = float(fv.current_tagp)
        if fv.current_sm is not None:
            flat["current_sm"] = float(fv.current_sm)

        # Growth rates
        if fv.growth_rates.delta_lai_1d is not None:
            flat["delta_lai_1d"] = fv.growth_rates.delta_lai_1d
        if fv.growth_rates.delta_lai_7d is not None:
            flat["delta_lai_7d"] = fv.growth_rates.delta_lai_7d
        if fv.growth_rates.delta_tagp_1d is not None:
            flat["delta_tagp_1d"] = fv.growth_rates.delta_tagp_1d
        if fv.growth_rates.delta_tagp_7d is not None:
            flat["delta_tagp_7d"] = fv.growth_rates.delta_tagp_7d

        # Water stress
        flat["cum_rftra_deficit"] = fv.water_stress.cumulative_rftra_deficit
        if fv.water_stress.mean_rftra_7d is not None:
            flat["mean_rftra_7d"] = fv.water_stress.mean_rftra_7d
        if fv.water_stress.mean_rftra_14d is not None:
            flat["mean_rftra_14d"] = fv.water_stress.mean_rftra_14d
        if fv.water_stress.current_sm_deficit is not None:
            flat["current_sm_deficit"] = fv.water_stress.current_sm_deficit

        # Thermal stress
        flat["cum_heat_days"] = float(fv.thermal_stress.cumulative_heat_days)
        flat["cum_cold_days"] = float(fv.thermal_stress.cumulative_cold_days)
        if fv.thermal_stress.mean_temp_range_7d is not None:
            flat["mean_temp_range_7d"] = fv.thermal_stress.mean_temp_range_7d
        if fv.thermal_stress.max_tmax_7d is not None:
            flat["max_tmax_7d"] = fv.thermal_stress.max_tmax_7d

        # Assimilation diagnostics
        flat["assim_cycles_count"] = float(fv.assimilation_diagnostics.assimilation_cycles_count)
        for var_name, val in fv.assimilation_diagnostics.mean_innovation.items():
            flat[f"mean_innov_{var_name}"] = val
        for var_name, val in fv.assimilation_diagnostics.latest_innovation.items():
            flat[f"latest_innov_{var_name}"] = val
        for var_name, val in fv.assimilation_diagnostics.prior_spread.items():
            flat[f"prior_spread_{var_name}"] = val
        for var_name, val in fv.assimilation_diagnostics.posterior_spread.items():
            flat[f"post_spread_{var_name}"] = val

        # Observation quality
        flat["total_obs_count"] = float(fv.observation_quality.total_obs_count)
        flat["valid_obs_count"] = float(fv.observation_quality.valid_obs_count)
        flat["rejected_obs_count"] = float(fv.observation_quality.rejected_obs_count)
        if fv.observation_quality.mean_quality_score is not None:
            flat["mean_obs_quality"] = fv.observation_quality.mean_quality_score
        if fv.observation_quality.latest_obs_age_days is not None:
            flat["latest_obs_age_days"] = fv.observation_quality.latest_obs_age_days

        return flat
