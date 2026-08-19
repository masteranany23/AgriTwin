# 📖 AgriTwin Master Project Record Book & Specification

---

## 1. Project Overview

### 1.1 Project Description
**AgriTwin** is a modular, physics-guided AI Digital Twin Platform for precision agriculture that creates a continuously synchronized virtual representation of real farms. The platform bridges deterministic biophysical crop growth modeling with stochastic remote sensing observations, smartphone-based RGB scouting, and machine learning residual correction to monitor, predict, and optimize agricultural operations. It is engineered for both technology-enabled commercial farms and resource-constrained smallholder farms (e.g. typical 2.5-acre holdings in India).

### 1.2 Vision
To develop an open, modular, explainable, and research-driven agricultural digital twin platform that enables every farmer to make data-driven decisions regardless of whether physical IoT sensors are available on their field.

### 1.3 Objectives
- **Real-Time Virtual Representation**: Maintain high-fidelity biophysical state tracking (Leaf Area Index, Soil Moisture, Total Above-Ground Biomass, Storage Organ Yield).
- **Zero-IoT Smallholder Accessibility**: Provide full functionality without requiring costly in-situ soil probes or weather stations through remote sensing and smartphone scouting.
- **Physics-Guided Machine Learning**: Pair process-based crop simulation (WOFOST) with Ensemble Kalman Filtering (EnKF) and Stacked Machine Learning bias correctors.
- **Explainable Farmer Decision Support**: Generate actionable, bilingual (Hindi & English) advisories for irrigation, fertilizer, and weather risk management.
- **Multi-Source Data Fusion**: Seamlessly combine Sentinel-2 optical, Sentinel-1 radar, ERA5-Land reanalysis, SoilGrids, and smartphone RGB images under monsoon cloud cover.

### 1.4 Core Architectural Pillars
```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 6: PRESENTATION, ADVISORY & DECISION SUPPORT                               │
│ • Bilingual Advisory Engine (Hindi/English) • Crop Recommendation & MSP Profit   │
│ • FastAPI REST API & Interactive Endpoints • 3D Procedural Visualization         │
└────────────────────────────────────────▲─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┴─────────────────────────────────────────┐
│ LAYER 5: POST-PROCESSING BIAS CORRECTION & ML RESIDUAL LEARNING                  │
│ • Stacked Ensemble (XGBoost + LightGBM + MLP + Ridge Meta-Model)                 │
│ • Deep Spatiotemporal Gaussian Process: Kernel RBF(x,y) × RBF(t)                 │
└────────────────────────────────────────▲─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┴─────────────────────────────────────────┐
│ LAYER 4: DATA ASSIMILATION & DYNAMIC STATE TRACKING                              │
│ • Closed-Loop Ensemble Kalman Filter (EnKF) with N=25..50 Members                │
│ • Stochastic Perturbation & Dynamic Kalman Gain Update: x_a = x_f + K(y - Hx_f)  │
└────────────────────────────────────────▲─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┴─────────────────────────────────────────┐
│ LAYER 3: MULTI-SOURCE DATA FUSION & OBSERVATION INGESTION                        │
│ • Sentinel-2 MSI (10m NDRE/LAI) • Smartphone W-Shape Scouting (GRVI)             │
│ • Temporal Spline / Savitzky-Golay Gap Filling • Monsoon-Aware HOLD_OPEN_LOOP    │
└────────────────────────────────────────▲─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┴─────────────────────────────────────────┐
│ LAYER 2: BIOPHYSICAL CROP SIMULATION ENGINE                                      │
│ • WOFOST 7.2 (PCSE): Carbon Balance, Phenology (DVS 0..2), WaterbalanceFD        │
│ • AgroManagement: Timed & Soil Moisture Deficit Irrigations                      │
└────────────────────────────────────────▲─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┴─────────────────────────────────────────┐
│ LAYER 1: DATA INGESTION, SOIL HYDROLOGY & RELATIONAL PERSISTENCE                 │
│ • ERA5-Land Reanalysis (4-layer SM) • NASA POWER API • ISRIC SoilGrids 250m      │
│ • SQLAlchemy ORM with PostgreSQL / SQLite Database Schema                        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Comprehensive Module Specifications & Implementation Status

### 3.1 Farm Management Module
- **Objective**: Maintain hierarchical digital representations of farms, fields, seasons, crops, machinery, and farmer management events.
- **Technologies**: PostgreSQL, PostGIS, FastAPI, SQLAlchemy ORM, Alembic.
- **Methodology**: Hierarchical `Farm → Field → Season → Crop` domain models with event-driven updates.
- **Submodules**:
  - `Farm Management`: Spatial boundaries, farm metadata, ownership.
  - `Field Management`: Field polygons, elevation, spatial centroids, SoilGrids associations.
  - `Crop Management`: Crop types, PCSE variety mappings, baseline physiological parameters.
  - `Season Management`: Kharif, Rabi, Zaid seasonal bounds, sowing/harvest dates.
  - `Farmer Activities`: Logging irrigation, fertilization, scouting, and tillage activities.
- **Status**: **Completed** (`backend/app/models/`, `backend/app/api/routes/`)

---

### 3.2 Observation Layer
- **Objective**: Ingest heterogeneous, multi-modal observation streams and standardize them into digital twin observation containers.
- **Technologies**: Sentinel-2 L2A API, NASA POWER, ERA5-Land, SoilGrids REST, FastAPI.
- **Methodology**: Ingestion with timestamp synchronization, metadata validation, and coordinate snapping.
- **Submodules**:
  - `Satellite Observation`: Automated Sentinel-2 Level-2A cloud filtering and NDRE/LAI extraction. *(Completed)*
  - `Weather Data`: Real-time NASA POWER operational API + historical ERA5-Land reanalysis with 4-layer soil moisture. *(Completed)*
  - `Soil Data`: ISRIC SoilGrids 250m queries with Van Genuchten hydraulic parameter derivation. *(Completed)*
  - `Smartphone Images`: 5-node W-shape field scouting with localized RGB color analysis. *(Completed)*
  - `Historical Records`: Kaggle 1997–2020 Indian district-level crop production records. *(Completed)*
  - `IoT & Drone Observation`: Evaluated and scoped for future phases to keep platform accessible for resource-constrained smallholders.
- **Status**: **Completed** (`backend/app/satellite/`, `backend/app/data_sources/`)

---

### 3.3 Data Fusion Module
- **Objective**: Reconstruct the true continuous state of the farm (LAI, Soil Moisture) by fusing multi-scale, asynchronous observations while accounting for cloud gaps and observational uncertainties.
- **Technologies**: EnKF, PyProj, SciPy Splines, Savitzky-Golay, NumPy, Pandas.
- **Functional Submodules**:
  1. **Observation Validation**:
     - *Physical Bounds Checking*: Enforces strict biophysical validity (e.g. $0 \le \text{LAI} \le 10$, $0 \le \text{SM} \le \text{Porosity}$).
     - *Quality Control Masks*: Evaluates Sentinel-2 cloud masks and bitmasks to discard corrupted data frames.
  2. **Temporal Alignment (Monsoon-Aware)**:
     - *Linear Interpolation*: Applied for short gaps (1–3 days) during linear vegetative growth.
     - *Cubic Spline Interpolation*: Applied for medium gaps (4–10 days) to capture developmental curvature.
     - *Savitzky-Golay Smoothing*: Sliding polynomial filter (odd window lengths $w=5,7,11$) to remove high-frequency atmospheric noise.
     - *Monsoon Cloud Gap Exception*: If gaps exceed `max_allowed_gap_days` (10 days), interpolation halts, outputs `NaN`, and triggers a `HOLD_OPEN_LOOP` flag preventing Kalman filter divergence.
  3. **Spatial Alignment (Unified Grid Mapping)**:
     - *Cartographic Transformation*: Re-projects coordinates between global WGS84 (`EPSG:4326`) and local UTM zones via `pyproj`.
     - *Point-to-Grid Proximity Snapping*: Snaps smartphone GPS points to the center of corresponding field grid cells.
  4. **Confidence Estimation & Error Mapping**:
     - Dynamically computes confidence scores that map inversely to the Observation Error Covariance Matrix ($R$):
       $$\text{High Confidence} \implies \text{Low } R \implies \text{High Kalman Gain } K$$
  5. **Multi-Source Fusion Optimization Engine**:
     - Dynamically shifts observation weights across modalities based on cloud cover:
       $$\text{Cloud Cover} < 40\% \implies \text{Prioritize Optical Sentinel-2 (10m)}$$
       $$40\% \le \text{Cloud Cover} < 70\% \implies \text{Blend Optical, Radar, and Ground Photos}$$
       $$\text{Cloud Cover} \ge 70\% \implies \text{Switch to Sentinel-1 Radar (RVI) + Smartphone GRVI}$$
     - Post-weight normalization ensures $\sum w_i = 1.0$.
- **Status**: **Completed** (`backend/app/services/temporal_interpolation_service.py`, `spatial_alignment_service.py`, `confidence_estimator.py`, `multi_source_fusion_service.py`, `data_fusion_pipeline.py`)

---

### 3.4 Digital Twin Core
- **Objective**: Maintain the real-time virtual state of each field and synchronize updates across simulation, data assimilation, and decision engines.
- **Technologies**: Python, FastAPI, SQLAlchemy, Redis Streams.
- **Methodology**: Central Twin State updated through state transitions and chronological persistence.
- **Submodules**:
  - `Twin State Manager`: In-memory and persistent state tracking (`FieldState`).
  - `State Synchronization`: Reconciling simulation runs with new observational events.
  - `State History & Versioning`: Tracking historical state evolution over seasons.
  - `Uncertainty Manager`: Propagating variance and EnKF covariance matrices.
- **Status**: **In Progress / State Logic Implemented** (`backend/app/twin/field_state.py`)

---

### 3.5 Physics Simulation Module
- **Objective**: Simulate process-based crop growth and soil water balance on a daily timestep under variable weather and management regimes.
- **Technologies**: PCSE (Python Crop Simulation Environment), WOFOST 7.2 (`Wofost72_WLP_FD`).
- **Methodology**: Daily mechanistic carbon assimilation, respiration, phenological development (`DVS`), and multi-layer soil hydrology.
- **Submodules**:
  - `WOFOST Engine`: Core biophysical crop simulation.
  - `Phenology Tracker`: Tracks Emergence ($\text{DVS}=0$), Anthesis ($\text{DVS}=1.0$), Maturity ($\text{DVS}=2.0$).
  - `Biomass Partitioning`: Leaves (`TWLV`), Stems (`TWST`), Roots (`TWRT`), Grains (`TWSO`).
  - `Soil Water Balance`: `WaterbalanceFD` with infiltration, percolation, and root-zone water extraction.
  - `Scenario Engine`: Evaluates what-if irrigation scenarios (rainfed vs. deficit irrigation vs. optimal scheduling).
- **Status**: **Completed** (`backend/app/simulation/engine.py`, `backend/app/scenario/`)

---

### 3.6 Data Assimilation Module
- **Objective**: Eliminate simulation drift by sequentially updating biophysical state variables with real-world observations via an Ensemble Kalman Filter.
- **Technologies**: EnKF (Ensemble Kalman Filter), NumPy, SciPy.
- **Methodology**: Monte Carlo state ensemble propagation with stochastic perturbations and optimal Kalman Gain state updates.
- **Mathematical Framework**:
  - State Vector: $\mathbf{x} = [\text{LAI}, \text{SM}, \text{TAGP}, \text{TWSO}]^T$
  - Forecast Ensemble: $N=25\text{--}50$ members with perturbed weather and soil properties.
  - Kalman Gain Update:
    $$\mathbf{K} = \mathbf{P}^f \mathbf{H}^T (\mathbf{H} \mathbf{P}^f \mathbf{H}^T + \mathbf{R})^{-1}$$
    $$\mathbf{x}_i^a = \mathbf{x}_i^f + \mathbf{K} (\mathbf{y}_i - \mathbf{H}\mathbf{x}_i^f)$$
- **Submodules**:
  - `Ensemble Manager`: Initializes and perturbs state ensemble.
  - `Observation Operator`: Maps model state space to observation space ($H$).
  - `State Updater`: Injects updated state vectors back into PCSE model instances.
  - `Variance Reduction Tracker`: Measures uncertainty reduction post-assimilation.
- **Status**: **Completed** (`backend/app/assimilation/`)

---

### 3.7 Bias Correction Module
- **Objective**: Learn and correct systematic biophysical simulation residuals using stacked machine learning models trained on long-term historical yields.
- **Technologies**: XGBoost, LightGBM, Multi-Layer Perceptron (MLP), Ridge Meta-Regressor, Deep Gaussian Processes (GPyTorch).
- **Methodology**: Residual learning ($y_{\text{actual}} - y_{\text{wofost}}$) with TimeSeriesSplit validation (zero lookahead leakage).
- **Submodules**:
  - `Stacked Ensemble Trainer`: Multi-model architecture with Ridge blending (`backend/scripts/stacked_ensemble_trainer.py`).
  - `Spatiotemporal GP Kernel`: Models spatially and temporally correlated residuals:
    $$k((x, y, t), (x', y', t')) = k_{\text{spatial}}(x, y; x', y') \times k_{\text{temporal}}(t, t')$$
  - `Training Data Generator`: High-throughput pipeline linking ERA5-Land daily weather, WOFOST simulations, and 127k+ Kaggle yield records (`scripts/generate_training_data.py`).
  - `Real-Time Inference Service`: Production bias corrector for real-time predictions (`backend/app/services/bias_correction_service.py`).
- **Status**: **Completed**

---

### 3.8 Computer Vision & 3.9 RGB Analytics Modules
- **Objective**: Extract biophysical, nitrogen, and chlorosis parameters from smartphone RGB imagery taken during field scouting without multispectral hardware.
- **Technologies**: OpenCV, NumPy, Scikit-learn.
- **Supported Vegetation Indices**:
  - $\text{GRVI} = \frac{G - R}{G + R}$ (Green-Red Vegetation Index)
  - $\text{VARI} = \frac{G - R}{G + R - B}$ (Visual Atmospheric Resistance Index)
  - $\text{GLI} = \frac{2G - R - B}{2G + R + B}$ (Green Leaf Index)
  - $\text{ExG} = 2G - R - B$ (Excess Green)
  - $\text{ExGR} = \text{ExG} - (1.4R - G)$ (Excess Green Minus Excess Red)
  - $\text{CIVE} = 0.441R - 0.811G + 0.385B + 18.78745$ (Color Index of Vegetation)
- **Submodules**:
  - `W-Shape Scouting Protocol`: 5-point field traversal sampling.
  - `Chlorosis & Nitrogen Deficit Estimator`: Converts GRVI depression into Urea top-dressing recommendations.
- **Status**: **Completed** (`backend/app/services/advisory_service.py`, `backend/app/services/scout_session_service.py`)

---

### 3.10 Remote Sensing Module
- **Objective**: Ingest, cloud-filter, and convert Sentinel-2 multispectral imagery into Leaf Area Index (LAI) and Normalized Difference Red Edge (NDRE) time-series.
- **Technologies**: Sentinel-2 MSI L2A, Rasterio, NumPy, SciPy.
- **Methodology**: Red Edge Band Inversion ($B_5, B_6, B_7$) and NIR ($B_8$) with Modified Beer-Lambert Law:
  $$\text{NDRE} = \frac{B_8 - B_5}{B_8 + B_5}$$
  $$\text{LAI} = -\frac{1}{k} \ln\left(1 - \frac{\text{NDRE}}{\text{NDRE}_{\max}}\right)$$
- **Status**: **Completed** (`backend/app/satellite/`)

---

### 3.11 Soil Intelligence Module
- **Objective**: Retrieve soil physical, chemical, and hydraulic parameters to configure WOFOST water retention curves.
- **Technologies**: ISRIC SoilGrids 250m REST API, Pedotransfer Functions.
- **Parameters Derived**: Clay %, Silt %, Sand %, Bulk Density, Field Capacity (`SMFCF`), Wilting Point (`SMW`), Total Porosity (`SM0`), Saturated Hydraulic Conductivity (`K0`).
- **Status**: **Completed** (`backend/app/data_sources/soil_source.py`)

---

### 3.12 Weather Intelligence Module
- **Objective**: Supply high-resolution historical climate reanalysis and operational forecast weather grids.
- **Technologies**: Copernicus CDS API (`reanalysis-era5-land`), NASA POWER REST API.
- **Features**:
  - Operational NASA POWER daily weather client.
  - ERA5-Land automated downloader with 15-day chunking to respect Copernicus 12,000-unit cost limits (`scripts/download_era5_daily.py`).
  - 4-layer volumetric soil moisture extraction ($0\text{--}7\text{cm}, 7\text{--}28\text{cm}, 28\text{--}100\text{cm}, 100\text{--}289\text{cm}$).
- **Status**: **Completed** (`backend/app/data_sources/era5_land_weather_source.py`, `scripts/download_era5_daily.py`)

---

### 3.13 Decision Intelligence & Advisory Module
- **Objective**: Translate digital twin biophysical states, weather forecasts, and soil moisture balances into actionable, high-impact farmer advisories.
- **Technologies**: Python, FastAPI, NumPy.
- **Features**:
  1. **Irrigation Trigger**: Evaluates root-zone soil moisture deficit against field capacity (`SMFCF`) & wilting point (`SMW`) combined with 48-hour precipitation forecasts.
  2. **Nitrogen Top-Dressing**: Translates smartphone GRVI and satellite NDRE chlorosis into Urea dosage (kg/acre).
  3. **Extreme Weather Alerts**: Scans $T_{\max} > 36^\circ\text{C}$ (heat stress during flowering $\text{DVS}=1.0$), frost risk, and heavy rainfall ($>35\text{mm}$).
  4. **Crop Recommendation & Profit Optimization**: Ranks alternative crops by projected net return (₹/acre) using government Minimum Support Prices (MSP), regional historical yields, and soil suitability.
  5. **Bilingual Delivery**: Generates structured, human-readable advice cards in **Hindi (`hi`)** and **English (`en`)**.
- **Status**: **Completed** (`backend/app/services/advisory_service.py`, `crop_recommendation_service.py`)

---

### 3.14 to 3.18 Extended Platform Modules

| Module | Core Functionality | Technologies | Status |
|---|---|---|---|
| **3.14 Knowledge Graph** | Relational graph linking crops, soil types, disease vectors, and agromanagement rules | NetworkX, JSON Graph Data | **In Progress** |
| **3.15 AI Assistant** | Context-aware agricultural assistant powered by Twin State context | Tool-Calling Agent / FastAPI | **In Progress** |
| **3.16 Visualization** | REST endpoints for time-series charts, assimilation replays, and twin states | FastAPI, JSON Schema, GeoJSON | **Completed (API Layer)** |
| **3.17 3D Digital Twin** | Procedural 3D farm canopy, weather effects, and soil moisture slice rendering | Three.js / React Three Fiber specs | **Design Phase** |
| **3.18 Infrastructure** | Containerized backend, automated DB migrations, comprehensive unit test suite | FastAPI, SQLite/PostgreSQL, Alembic, Pytest | **Completed** |

---

## 3. Verification & Quality Matrix

| Test Category | Test File | Key Validations | Status |
|---|---|---|---|
| **Data Assimilation** | `tests/test_enkf.py`, `test_assimilation_service.py` | Kalman gain computation, state covariance reduction, ensemble bounds | `PASSED` |
| **Simulation Engine** | `tests/test_irrigation.py`, `test_scenario_api.py` | WOFOST 7.2 water balance, phenology transitions, yield computation | `PASSED` |
| **Data Fusion** | `tests/test_data_sources.py` | Observation validation, temporal spline interpolation, spatial snapping | `PASSED` |
| **Advisory & Crop Rec** | `tests/test_advisory_service.py` | Irrigation triggers, urea dosage, Hindi/English cards, MSP profit ranking | `PASSED` |
| **Scout Session & RGB**| `tests/test_scout_session_processing.py` | 5-point W-shape parsing, GRVI calculation, confidence estimation | `PASSED` |
| **API Endpoints** | `tests/test_assimilation_api.py`, `test_farmer_workflow.py` | Full end-to-end user workflow, REST request/response schemas | `PASSED` |

---

## 4. Current Workstream & Next Milestones

1. **ERA5-Land Ingestion**: Complete background reanalysis downloads for historical training (`scripts/download_era5_daily.py`).
2. **Bias Corrector Calibration**: Run `scripts/generate_training_data.py` across 127k+ cleaned Kaggle district records and train final stacked models.
3. **Frontend Dashboard Integration**: Connect Layer 6 FastAPI advisory endpoints with the client visualization interface.
