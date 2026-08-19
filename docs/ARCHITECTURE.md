# 🏛️ AgriTwin System Architecture & 6-Layer Reference

## 1. Architectural Overview

AgriTwin is structured as a modular, 6-layer digital twin and machine learning platform. It reconciles deterministic biophysical crop modeling with stochastic remote sensing observations and machine learning residual correction.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 6: PRESENTATION & DECISION SUPPORT                                    │
│ • FastAPI HTTP Endpoints • Interactive Swagger / ReDoc • Scenario Sweeping  │
└──────────────────────────────────────▲──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────┐
│ LAYER 5: POST-PROCESSING BIAS CORRECTION & UNCERTAINTY QUANTIFICATION       │
│ • Stacked Ensemble (XGBoost + LightGBM + MLP + Ridge)                       │
│ • Deep Gaussian Process with Spatiotemporal Kernel RBF(x,y) × RBF(t)        │
└──────────────────────────────────────▲──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────┐
│ LAYER 4: DATA ASSIMILATION & STATE TRACKING                                 │
│ • Closed-Loop Ensemble Kalman Filter (EnKF)                                 │
│ • Stochastic Perturbations & Dynamic Kalman Gain (K) Updates                │
└──────────────────────────────────────▲──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────┐
│ LAYER 3: DATA FUSION & OBSERVATION INGESTION                                │
│ • Sentinel-2 MSI NDRE/NDVI (10m) • Farmer Smartphone GRVI (5-point W-Shape) │
│ • Temporal Gap-Filling (Cloud Inpainting) • Spatial Resolution Alignment    │
└──────────────────────────────────────▲──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────┐
│ LAYER 2: BIOPHYSICAL CROP SIMULATION ENGINE                                 │
│ • WOFOST 7.2 (PCSE): Carbon, Phenology (DVS), Transpiration, Grain Filling  │
│ • AgroManagement: Timed & Soil Moisture-Triggered Irrigation Applications    │
└──────────────────────────────────────▲──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────┐
│ LAYER 1: DATA INGESTION, SOIL HYDROLOGY & PERSISTENCE                       │
│ • ERA5-Land & NASA POWER Weather APIs • ISRIC SoilGrids REST API            │
│ • SQLAlchemy ORM & PostgreSQL / SQLite Relational Schema                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer-by-Layer Detailed Breakdown

### Layer 1: Environmental Data Ingestion & Storage
- **NASA POWER API**: Ingests daily operational meteorological variables:
  - Minimum/Maximum Temperature ($T_{\min}, T_{\max}$) [°C]
  - Solar Radiation ($R_{ss}$) [$\text{kJ/m}^2/\text{day}$]
  - Vapor Pressure ($e_a$) [$\text{kPa}$]
  - Wind Speed at 2m ($u_2$) [$\text{m/s}$]
  - Precipitation ($P$) [$\text{mm/day}$]
- **ERA5-Land Reanalysis**: High-resolution (0.1°) historical climate records (1997–present) and 4-layer volumetric soil moisture ($0\text{--}7\text{cm}$, $7\text{--}28\text{cm}$, $28\text{--}100\text{cm}$, $100\text{--}289\text{cm}$).
- **SoilGrids 250m API**: Fetches physical and hydraulic soil properties (clay %, sand %, silt %, bulk density, organic carbon), converted into Van Genuchten parameters for PCSE's `WaterbalanceFD`.
- **Relational Storage**: Managed via SQLAlchemy ORM with support for both SQLite (local development) and PostgreSQL (production).

### Layer 2: Biophysical Crop Simulation Engine
- **WOFOST 7.2 Core**: Implemented via PCSE (`pcse.models.Wofost72_WLP_CWB`).
- **Phenological Stages**: Tracks Development Stage (`DVS`):
  - $\text{DVS} = 0.0$: Emergence / Sowing
  - $\text{DVS} = 1.0$: Anthesis / Flowering (maximum drought sensitivity)
  - $\text{DVS} = 2.0$: Physiological Maturity / Harvest
- **Carbon Assimilation & Partitioning**: Computes daily gross $CO_2$ assimilation, maintenance respiration, and partitions net assimilate among Leaves (`TWLV`), Stems (`TWST`), Roots (`TWRT`), and Storage Organs (`TWSO`).

### Layer 3: Multi-Source Data Fusion & Ingestion
- **Sentinel-2 Satellite Pipeline**: Automatically queries Sentinel-2 Level-2A imagery, filters cloud contamination, and extracts Red Edge ($B_5, B_6, B_7$) and NIR ($B_8$) bands for NDRE computation. Inverts NDRE to Leaf Area Index (LAI).
- **W-Shape Smartphone Scouting**: Computes Green-Red Vegetation Index ($\text{GRVI} = \frac{G - R}{G + R}$) across 5 spatial nodes in a field to provide ground reality checks during cloudy monsoon periods.
- **Temporal Interpolation**: Cubic spline and Gaussian Process smoothing recover trajectory continuity across cloud gaps, flagging data points with explicit uncertainty bounds.

### Layer 4: Sequential Data Assimilation (EnKF)
- **State Vector ($\mathbf{x}$)**: Comprises dynamically updated state variables:
  $$\mathbf{x} = [\text{LAI}, \text{SM}, \text{TAGP}, \text{TWSO}]^T$$
- **Ensemble Representation**: Maintains $N$ stochastic ensemble members ($N=25\text{--}50$) with perturbed weather parameters and initial states.
- **Closed-Loop Feedback**: Computes sample covariance $\mathbf{P}^f$ and optimal Kalman Gain $\mathbf{K}$ at each observation timestamp to update ensemble states:
  $$\mathbf{K} = \mathbf{P}^f \mathbf{H}^T (\mathbf{H} \mathbf{P}^f \mathbf{H}^T + \mathbf{R})^{-1}$$
  $$\mathbf{x}_i^a = \mathbf{x}_i^f + \mathbf{K} (\mathbf{y}_i - \mathbf{H}\mathbf{x}_i^f)$$

### Layer 5: ML Bias Correction & Uncertainty Quantification
- **Stacked ML Ensemble**: Combines four distinct model families:
  1. **XGBoost Regressor**: Captures sharp non-linear interactions.
  2. **LightGBM Regressor**: Fast gradient boosted trees with leaf-wise expansion.
  3. **Multi-Layer Perceptron (MLP)**: Deep non-linear feature representation.
  4. **Ridge Meta-Model**: Calibrated linear weighting over base model predictions.
- **Time-Series Cross-Validation**: `TimeSeriesSplit(n_splits=5)` guarantees zero look-ahead data leakage.
- **Deep Gaussian Process Layer**: Models spatial and temporal residual correlation:
  $$k((x, y, t), (x', y', t')) = k_{\text{spatial}}(x, y; x', y') \times k_{\text{temporal}}(t, t')$$

### Layer 6: Farmer Decision Support & Advisory Engine
- **Crop Recommendation & Profit Engine**: Ranks crops by projected net return (₹/acre) using MSP benchmarks, soil suitability, and seasonal calendars.
- **Daily Actionable Advisories**:
  - 💧 **Irrigation Trigger**: Evaluates root-zone soil moisture deficit against field capacity (`SMFCF`) & wilting point (`SMW`) + 48h rain forecast.
  - 🚨 **Nitrogen Top-Dressing**: Translates smartphone GRVI and satellite NDRE chlorosis into Urea dosage (kg/acre).
  - 🌧️ **Extreme Weather Alerts**: Scans temperature ($T_{\max} > 36^\circ\text{C}$ heat stress, frost) and heavy precipitation.
- **Bilingual Delivery**: Directly outputs human-readable Hindi (`hi`) and English (`en`) formatted cards for WhatsApp, SMS, and mobile UI.

---

## 3. Lifecycle Execution Matrix

| Scenario / Trigger | Modules Run | Agronomic Objective & Delivery |
|---|---|---|
| **Pre-Season Planning** | `CropRecommendationService` + `SoilService` | Suggests most profitable crop based on soil, climate & MSP. |
| **Sowing Day** | `WOFOST 7.2` (Initial Run) | Establishes baseline trajectory and phenology stages. |
| **Every 5 Days** | `Sentinel-2 Fetcher` $\to$ `EnKF` | Assimilates satellite LAI/NDRE to correct growth trajectory. |
| **Farmer Submits Photos** | `ScoutSessionProcessor` (GRVI) $\to$ `EnKF` | Detects leaf yellowing/N-stress with 30% "Gentle Nudge" error. |
| **Daily Routine** | `AdvisoryService` + `WeatherService` | Checks soil moisture and weather hazards; advises watering/fertilizing. |
| **Harvest / End of Season** | `BiasCorrectionService` + `OutputGenerator` | Delivers calibrated yield (kg/ha & Q/acre), ± error margin, and historical delta. |

---

## 4. End-to-End Data Flow Diagram

```mermaid
flowchart TD
    subgraph Data Sources
        WP[NASA POWER / ERA5-Land]
        SP[SoilGrids API]
        SAT[Sentinel-2 Satellite]
        PH[Farmer Phone GRVI Photos]
    end

    subgraph Core Physics & Assimilation
        WOFOST[WOFOST 7.2 Simulation]
        FUSION[Data Fusion Pipeline]
        ENKF[EnKF State Assimilation]
    end

    subgraph ML Bias Correction
        STACK[Stacked ML Ensemble]
        GP[Deep Gaussian Process]
    end

    subgraph Decision Support & Output
        REC[Crop Recommendation Engine]
        ADV[Advisory & Alert Engine]
        API[FastAPI Endpoints / WhatsApp Card]
    end

    WP -->|Weather Grid| WOFOST
    SP -->|Soil Parameters| WOFOST
    SP -->|Soil Texture| REC
    SAT -->|NDRE / LAI| FUSION
    PH -->|GRVI Index (W-Shape)| FUSION
    PH -->|Chlorosis Flag| ADV
    FUSION -->|Observation Vector y, R| ENKF
    WOFOST -->|Prior State x_f| ENKF
    ENKF -->|Posterior State x_a| STACK
    STACK -->|Ensemble Prediction| GP
    GP -->|Corrected Yield & Uncertainty| ADV
    WOFOST -->|Daily SM & RFTRA| ADV
    REC -->|Profit Ranking| API
    ADV -->|Bilingual Alerts & Summaries| API
```
