**🌾 AgriTwin — Agricultural Digital Twin Platform**

> **A modular, physics-guided AI Digital Twin Platform for precision agriculture. Fuses process-based crop simulation (WOFOST), multi-source Earth observation data, and sequential data assimilation (EnKF) to create a continuously self-correcting virtual replica of crop fields.**


## 🌟 Overview & Vision

### What Is AgriTwin?

Imagine having a **digital mirror of your farm field** on your computer. Just like a navigation app predicts traffic using live road data, AgriTwin predicts **crop growth, water needs, and potential yields** by combining:

1. **🔬 Physics-Based Science** — The WOFOST crop growth model simulates daily phenology, water balance, and dry matter accumulation.
2. **🛰️ Real-World Observations** — Sentinel-2 satellites (NDRE/LAI) and smartphone photos (GRVI) provide reality checks.
3. **🧠 Self-Correcting AI** — The Ensemble Kalman Filter (EnKF) continuously nudges the virtual model back to reality without overreacting to noisy data.

### Who Is This For?

| **Audience** | **How They Use AgriTwin** |
|-------------|---------------------------|
| **Researchers** | Validate new crop models, test assimilation strategies, analyze 75-year back-casts. |
| **Developers** | Extend modules (new data sources, ML models, visualization layers). |
| **Agronomists / FPOs** | Run "what-if" scenarios (sowing dates, irrigation plans) for advisory services. |
| **Farmers** | Access predictions via the UI (smartphone scout sessions, yield forecasts). |

---

## 🧬 The Research Foundation

AgriTwin is built on **peer-reviewed science** adapted for smallholder agriculture. The table below summarizes the key research pillars and their implementation status.

### Core Scientific Decisions

| **Research Pillar** | **Source** | **Decision** | **Implementation** | **Key Result** |
|---------------------|------------|--------------|-------------------|----------------|
| **Option A: Multi-Model Ensemble** | *J. Hydrology* (2025), AERU (2010) | ❌ **Rejected** | DSSAT/AquaCrop require MCMC calibration & 40+ genetic coefficients. 0% adoption in smallholder contexts. | N/A |
| **Option B: E-WOFOST (4-layer SM)** | *Computers & Electronics in Ag.* (2025) | ✅ **Adopted** | ERA5-Land provides 4-layer soil moisture (0-7, 7-28, 28-100, 100-289 cm). Joint LAI+SM assimilation. | **R² = 0.85–0.90**, RMSE = 441–741 kg/ha |
| **Option C: NDRE (Satellite)** | Sentinel-2 Red Edge bands | ✅ **Adopted** | `NDRE = (NIR - RedEdge) / (NIR + RedEdge)` using B08 & B07. Confidence scoring (0.85 clear / 0.0 cloudy). | Detects N-stress without saturating at high LAI |
| **Option D: GRVI (Smartphone)** | *Plants* (2024) | ✅ **Adopted** | `GRVI = (Green - Red) / (Green + Red)`. W-Shape protocol (5 photos). **30% observation error** ("Gentle Nudge"). | R² > 0.85 with SPAD. Replaces ₹40,000 chlorophyll meters. |
| **Option E: ERA5-Land** | *ESSD* (2021) | ✅ **Adopted (Hybrid)** | ERA5-Land (>60 days) + NASA POWER (<60 days & forecasts). | 9 km, hourly, 4-layer SM, 1950+ coverage |
| **EnKF (Data Assimilation)** | Ensemble Kalman Filter | ✅ **Adopted** | 25-member ensemble. Matrix Kalman Gain (covariance of LAI, SM, WLV, WST, WRT, WSO). | Prevents model drift, corrects input biases |
| **XGBoost Bias Correction** | *MDPI Agriculture* (2025) | ✅ **Adopted** | 75-year ERA5-Land back-cast (1950–2025). Predicts WOFOST residual errors. | County-level: r = 0.659, RMSE = 578 kg/ha |

---

### 🇮🇳 India-Specific Research Adaptations

AgriTwin is designed for **Indian smallholder farmers** (average 2.5 acres, ₹15,000–₹20,000 annual income). These adaptations are *research-driven* to overcome local constraints:

| **Challenge** | **Technical Adaptation** | **Implementation** | **Research Basis** |
|---------------|-------------------------|-------------------|-------------------|
| **Monsoon Cloud Cover** (>70% Jun–Sept) | Confidence drops to 0.0 → `HOLD_OPEN_LOOP` trigger. ERA5-Land gap-filling when gaps > 10 days. | `TemporalInterpolationService._apply_gap_masking()` | Option C |
| **Fragmented Land Holdings** (2.5 acres avg) | W-Shape protocol (5 photos) with GPS EXIF. Spatial Alignment snaps points to 10m grid. | `scout_sessions.py`, `spatial_alignment_service.py` | Option D |
| **No NIR Sensor on Phones** | Uses GRVI = (G - R)/(G + R) instead of NDVI. | `scout_sessions.py._calculate_grvi()` | Option D |
| **Low-Cost, Offline Constraints** | 30MB upload limit. Compressed to 1280×1280 (80% storage reduction). Background processing. | `scout_sessions.py._compress_image()` | Operational resilience |
| **Crop Diversity** (>100 crops) | Crop-specific extinction coefficients: wheat (k=0.45), rice (k=0.55), maize (k=0.50). | `satellite_fetcher.py.CROP_K_COEFFICIENTS` | E-WOFOST calibration |

#### How AgriTwin Reduces WOFOST Dependency

| **WOFOST's Weakness** | **Our Correction Strategy** | **How It Reduces WOFOST Dependency** |
|----------------------|---------------------------|-------------------------------------|
| **N-Mineralization Flaw** (Cambridge 2015) | Farmer's phone GRVI + Sentinel-2 NDRE | **Bypass WOFOST's N-logic entirely.** Use plant's actual greenness instead of soil nitrogen guesses. |
| **Single-Layer Soil Water** (E-WOFOST 2025) | ERA5-Land 4-layer SM (0-7cm to 100cm) | **Replace "single bucket" with 4 real layers.** WOFOST follows satellite/reanalysis data. |
| **Missing Extreme Heat Spikes** | ERA5-Land hourly temperature | **Expose WOFOST to actual 2-hour heat spikes.** Feed hour-by-hour reality instead of daily averages. |
| **Systematic Yield Bias** (Cambridge 2015) | XGBoost learns 75-year residuals | **Let AI handle correction.** Predict the error and subtract it from WOFOST's output. |

---

### 🧬 Research-to-Code Implementation Map

For researchers and developers: here is exactly where each scientific component lives in the codebase.
### Core Scientific Decisions

| **Research Pillar** | **Source** | **Decision** | **Implementation** | **Key Result** |
|---------------------|------------|--------------|-------------------|----------------|
| **Option A: Multi-Model Ensemble** | *J. Hydrology* (2025), AERU (2010) | ❌ **Rejected** | DSSAT/AquaCrop require MCMC calibration & 40+ genetic coefficients. 0% adoption in smallholder contexts. | N/A |
| **Option B: E-WOFOST (4-layer SM)** | *Computers & Electronics in Ag.* (2025) | ✅ **Adopted** | ERA5-Land provides 4-layer soil moisture (0-7, 7-28, 28-100, 100-289 cm). Joint LAI+SM assimilation. | **R² = 0.85–0.90**, RMSE = 441–741 kg/ha |
| **Option C: NDRE (Satellite)** | Sentinel-2 Red Edge bands | ✅ **Adopted** | `NDRE = (NIR - RedEdge) / (NIR + RedEdge)` using B08 & B07. Confidence scoring (0.85 clear / 0.0 cloudy). | Detects N-stress without saturating at high LAI |
| **Option D: GRVI (Smartphone)** | *Plants* (2024) | ✅ **Adopted** | `GRVI = (Green - Red) / (Green + Red)`. W-Shape protocol (5 photos). **30% observation error** ("Gentle Nudge"). | R² > 0.85 with SPAD. Replaces ₹40,000 chlorophyll meters. |
| **Option E: ERA5-Land** | *ESSD* (2021) | ✅ **Adopted (Hybrid)** | ERA5-Land (>60 days) + NASA POWER (<60 days & forecasts). | 9 km, hourly, 4-layer SM, 1950+ coverage |
| **EnKF (Data Assimilation)** | Ensemble Kalman Filter | ✅ **Adopted** | 25-member ensemble. Matrix Kalman Gain (covariance of LAI, SM, WLV, WST, WRT, WSO). | Prevents model drift, corrects input biases |
| **XGBoost Bias Correction** | *MDPI Agriculture* (2025) | ✅ **Adopted** | 75-year ERA5-Land back-cast (1950–2025). Predicts WOFOST residual errors. | County-level: r = 0.659, RMSE = 578 kg/ha |

#### Satellite Vegetation Indices Comparison

| **Index** | **Formula** | **Best Use Case** |
|-----------|-------------|-------------------|
| **NDVI** | `(NIR - Red) / (NIR + Red)` | General crop health; **saturates at high LAI** |
| **EVI** | `2.5 × (NIR - Red) / (NIR + 6Red - 7.5Blue + 1)` | High LAI regions; corrects for atmosphere & soil |
| **GNDVI** | `(NIR - Green) / (NIR + Green)` | **Chlorophyll/Nitrogen sensitivity** (better than NDVI) |
| **NDRE** | `(NIR - RedEdge) / (NIR + RedEdge)` | **Most sensitive to canopy chlorophyll**; does not saturate |

**AgriTwin Decision:** NDRE for satellites (no saturation, N-detection), GRVI for smartphones (no NIR sensor needed).

## 📐 System Architecture

### Modular Architecture Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         MODULAR ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   3.1 Farm Management                        │   │
│  │  Farm → Field → Season → Crop hierarchy with event updates   │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                3.2 Observation Layer                         │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐  │   │
│  │  │ Sentinel-2   │ │ Smartphone   │ │ Weather & Soil    │  │   │
│  │  │ NDRE/LAI     │ │ GRVI (W-Shape)│ │ (ERA5-Land +     │  │   │
│  │  │ (5-day)      │ │ (5 photos)   │ │  SoilGrids)       │  │   │
│  │  └──────────────┘ └──────────────┘ └────────────────────┘  │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  3.3 Data Fusion Module                      │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌─────────┐ │   │
│  │  │ Observation│ │ Temporal   │ │ Spatial    │ │Confidence│ │   │
│  │  │ Validation │ │ Alignment  │ │ Alignment  │ │Estimation│ │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └─────────┘ │   │
│  │  ┌────────────────────────────────────────────────────────┐ │   │
│  │  │           Multi-source Fusion                          │ │   │
│  │  │   (S2 + S1 SAR + GRVI, cloud-adaptive weighting)      │ │   │
│  │  └────────────────────────────────────────────────────────┘ │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               3.5 Physics Simulation                         │   │
│  │  ┌────────────────────────────────────────────────────────┐ │   │
│  │  │   WOFOST Engine (PCSE) with E-WOFOST enhancements      │ │   │
│  │  │   - 4-layer soil moisture (ERA5-Land)                  │ │   │
│  │  │   - Hourly heat stress tracking                         │ │   │
│  │  └────────────────────────────────────────────────────────┘ │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              3.6 Data Assimilation (EnKF)                    │   │
│  │  ┌────────────────────────────────────────────────────────┐ │   │
│  │  │  Ensemble Manager (25 members) → Kalman Gain →        │ │   │
│  │  │  State Update (LAI, SM, WLV, WST, WRT, WSO)          │ │   │
│  │  │  30% "Gentle Nudge" for farmer photos                 │ │   │
│  │  └────────────────────────────────────────────────────────┘ │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              3.7 Bias Correction (XGBoost)                    │   │
│  │  ┌────────────────────────────────────────────────────────┐ │   │
│  │  │  75-year ERA5-Land back-cast → Residual Learning →    │ │   │
│  │  │  Yield correction (TWSO)                              │ │   │
│  │  └────────────────────────────────────────────────────────┘ │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   3.4 Digital Twin Core                     │   │
│  │  ┌────────────────────────────────────────────────────────┐ │   │
│  │  │  Twin State Manager → Event Bus → Version Control      │ │   │
│  │  │  WebSockets → UI Real-time Updates (Coming Soon)      │ │   │
│  │  └────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 📂 Directory & Module Guide

| **Directory** | **Module(s)** | **Purpose** |
|---------------|---------------|-------------|
| `api/routes/` | 3.2, 3.9 | Scout session endpoints (GRVI, W-Shape protocol). |
| `assimilation/` | 3.6 | EnKF ensemble management, filter math, state vector, update logic. |
| `data_sources/` | 3.12, 3.11 | ERA5-Land, NASA POWER, SoilGrids, sensor adapters. |
| `models/` | 3.1 | Farm, Field, SimulationRun, DailyOutput (SQLAlchemy). |
| `repositories/` | 3.1 | SQLAlchemy database access layer. |
| `satellite/` | 3.10 | Sentinel-1/2 providers, vegetation indices, LAI estimators. |
| `scenario/` | 3.13 | Sowing date, variety, irrigation sweep generators. |
| `services/` | 3.3, 3.7 | Data fusion pipeline, temporal/spatial alignment, bias correction, window generator. |
| `simulation/` | 3.5 | PCSE WOFOST execution engine, agromanagement, output parsing. |
| `tests/` | All | 290 unit and integration tests. |
| `alembic/` | 3.18 | Database schema migrations. |
| `docs/` | N/A | Technical documentation (EnKF design, weather pipeline, etc.). |

---

## ⚙️ Developer Quickstart

### Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/AgriTwin.git
cd AgriTwin

# 2. Create virtual environment (using uv for 10-100x faster installs)
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
uv sync
```

**Configuration (`.env`)**

```env
# Application
APP_NAME=AgriTwin
APP_ENV=development
DEBUG=True
API_V1_PREFIX=/api/v1

# Database (SQLite for dev, PostgreSQL for production)
DATABASE_URL=sqlite:///./agritwin.db

# SentinelHub (for Sentinel-2 data)
SENTINEL_HUB_CLIENT_ID=your_client_id
SENTINEL_HUB_CLIENT_SECRET=your_client_secret

# ECMWF CDS API (for ERA5-Land)
CDS_API_KEY=your_api_key
CDS_API_URL=https://cds.climate.copernicus.eu/api
```

### Running the API Server

```bash
# 1. Apply database migrations
alembic upgrade head

# 2. Start the FastAPI server
uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: http://localhost:8000/docs

### Testing & Demo

```bash
# Run all 290 tests
pytest

# Run the automated EnKF assimilation demo
python3 run_demo.py
```

**Expected Demo Output:**

- Open-loop yield: **7271.7 kg/ha**
- EnKF-corrected yield: **5774.36 kg/ha**
- Yield convergence shown across 25 ensemble members.

---


### ⚠️ Critical Constraints

| **Constraint** | **Detail** |
|---------------|-----------|
| **Image Upload Limit** | 30 MB total (5 images). Compressed to 1280×1280 at 75% quality. |
| **ERA5-Land Delay** | 2–3 month publication delay. NASA POWER kept as real-time fallback. |
| **IoT Sensors** | ❌ Explicitly Rejected (costly for smallholders, maintenance). |
| **Drones** | ❌ Explicitly Rejected (costly, licensing, maintenance). |
| **DSSAT/AquaCrop** | ❌ Rejected (MCMC calibration, 40+ genetic coefficients, 0% adoption). |
| **GRVI→LAI Formula** | `LAI = 0.5 + (GRVI × 3.0)` is a linear approximation. Needs field validation. |
| **XGBoost Model Drift** | Trained on 75 years of ERA5 data. Schedule **yearly retraining**. |
| **SentinelHub Quota** | Free tier has API limits. Cache usage is critical. |

---

## 📡 API Reference

### 1. Submit Smartphone Scout Session (5 Photos)

```bash
curl -X POST http://localhost:8000/fields/YOUR_FIELD_UUID/scout-session \
     -F "images=@photo1.jpg" \
     -F "images=@photo2.jpg" \
     -F "images=@photo3.jpg" \
     -F "images=@photo4.jpg" \
     -F "images=@photo5.jpg" \
     -F "session_notes=W-shape walk, clear sky"
```

**Response:**
```json
{
  "session_id": "uuid",
  "field_id": "uuid",
  "timestamp": "2020-07-15T11:30:00Z",
  "processing_status": "completed",
  "results": {
    "median_grvi": 0.42,
    "estimated_lai": 1.76,
    "observation_error": 0.53,
    "confidence": 0.70,
    "quality_score": 0.85
  }
}
```

**See:** `docs/w_shape_grvi_protocol.md` for complete W-Shape protocol specification.

---

```bash
curl -X POST http://localhost:8000/simulate \
     -H 'Content-Type: application/json' \
     -d '{
       "latitude": 26.8,
       "longitude": 80.9,
       "crop": "rice",
       "variety": "Rice_IR64",
       "sowing_date": "2020-06-20",
       "harvest_date": "2020-11-10",
       "use_real_weather": true,
       "use_real_soil": true
     }'
```

### 3. Trigger EnKF Assimilation

```bash
curl -X POST http://localhost:8000/assimilation/run \
     -H 'Content-Type: application/json' \
     -d '{
       "simulation_id": "YOUR_SIMULATION_UUID",
       "field_id": "YOUR_FIELD_UUID",
       "ensemble_size": 25
     }'
```

### 4. Fetch Assimilation History & Yield Evolution

```bash
# History audit trail (priors, posteriors, innovations)
curl -X GET http://localhost:8000/assimilation/YOUR_SIMULATION_UUID/history

# Yield convergence across ensemble cycles
curl -X GET http://localhost:8000/assimilation/YOUR_SIMULATION_UUID/yield-evolution

# Daily timeseries: Open-Loop vs EnKF vs Observations
curl -X GET http://localhost:8000/assimilation/YOUR_SIMULATION_UUID/timeseries
```

---

**Built with ❤️ for farmers, agronomists, and researchers worldwide.** 🌾

---

*For detailed implementation notes, see the docs/ directory.*
