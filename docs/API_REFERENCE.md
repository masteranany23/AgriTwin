# 📡 AgriTwin REST API Reference

The AgriTwin API is built with **FastAPI** and provides high-performance, asynchronous endpoints for crop simulation, satellite data ingestion, EnKF data assimilation, and scenario analysis.

Base URL: `http://localhost:8000`  
Swagger Interactive UI: `http://localhost:8000/docs`  
ReDoc Documentation: `http://localhost:8000/redoc`

---

## 1. Simulation Endpoints (`/simulate`)

### `POST /simulate`
Executes an open-loop or baseline WOFOST 7.2 simulation for a target geographic coordinate and crop variety.

#### Request Body
```json
{
  "latitude": 26.8,
  "longitude": 80.9,
  "crop": "rice",
  "variety": "Rice_IR64",
  "sowing_date": "2020-06-20",
  "harvest_date": "2020-11-10",
  "max_duration": 220,
  "use_real_weather": true,
  "use_real_soil": true,
  "field_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "irrigation_events": [
    {"date": "2020-07-05", "amount_mm": 50.0},
    {"date": "2020-07-20", "amount_mm": 50.0},
    {"date": "2020-08-05", "amount_mm": 50.0},
    {"date": "2020-08-20", "amount_mm": 50.0}
  ]
}
```

#### Response (`200 OK`)
```json
{
  "simulation_id": "c1f74d91-88c6-4191-a768-c1017ef8820e",
  "status": "COMPLETED",
  "summary": {
    "doe": "2020-06-20",
    "doh": "2020-11-10",
    "crop_name": "rice",
    "variety": "Rice_IR64"
  },
  "metrics": {
    "final_twso_kg_ha": 4820.5,
    "peak_lai": 4.12,
    "total_biomass_kg_ha": 11250.0,
    "water_stress_days": 3
  },
  "daily_outputs_count": 144
}
```

---

## 2. Field Management (`/fields`)

### `POST /fields`
Registers a new agricultural plot with GPS polygon coordinates.

#### Request Body
```json
{
  "name": "North Paddy Plot A",
  "latitude": 26.8,
  "longitude": 80.9,
  "boundary_geojson": {
    "type": "Polygon",
    "coordinates": [[
      [80.89, 26.79],
      [80.91, 26.79],
      [80.91, 26.81],
      [80.89, 26.81],
      [80.89, 26.79]
    ]]
  },
  "farm_name": "Sunrise Cooperative"
}
```

### `GET /fields`
Lists all registered fields.

---

## 3. Satellite Data Ingestion (`/satellite`)

### `GET /satellite/lai`
Fetches Sentinel-2 scenes for a target field, computes NDVI/NDRE, inverts to LAI, and creates observation records.

#### Query Parameters
- `field_id` (UUID): Field identifier.
- `start_date` (Date, YYYY-MM-DD): Window start.
- `end_date` (Date, YYYY-MM-DD): Window end.
- `index_name` (string, default `"NDVI"`): Index for LAI inversion (`"NDVI"`, `"NDRE"`, `"OSAVI"`, `"SeLI"`).
- `max_cloud_cover` (float, default `0.2`): Max allowable cloud fraction.
- `uncertainty` (float, default `0.3`): Observation standard deviation.

---

## 4. Closed-Loop Assimilation (`/assimilation`)

### `POST /assimilation/run`
Executes an Ensemble Kalman Filter (EnKF) assimilation cycle across all available observations for a simulation.

#### Request Body
```json
{
  "simulation_id": "c1f74d91-88c6-4191-a768-c1017ef8820e",
  "field_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "ensemble_size": 25
}
```

#### Response (`200 OK`)
```json
{
  "assimilation_run_id": "e4b3c2a1-5717-4562-b3fc-2c963f66afa6",
  "status": "COMPLETED",
  "executed_cycles": 12,
  "observations_assimilated": 12,
  "execution_time_seconds": 1.45
}
```

### `GET /assimilation/{simulation_id}/history`
Returns the cycle-by-cycle audit log containing prior states, observation vectors, innovations, posterior states, and quality scores.

### `GET /assimilation/{simulation_id}/yield-evolution`
Returns the sequential evolution of projected final storage organ yield (`TWSO`) as observations are assimilated across the season.

### `GET /assimilation/{simulation_id}/timeseries`
Returns daily time series comparing Open-Loop Simulation vs. EnKF Assimilated State vs. Ingested Observations.

---

## 5. Scenario Sweep Analysis (`/scenarios`)

### `POST /scenarios/sweep`
Executes multi-scenario simulations across varied sowing dates, irrigation tiers, or crop varieties to identify optimal agronomic management.

---

---

## 6. Farmer Advisory & Decision Support (`/advisory`)

### `POST /advisory/recommend-crop`
Evaluates candidate crops based on geographic location, season (Kharif/Rabi/Zaid), soil characteristics, and economic return (MSP vs cultivation cost) to recommend the most profitable crop choice.

#### Request Body
```json
{
  "latitude": 26.8,
  "longitude": 80.9,
  "season": "rabi",
  "land_area_acres": 2.5
}
```

#### Response (`200 OK`)
```json
{
  "latitude": 26.8,
  "longitude": 80.9,
  "season": "rabi",
  "land_area_acres": 2.5,
  "top_recommendation": {
    "crop_name": "Mustard",
    "variety_name": "Pusa Bold / Giriraj",
    "expected_yield_kg_ha": 1950.0,
    "expected_yield_quintal_acre": 7.89,
    "msp_inr_per_quintal": 5650.0,
    "net_profit_per_acre_inr": 33078.5,
    "total_net_profit_inr": 82696.25,
    "optimal_sowing_window": "Oct 15 - Oct 30",
    "key_advantage_en": "Low water requirement, highest net profit margin per acre due to high oilseed MSP.",
    "key_advantage_hi": "कम पानी की जरूरत और उच्च एमएसपी के कारण प्रति एकड़ सबसे अधिक शुद्ध मुनाफा।"
  },
  "ranked_options": [...],
  "summary_message_en": "For your 2.5 acre farm in the Rabi season, 'Mustard' is the top recommendation...",
  "summary_message_hi": "आपके 2.5 एकड़ खेत के लिए Rabi मौसम में, 'Mustard' सबसे लाभकारी फसल है..."
}
```

### `GET /advisory/field/{field_id}/daily`
Evaluates latest digital twin state and weather forecast to produce daily actionable alerts (irrigation, nitrogen, extreme weather) in English and Hindi.

### `GET /advisory/field/{field_id}/summary`
Compiles the complete executive summary card including current growth stage, expected yield (kg/ha and Q/acre), calibrated confidence interval, and ready-to-display WhatsApp cards.

---

## 7. System Health Check (`/health`)

### `GET /health`
Returns service status and database connectivity.
```json
{
  "status": "ok",
  "service": "agritwin",
  "version": "1.0.0",
  "database": "connected"
}
```
