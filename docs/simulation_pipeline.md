# 🌾 WOFOST 7.2 Crop Simulation & AgroManagement Pipeline

## 1. WOFOST Biophysical Crop Model

**WOFOST** (World Food Studies) 7.2 is a mechanistic crop growth simulation model developed by Wageningen University and implemented in Python via **PCSE** (`Python Crop Simulation Environment`).

### Core Physical Processes Simulated Daily:
1. **Light Interception & $CO_2$ Assimilation**: Driven by solar radiation, temperature, and current Leaf Area Index (`LAI`).
2. **Maintenance & Growth Respiration**: Temperature-dependent carbon costs for living tissue maintenance.
3. **Dry Matter Partitioning**: Dynamic allocation fractions to leaves (`TWLV`), stems (`TWST`), roots (`TWRT`), and storage organs (`TWSO`) governed by Development Stage (`DVS`).
4. **Soil Water Balance & Transpiration**: Multi-layer soil water balance (`WaterbalanceFD`) computing potential transpiration (`TRAMX`), actual transpiration (`TRA`), and water stress index (`RFTRA = TRA / TRAMX`).

---

## 2. Supported Crops & Varieties

AgriTwin provides parameterized crop definitions:

| Crop Key | Supported Varieties | Default Base Temp ($T_b$) | Typical Season Length |
|---|---|---|---|
| `rice` | `Rice_IR64`, `Rice_501`, `Rice_HYV` | 8.0 °C | 120–150 days |
| `wheat` | `Wheat_Triticum_aestivum`, `Winter_Wheat_101` | 0.0 °C | 110–140 days |
| `maize` | `Maize_VanHeemst_1988` | 10.0 °C | 100–130 days |
| `soybean` | `Soybean_901` | 7.0 °C | 90–120 days |

---

## 3. AgroManagement & Dynamic Irrigation

The `AgroManagement` subsystem controls crop calendar events, fertilizer applications, and irrigation scheduling.

### 3.1 Timed Irrigation
Fixed-date or phenology-based water applications:
```json
[
  {"date": "2020-07-05", "amount_mm": 50.0},
  {"date": "2020-07-20", "amount_mm": 50.0},
  {"date": "2020-08-05", "amount_mm": 50.0}
]
```

### 3.2 Automated Irrigation Tiers
AgriTwin includes standard tiered irrigation libraries for scenario exploration:
- **Rainfed Control**: 0 mm applied (quantifies baseline water-deficit stress).
- **2-Event Critical Window**: 100 mm applied at tillering (DAS 30) and flowering (DAS 90).
- **4-Event Standard**: 200 mm applied at early vegetative, active tillering, heading, and grain filling.
- **Full Non-Limiting**: Water applied whenever soil moisture drops below critical threshold ($\text{SM} < \text{SMCR}$).

---

## 4. Environmental Providers

### 4.1 Weather Pipeline
AgriTwin employs a hybrid weather provider:
- **Operational Runs**: NASA POWER REST API fetches daily global solar radiation, temperature extremes, humidity, and rainfall.
- **Reanalysis & Batch Training**: ERA5-Land provides historical hourly and daily meteorological grids with local file caching in `.agritwin_cache/weather/`.

### 4.2 Soil Hydrology Pipeline
- Soil hydraulic parameters are queried dynamically from **ISRIC SoilGrids 250m** at field GPS coordinates.
- Parameters extracted:
  - $\text{SM0}$: Volumetric soil moisture at saturation [$cm^3/cm^3$]
  - $\text{SMFCF}$: Soil moisture at field capacity [$cm^3/cm^3$]
  - $\text{SMW}$: Soil moisture at permanent wilting point [$cm^3/cm^3$]
  - $\text{CRAIRC}$: Critical air content for aeration [$cm^3/cm^3$]
  - $\text{K0}$: Saturated hydraulic conductivity [$cm/\text{day}$]
