# era5_land_integration.md
# AgriTwin — ERA5-Land Integration (Weather & Soil Moisture)

---

## 1. What is ERA5-Land?

**ERA5-Land** is AgriTwin's primary weather and soil moisture data source from ECMWF (European Centre for Medium-Range Weather Forecasts).

### Key Features

- **75+ years** of historical data (1950–present)
- **9 km resolution** (vs 31 km in ERA5)
- **4-layer soil moisture** (0-7, 7-28, 28-100, 100-289 cm)
- **Hourly temporal resolution**
- **Back-casting capability** for XGBoost training
- **2-3 month publication delay** (near-real-time)

---

## 2. Scientific Foundation

### What is ERA5-Land?

A reanalysis dataset that combines:
1. **Global atmospheric model** — ECMWF IFS
2. **Data assimilation** — Billions of observations (satellites, stations, buoys)
3. **Land surface model** — HTESSEL (soil moisture, temperature)

### Research Basis

| **Source** | **Key Finding** |
|-----------|----------------|
| *ESSD* (2021) | ERA5-Land provides 4-layer SM at 9 km, hourly |
| *Computers & Electronics in Ag.* (2025) | E-WOFOST + 4-layer SM → R² = 0.85-0.90 |
| *MDPI Agriculture* (2025) | 75-year back-cast enables XGBoost bias correction |
| AgriTwin Research | 2-3 month delay → Hybrid router (ERA5-Land + NASA POWER) |

### How ERA5-Land Reduces WOFOST Dependency

Here is exactly how we reduce dependency on WOFOST at every critical failure point:

| **WOFOST's Weakness (From Research)** | **Our Correction Strategy** | **How It Reduces WOFOST Dependency** |
|---------------------------------------|---------------------------|-------------------------------------|
| **1. N-Mineralization Flaw** (Cambridge 2015) | Farmer's phone calculates **GRVI** (Chlorophyll proxy) + Sentinel-2 **NDRE** | **We bypass WOFOST's N-logic entirely.** Instead of trusting WOFOST's guess about soil nitrogen, we use the plant's **actual greenness** to decide if N-stress is happening. |
| **2. Single-Layer Soil Water** (E-WOFOST 2025) | **ERA5-Land** provides 4-layer soil moisture (0-7cm to 100cm) | **We replace WOFOST's "single bucket" with 4 real layers.** WOFOST no longer guesses water distribution; it follows the satellite/reanalysis data. |
| **3. Missing Extreme Heat Spikes** (Hourly vs. Daily) | **ERA5-Land** provides hourly temperature data | **We expose WOFOST to the *actual* 2-hour heat spike.** WOFOST can't simulate heat stress if it only sees a daily average. We feed it the raw hour-by-hour reality. |
| **4. Systematic Yield Bias** (Cambridge 2015) | **XGBoost** learns WOFOST's historical residuals (75 years of back-casts) | **We let AI handle the correction.** Instead of rebuilding WOFOST's physics to fix its bias, we train a model to *predict the error* and subtract it from WOFOST's output. |

---

## 3. Available Variables

### Weather Variables (Daily Aggregation)

| **Variable** | **Units** | **Temporal** | **WOFOST Mapping** |
|-------------|-----------|-------------|-------------------|
| 2m Temperature | K | Hourly → Daily min/max/mean | TMIN, TMAX, TEMP |
| Total Precipitation | m | Hourly → Daily sum | RAIN |
| Surface Solar Radiation | J/m² | Hourly → Daily sum → MJ/m² | IRRAD |
| 10m Wind Speed | m/s | Hourly → Daily mean | WIND |
| 2m Dewpoint Temperature | K | Hourly → Daily mean | VAP (vapor pressure) |

### Soil Moisture Variables (4-Layer)

| **Layer** | **Depth (cm)** | **Variable** | **WOFOST Usage** |
|----------|---------------|-------------|------------------|
| **Layer 1** | 0 - 7 | `swvl1` | Surface SM for EnKF |
| **Layer 2** | 7 - 28 | `swvl2` | Shallow root zone |
| **Layer 3** | 28 - 100 | `swvl3` | Deep root zone |
| **Layer 4** | 100 - 289 | `swvl4` | Subsoil drainage |

**Note:** WOFOST uses **total root zone SM** aggregated from layers 1-3.

---

## 4. Why 4-Layer Soil Moisture Matters

### E-WOFOST Enhancement

**Traditional WOFOST:**
- 1-layer bucket model
- Total SM = SMFCF (field capacity) to SMW (wilting point)
- No vertical distribution

**E-WOFOST with ERA5-Land:**
- 4-layer profile → Better root water uptake
- Joint LAI + SM assimilation
- **Validation:** R² = 0.85-0.90 vs single-layer R² = 0.70-0.75

### XGBoost Feature Engineering

4-layer SM becomes **dynamic features** for bias correction:

```python
# Sliding window features (30-day)
features = [
    'lai_mean', 'lai_std', 'lai_trend',
    'sm_layer1_mean', 'sm_layer1_std',  # Surface (0-7 cm)
    'sm_layer2_mean', 'sm_layer2_std',  # Shallow (7-28 cm)
    'sm_layer3_mean', 'sm_layer3_std',  # Deep (28-100 cm)
    'sm_layer4_mean', 'sm_layer4_std',  # Subsoil (100-289 cm)
    'heat_strain_hours',  # Hours >34°C
    'dvs_phase'  # VEGETATIVE, REPRODUCTIVE, GRAIN_FILL
]
```

**Result:** XGBoost learns WOFOST overestimates yield when:
- Surface SM (layer 1) is low → Water stress
- Heat strain hours > 50 → Heat stress
- DVS = GRAIN_FILL → Critical period

---

## 5. Data Access

### ECMWF Climate Data Store (CDS)

**Registration:**
1. Create account: https://cds.climate.copernicus.eu/
2. Copy API key from account page
3. Add to `.env`:
   ```env
   CDS_API_KEY=your_api_key
   CDS_API_URL=https://cds.climate.copernicus.eu/api
   ```

**API Limits:**
- Free tier: 100,000 requests/month
- Max download: 20 GB/request
- Queue wait: 1-10 minutes

---

## 6. Hourly to Daily Aggregation

```python
def aggregate_hourly_to_daily(ds: xr.Dataset) -> pd.DataFrame:
    """Convert hourly ERA5-Land to daily WOFOST format."""
    
    for date in pd.date_range(ds.time[0], ds.time[-1], freq='D'):
        day_ds = ds.sel(time=slice(date, date + timedelta(hours=23)))
        
        # Temperature (K → °C)
        temp_hourly = day_ds['t2m'].values - 273.15
        tmin = float(np.min(temp_hourly))
        tmax = float(np.max(temp_hourly))
        tmean = float(np.mean(temp_hourly))
        
        # Precipitation (m → mm)
        rain = float(np.sum(day_ds['tp'].values) * 1000)
        
        # Solar radiation (J/m² → MJ/m²)
        irrad = float(np.sum(day_ds['ssrd'].values) / 1e6)
        
        # Wind speed (u, v → magnitude)
        u_wind = day_ds['u10'].values
        v_wind = day_ds['v10'].values
        wind = float(np.mean(np.sqrt(u_wind**2 + v_wind**2)))
        
        # Soil moisture (4 layers, m³/m³)
        sm_layer1 = float(np.mean(day_ds['swvl1'].values))
        sm_layer2 = float(np.mean(day_ds['swvl2'].values))
        sm_layer3 = float(np.mean(day_ds['swvl3'].values))
        sm_layer4 = float(np.mean(day_ds['swvl4'].values))
        
        # Total root zone SM (layers 1-3, weighted by depth)
        sm_root_zone = (sm_layer1 * 7 + sm_layer2 * 21 + sm_layer3 * 72) / 100
```

---

## 7. Hybrid Weather Router

### Problem: Real-Time Gap

ERA5-Land has **2-3 month publication delay**:

```
Today: 2025-01-15
Latest ERA5-Land: 2024-10-31
Gap: 76 days (missing)
```

### Solution: Hybrid Router

```python
class HybridWeatherService:
    """Route weather requests to ERA5-Land or NASA POWER."""
    
    def get_weather(self, latitude, longitude, start_date, end_date):
        """Route based on date range."""
        
        days_ago = (date.today() - end_date).days
        
        if days_ago >= 60:
            # Historical → ERA5-Land (high quality, 4-layer SM)
            return self.era5_source.fetch_weather_and_soil(...)
        else:
            # Recent → NASA POWER (real-time, no SM)
            nasa_data = self.nasa_source.fetch_weather(...)
            # Add placeholder SM from SoilGrids static estimate
            return nasa_data
```

**Routing Logic:**

```
┌────────────────────────────────────────────────┐
│         Weather Data Router                    │
├────────────────────────────────────────────────┤
│                                                │
│  Date: 2020-06-01 to 2020-11-30               │
│  Days Ago: 1,500 (>60 threshold)              │
│  ✅ Route to: ERA5-Land                        │
│  - Fetch 4-layer SM                            │
│  - Cache for future use                        │
│                                                │
├────────────────────────────────────────────────┤
│                                                │
│  Date: 2024-11-01 to 2025-01-15               │
│  Days Ago: 30 (<60 threshold)                 │
│  ✅ Route to: NASA POWER                       │
│  - No SM data                                  │
│  - Use SoilGrids static estimate               │
│                                                │
└────────────────────────────────────────────────┘
```

---

## 8. XGBoost Bias Correction Training

### 75-Year Back-Cast

```python
class BiasCorrectrioonTrainer:
    """Train XGBoost on 75-year ERA5-Land back-casts."""
    
    def generate_training_data(self, latitude, longitude, crop, variety):
        """Run WOFOST for 75 years (1950-2025)."""
        
        for year in range(1950, 2026):
            # Fetch ERA5-Land for this year
            weather_data = self.era5_source.fetch_weather_and_soil(
                latitude, longitude,
                date(year, 1, 1), date(year, 12, 31)
            )
            
            # Run WOFOST simulation
            simulation = self.wofost_engine.run(
                crop, variety, weather_data,
                sowing_date=date(year, 6, 20),
                harvest_date=date(year, 11, 10)
            )
            
            # Extract sliding windows (30-day)
            windows = self.window_generator.generate_windows(
                simulation.get_daily_outputs(), window_size=30
            )
```

### Sliding Window Features

| **Feature** | **Description** |
|------------|----------------|
| `lai_mean`, `lai_std`, `lai_trend` | LAI statistics over 30 days |
| `sm_layer1_mean`, `sm_layer1_std` | Surface SM (0-7 cm) |
| `sm_layer2_mean`, `sm_layer2_std` | Shallow root zone (7-28 cm) |
| `sm_layer3_mean`, `sm_layer3_std` | Deep root zone (28-100 cm) |
| `sm_layer4_mean`, `sm_layer4_std` | Subsoil (100-289 cm) |
| `heat_strain_hours` | Hours >34°C (heat stress) |
| `dvs_phase` | 0=VEGETATIVE, 1=REPRODUCTIVE, 2=GRAIN_FILL |
| **`yield_error`** | **Target: predicted - actual yield** |

### XGBoost Training

```python
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200
}

model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

# Validation: County-level
# r = 0.659, RMSE = 578 kg/ha
```

---

## 9. EnKF Joint LAI + SM Assimilation

```python
def assimilate_lai_and_sm(
    ensemble,
    observation_lai,
    observation_sm,  # ERA5-Land layer 1 (0-7 cm)
    observation_error_lai,
    observation_error_sm
):
    """Update ensemble using joint LAI + SM observation."""
    
    # Observation vector (2D: LAI + SM)
    observation = np.array([observation_lai, observation_sm])
    
    # Observation error covariance (R matrix)
    R = np.diag([observation_error_lai**2, observation_error_sm**2])
    
    # Measurement operator (maps state to observations)
    H = np.array([
        [1, 0, 0, 0, 0, 0],  # LAI directly observed
        [0, 1, 0, 0, 0, 0]   # SM directly observed
    ])
    
    # Kalman Gain: K = P^f H^T (H P^f H^T + R)^-1
    # Update: x_analysis = x_forecast + K @ innovation
```

**Result:** Joint LAI + SM → **R² = 0.85-0.90** (vs LAI-only R² = 0.70-0.75)

---

## 10. Performance & Costs

### Data Volume

**Single location, 1 year:**
- Variables: 10 (weather + 4 SM layers)
- Temporal: Hourly (8,760 values/year)
- File size: ~5 MB (NetCDF compressed)

**75-year back-cast:**
- 75 years × 5 MB = **375 MB** per location

### API Usage

**CDS API (free tier):**
- 100,000 requests/month
- Max 20 GB/request

**AgriTwin usage:**
- 1,000 fields × 1 request/field = **1,000 requests/month** (well below limit)

---

## 11. Best Practices

### For Developers

✅ **Do:**
- Cache ERA5-Land downloads (save API calls)
- Use hybrid router (ERA5-Land + NASA POWER)
- Aggregate hourly → daily carefully
- Include 4-layer SM in XGBoost features
- Monitor CDS queue times

❌ **Don't:**
- Query CDS for recent data (<60 days) — use NASA POWER
- Skip caching (CDS is slow)
- Use ERA5 instead of ERA5-Land (31 km vs 9 km)
- Ignore SM layers (critical for E-WOFOST)

---

## 12. Code References

| **Component** | **File Path** |
|--------------|--------------|
| ERA5-Land Source | `data_sources/era5_land_source.py` |
| Hybrid Weather Router | `services/weather_service.py` |
| Hourly Aggregation | `services/era5_aggregator.py` |
| Window Generator | `services/window_generator.py` |
| Bias Correction Trainer | `services/bias_correction_trainer.py` |
| EnKF Joint Assimilation | `assimilation/filters/enkf.py` |
