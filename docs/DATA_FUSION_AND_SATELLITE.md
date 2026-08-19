# 🛰️ Multi-Source Data Fusion & Satellite Pipeline

## 1. Multi-Source Observation Architecture

AgriTwin integrates diverse remote sensing and ground observations into a unified state observation stream for EnKF assimilation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ OBSERVATION SOURCES                                                         │
│                                                                             │
│  [1. Sentinel-2 MSI]          [2. Farmer Smartphone]      [3. Ground Sensors]│
│  • 10m/20m Multi-spectral     • 5-Point W-Shape Pattern   • IoT Soil Probes  │
│  • NDRE, NDVI, SeLI           • Green-Red Index (GRVI)    • 4-Layer Moisture │
│  • Cloud Mask Filtering       • In-field Calibration      • Hourly Logging   │
└───────────────────────┬───────────────────┬──────────────────────┬──────────┘
                        │                   │                      │
                        ▼                   ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODULE 3.3 DATA FUSION PIPELINE                                             │
│                                                                             │
│  1. Validation & Range Check: Physical constraints on LAI and Soil Moisture │
│  2. Temporal Interpolation: Cloud gap recovery via spline & GP inpainting   │
│  3. Spatial Alignment: Field polygon clipping & bilinear resampling         │
│  4. Dynamic Confidence Scoring: Cloud fraction, sensor age, geometry penalty │
│  5. Bayesian Fusion: Inverse-variance weighting & observation covariance R   │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │   Assimilated Observation y   │
                        │  Uncertainty Covariance R     │
                        └───────────────────────────────┘
```

---

## 2. Sentinel-2 Vegetation Indices & LAI Inversion

Sentinel-2 Level-2A surface reflectance bands are processed into vegetation indices:

### 2.1 Normalized Difference Red Edge Index (NDRE)
Red Edge bands are sensitive to chlorophyll content and avoid NIR saturation at high canopy densities:
$$\text{NDRE} = \frac{B_8 - B_5}{B_8 + B_5}$$
where $B_8$ is NIR (842 nm) and $B_5$ is Red Edge 1 (705 nm).

### 2.2 Normalized Difference Vegetation Index (NDVI)
$$\text{NDVI} = \frac{B_8 - B_4}{B_8 + B_4}$$

### 2.3 LAI Inversion
Empirical exponential canopy radiative transfer inversion:
$$\text{LAI} = a \cdot \exp(b \cdot \text{VI}) + c$$
Default calibrated parameters for cereal crops:
$$\text{LAI} = 0.28 \cdot \exp(3.8 \cdot \text{NDRE}) - 0.20$$

---

## 3. W-Shape Smartphone Scouting Protocol

During prolonged monsoon cloud cover (>70% cloud contamination), satellite optical passes are obscured. Farmers capture 5 smartphone photos following a standard **W-Shape sampling transect** across the field.

```
Node 1 (Corner) ──────── Node 3 (Center) ──────── Node 5 (Far Corner)
        \                      /   \                      /
         \                    /     \                    /
          Node 2 (Quarter) ───       Node 4 (Three-Quarter)
```

### Green-Red Vegetation Index (GRVI)
$$\text{GRVI} = \frac{G - R}{G + R}$$
The median GRVI across the 5 photos is converted to an LAI estimate. Smartphone observations are assigned a conservative observation uncertainty $\sigma = 0.30$ ($R=0.090$) to act as a **"Gentle Nudge"** during cloudy periods.

---

## 4. Cloud Gap Recovery & Temporal Interpolation

When observation intervals exceed 10 days due to cloud persistence:
1. Interpolation engine evaluates time delta $\Delta t$.
2. Generates smoothed daily trajectory with expanded error variance $\sigma(t) = \sigma_0 + \alpha \Delta t$.
3. Quality flags (`HIGH_CONFIDENCE`, `INTERPOLATED`, `CLOUDY_DEGRADED`) are stamped on each record.

---

## 5. Bayesian Multi-Source Fusion

When multiple observations coincide on the same day $t$:
$$y_{\text{fused}} = \left( \sum_{k} \frac{1}{\sigma_k^2} \right)^{-1} \sum_{k} \frac{y_k}{\sigma_k^2}$$
$$R_{\text{fused}} = \left( \sum_{k} \frac{1}{\sigma_k^2} \right)^{-1}$$
This guarantees maximum-likelihood state estimates with mathematically rigorous uncertainty bounds.
