# 🧠 ML Bias Correction & Uncertainty Quantification

## 1. The Need for Bias Correction in Crop Modeling

Mechanistic crop models such as **WOFOST 7.2** assume idealized agronomic conditions (uniform soil, pest-free environments, perfect seed viability). As a result, raw WOFOST yield forecasts exhibit systematic deviations from observed regional yields (such as India's DES Kaggle APY dataset).

AgriTwin solves this using a two-stage hybrid machine learning correction pipeline:
1. **Stage 1: Multi-Model Stacked Ensemble** (captures non-linear phenological and environmental feature interactions).
2. **Stage 2: Deep Gaussian Process (GP)** (models spatiotemporal residual correlations and produces calibrated confidence intervals).

---

## 2. Feature Engineering & Sliding Windows

From daily WOFOST and weather outputs, AgriTwin computes phenology-aligned features:

| Feature Name | Description | Agronomic Rationale |
|---|---|---|
| `lai_mean`, `lai_std` | 30-day sliding window mean & variance of LAI | Captures vegetative canopy development rate |
| `lai_trend` | Slope of linear regression on LAI over 30 days | Detects premature leaf senescence or accelerated growth |
| `sm_layer1_mean`..`sm_layer4_mean` | 4-layer soil moisture mean across root zone | Quantifies water stress across profile |
| `heat_strain_hours` | Total hours where $T_{\max} > 35^\circ C$ during flowering ($\text{DVS} \in [0.9, 1.2]$) | Direct heat-induced spikelet sterility |
| `dvs_phase` | Categorical development stage index | Phenological context |
| `historical_ics_ratio` | Department of Economics & Statistics (DES) ratio | Long-term regional baseline calibration |

---

## 3. Stage 1: Multi-Model Stacked Ensemble Architecture

```
                       ┌───────────────────────┐
                       │  Input Feature Vector │
                       └───────────┬───────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ XGBoost Regressor│      │LightGBM Regressor│      │  MLP Regressor   │
│ (n_est=100, d=6) │      │ (n_est=100, d=6) │      │  (100x50 layers) │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │   Ridge Meta-Learner  │
                       │    (Regularized α)    │
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │   Ensemble Estimate   │
                       └───────────────────────┘
```

### Time-Series Cross-Validation
To eliminate future data leakage in temporal yield sequences, models are evaluated using expanding-window `TimeSeriesSplit(n_splits=5)`:
- Split 1: Train [1997–2005], Test [2006–2008]
- Split 2: Train [1997–2008], Test [2009–2011]
- Split 3: Train [1997–2011], Test [2012–2014]
- Split 4: Train [1997–2014], Test [2015–2017]
- Split 5: Train [1997–2017], Test [2018–2020]

---

## 4. Stage 2: Deep Gaussian Process Spatiotemporal Correction

Residual errors $\epsilon = y_{\text{true}} - \hat{y}_{\text{ensemble}}$ exhibit spatial clustering (neighboring districts share unmodeled soil/microclimate factors) and temporal persistence (multi-year drought cycles).

### Kernel Formulation
We use a product kernel over geographic coordinates $(\text{lat}, \text{lon})$ and year $t$:
$$k((x, y, t), (x', y', t')) = \sigma_f^2 \exp\left(-\frac{\|\mathbf{s} - \mathbf{s}'\|^2}{2 \ell_s^2}\right) \exp\left(-\frac{|t - t'|^2}{2 \ell_t^2}\right) + \sigma_n^2 \delta$$
where:
- $\ell_s$: Spatial length scale [degrees / km]
- $\ell_t$: Temporal length scale [years]
- $\sigma_f^2$: Signal variance
- $\sigma_n^2$: Observation noise variance

### Posterior Prediction & Uncertainty
For a new query point $\mathbf{x}_* = (\text{lat}_*, \text{lon}_*, t_*)$:
$$y_* \sim \mathcal{N}(\mu_*, \sigma_*^2)$$
$$\mu_* = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$
$$\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

The 95% confidence interval is computed as:
$$\text{CI}_{95\%} = [\mu_* - 1.96\sigma_*, \mu_* + 1.96\sigma_*]$$

---

## 5. Execution via CLI & Microservice

The bias correction pipeline can be run standalone via `bias-corrector/run.py`:

```bash
# 1. Train the stacked ensemble
python bias-corrector/run.py train

# 2. Train with Deep GP enabled
python bias-corrector/run.py train --enable-gp

# 3. Start the dedicated microservice API
python bias-corrector/run.py api --port 8001
```
