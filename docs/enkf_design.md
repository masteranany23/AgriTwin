# 🔄 Ensemble Kalman Filter (EnKF) Design & Assimilation Theory

## 1. Introduction

The Ensemble Kalman Filter (EnKF) is a Monte Carlo implementation of the Kalman Filter designed for high-dimensional, non-linear dynamical systems like biophysical crop models. 

In AgriTwin, EnKF fuses remote sensing observations (Sentinel-2 LAI, NDRE) and ground scouting data with the WOFOST 7.2 simulation state to correct error trajectories in-season.

---

## 2. Mathematical Formulation

### 2.1 State Vector Definition
The crop state vector $\mathbf{x} \in \mathbb{R}^n$ ($n=4$) is defined as:
$$\mathbf{x} = \begin{bmatrix} \text{LAI} \\ \text{SM} \\ \text{TAGP} \\ \text{TWSO} \end{bmatrix} = \begin{bmatrix} \text{Leaf Area Index } [m^2/m^2] \\ \text{Volumetric Soil Moisture } [cm^3/cm^3] \\ \text{Total Above-Ground Production } [kg/ha] \\ \text{Total Weight of Storage Organs } [kg/ha] \end{bmatrix}$$

### 2.2 Ensemble Representation
An ensemble of $N$ stochastic state vectors ($N \in [25, 50]$) represents the probability distribution of the system:
$$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N] \in \mathbb{R}^{n \times N}$$

The ensemble mean $\mathbf{\bar{x}}$ and perturbation matrix $\mathbf{A}'$ are given by:
$$\mathbf{\bar{x}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i$$
$$\mathbf{A}' = \mathbf{X} - \mathbf{\bar{x}} \mathbf{1}_N^T$$

The background sample error covariance $\mathbf{P}^f$ is:
$$\mathbf{P}^f = \frac{1}{N-1} \mathbf{A}' (\mathbf{A}')^T$$

---

## 3. The EnKF Assimilation Cycle

At each observation timestamp $t_k$:

### Step 1: Forecast Step
Each ensemble member is integrated forward in time using the non-linear WOFOST forward operator $\mathcal{M}$:
$$\mathbf{x}_{i, k}^f = \mathcal{M}(\mathbf{x}_{i, k-1}^a, \mathbf{w}_{i, k})$$
where $\mathbf{w}_{i, k} \sim \mathcal{N}(0, \mathbf{Q})$ represents model/weather process noise.

### Step 2: Perturbed Observations
Observations $\mathbf{y}_k \in \mathbb{R}^m$ are perturbed with Gaussian observation noise $\boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \mathbf{R})$:
$$\mathbf{y}_{i, k} = \mathbf{y}_k + \boldsymbol{\epsilon}_i, \quad i=1, \dots, N$$

Observation covariance $\mathbf{R} \in \mathbb{R}^{m \times m}$ is dynamic:
- High quality Sentinel-2 scene ($<5\%$ cloud cover): $\sigma_{\text{LAI}} = 0.15 \implies R = 0.0225$
- Cloudy scene / Interpolated estimate: $\sigma_{\text{LAI}} = 0.35 \implies R = 0.1225$
- Smartphone GRVI estimate: $\sigma_{\text{LAI}} = 0.30 \implies R = 0.0900$

### Step 3: Kalman Gain Computation
The observation operator $\mathbf{H}: \mathbb{R}^n \to \mathbb{R}^m$ maps state space to observation space (e.g., selecting the LAI row):
$$\mathbf{K}_k = \mathbf{P}_k^f \mathbf{H}^T \left( \mathbf{H} \mathbf{P}_k^f \mathbf{H}^T + \mathbf{R}_k \right)^{-1}$$

### Step 4: Analysis (State Update)
Each ensemble member is updated independently:
$$\mathbf{x}_{i, k}^a = \mathbf{x}_{i, k}^f + \mathbf{K}_k \left( \mathbf{y}_{i, k} - \mathbf{H} \mathbf{x}_{i, k}^f \right)$$

Physical bound constraints are applied immediately post-update:
$$\text{LAI} \ge 0, \quad \text{SMW} \le \text{SM} \le \text{SM0}, \quad \text{TAGP} \ge \text{TWSO} \ge 0$$

---

## 4. Module Architecture

The EnKF module is located in `backend/app/assimilation/`:

```
assimilation/
├── filters/
│   └── enkf.py             # Pure mathematical EnKF implementation
├── ensemble/
│   ├── ensemble_manager.py # Manages N WOFOST instances & perturbations
│   └── ensemble_member.py  # Single state representation
├── state/
│   └── state_vector.py     # State vector definitions & conversions
├── updater/
│   └── state_updater.py    # Injects posterior state back into PCSE engine
└── services/
    └── assimilation_service.py # Orchestrates the full sequential loop
```

---

## 5. Innovation & Quality Diagnostics

For every cycle $k$, AgriTwin computes diagnostic metrics:
1. **Innovation Vector**: $\mathbf{d}_k = \mathbf{y}_k - \mathbf{H}\mathbf{\bar{x}}_k^f$
2. **Normalized Innovation Squared (NIS)**: $\epsilon_k = \mathbf{d}_k^T (\mathbf{H}\mathbf{P}_k^f\mathbf{H}^T + \mathbf{R}_k)^{-1} \mathbf{d}_k$
3. **Quality Score**: $Q_k = \exp(-0.5 \cdot |\epsilon_k - m|)$ where $m = \dim(\mathbf{y})$.
