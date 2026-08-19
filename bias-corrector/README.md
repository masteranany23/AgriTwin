# AgriTwin Bias Corrector

Standalone microservice for post-processing WOFOST yield predictions with machine learning-based bias correction.

## Features

- **Multiple Correction Modes**:
  - `constant`: ICS ratio multiplication (historical baseline)
  - `xgboost`: Single XGBoost model fallback
  - `ensemble`: Stacked ensemble (XGBoost + RandomForest + LightGBM)
  - `ensemble_gp`: Ensemble + Gaussian Process spatial-temporal correction

- **Stacked Ensemble**:
  - Base estimators: XGBoost, Random Forest, LightGBM
  - Meta-learner: Ridge regression
  - Time-series cross-validation to prevent leakage

- **Deep GP Correction**:
  - Spatial-temporal kernel: RBF(lat, lon) × RBF(year)
  - Provides uncertainty quantification (confidence intervals)
  - Implemented with `gpytorch`

- **ICS Historical Extraction**:
  - Extracts ICS ratios from DES PDFs (2010-2018)
  - Uses `tabula-py` for PDF table parsing
  - Fallback to manual text extraction

- **Monitoring & Logging**:
  - Async prediction logging to CSV
  - PSI-based drift detection
  - RMSE, R², MAPE metrics

- **Production-Ready**:
  - FastAPI with health checks
  - Docker deployment
  - Volume mounts for models and logs
  - Configurable via YAML

## Directory Structure

```
bias-corrector/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py                    # CLI entrypoint
├── README.md
├── config/
│   ├── development.yaml
│   └── production.yaml
├── src/
│   ├── api/
│   │   ├── app.py           # FastAPI application
│   │   ├── schemas.py       # Pydantic models
│   │   └── config.py        # Config loader
│   ├── model/
│   │   ├── ensemble.py      # Stacked ensemble
│   │   ├── gp_correction.py # Deep GP
│   │   └── correction.py    # Orchestrator
│   ├── features/
│   │   ├── histogram_builder.py  # Sentinel-2 histograms
│   │   └── pipeline.py           # Feature engineering
│   ├── data/
│   │   ├── loader.py        # Data loading
│   │   └── ics_extractor.py # ICS PDF extraction
│   ├── monitoring/
│   │   ├── logger.py        # Prediction logging
│   │   ├── drift.py         # PSI drift detection
│   │   └── metrics.py       # RMSE/R²/MAPE
│   └── utils/
│       └── helpers.py       # Utilities
├── data/
│   ├── training/            # train.csv
│   └── ics_ratios/          # ICS JSON files
├── models/                  # Saved model artifacts
└── logs/                    # Prediction logs
```

## Setup

### 1. Install Dependencies

```bash
cd bias-corrector
pip install -r requirements.txt
```

### 2. Prepare Training Data

Create `data/training/train.csv` with columns:

```
state, district, crop_key, year, wofost_yield, actual_yield,
latitude, longitude, lai_mean, ndvi_mean, ndre_mean,
rainfall_total, temperature_mean, soil_moisture_mean
```

Example row:
```csv
Punjab,Ludhiana,Wheat,2020,4500,4800,30.9,75.85,3.5,0.75,0.25,650,22.5,0.28
```

### 3. Extract ICS Ratios (Optional)

Place DES PDFs in `data/raw/des_pdfs/` and run:

```bash
python run.py extract-ics --start-year 2010 --end-year 2018
```

Or manually create `data/ics_ratios/ics_2019_2023.json`:

```json
{
  "Punjab_Ludhiana_Wheat_2020": 1.08,
  "Punjab_Ludhiana_Rice_Kh_2020": 0.95
}
```

### 4. Train Model

```bash
python run.py train --enable-gp
```

This will:
- Train stacked ensemble with cross-validation
- Evaluate on test set
- Train GP correction layer (if `--enable-gp`)
- Save models to `models/`

### 5. Start API

```bash
python run.py api --port 8000
```

Or with Docker:

```bash
docker-compose up --build
```

## Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "model_type": "ensemble_gp",
  "ics_ratios_loaded": true,
  "uptime_seconds": 123.45
}
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Punjab",
    "district": "Ludhiana",
    "crop_key": "Wheat",
    "year": 2023,
    "wofost_yield": 4500.0,
    "latitude": 30.9,
    "longitude": 75.85,
    "lai_mean": 3.5,
    "ndvi_mean": 0.75,
    "ndre_mean": 0.25,
    "rainfall_total": 650.0,
    "temperature_mean": 22.5,
    "soil_moisture_mean": 0.28
  }'
```

Response:
```json
{
  "original_yield": 4500.0,
  "corrected_yield": 4680.5,
  "correction_factor": 1.04,
  "ics_ratio": null,
  "model_version": "1.0.0",
  "confidence_interval": {
    "lower": 4450.2,
    "upper": 4910.8,
    "mean": 4680.5
  },
  "warnings": [],
  "timestamp": "2023-11-15T10:30:00Z"
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "state": "Punjab",
        "district": "Ludhiana",
        "crop_key": "Wheat",
        "year": 2023,
        "wofost_yield": 4500.0,
        "latitude": 30.9,
        "longitude": 75.85
      },
      {
        "state": "Haryana",
        "district": "Karnal",
        "crop_key": "Rice_Kh",
        "year": 2023,
        "wofost_yield": 5200.0,
        "latitude": 29.68,
        "longitude": 76.98
      }
    ]
  }'
```

### Test Prediction (CLI)

```bash
python run.py test
```

## Configuration

Edit `config/development.yaml` or `config/production.yaml`:

```yaml
model:
  type: "ensemble_gp"         # constant | xgboost | ensemble | ensemble_gp
  version: "1.0.0"
  path: "models/ensemble.pkl"
  gp_path: "models/gp_correction.pkl"
  enable_gp: true
  ensemble:
    n_splits: 5
    random_state: 42
  gp:
    noise_variance: 0.1
    training_iterations: 50
    use_gpu: false

data:
  training_data_path: "data/training/train.csv"
  ics_path: "data/ics_ratios/"
  ics_files:
    - "ics_2019_2023.json"
    - "ics_historical.json"

monitoring:
  log_predictions: true
  log_dir: "logs"
  log_buffer_size: 100
  drift_threshold: 0.15

api:
  title: "AgriTwin Bias Correction API"
  version: "1.0.0"

system:
  log_level: "INFO"
```

## Environment Variables

- `ENV`: Environment name (`development` or `production`)
- `PYTHONUNBUFFERED`: Set to `1` for real-time logging

## Model Types

| Type | Description | Use Case |
|------|-------------|----------|
| `constant` | ICS ratio multiplication | Simple baseline, requires ICS data |
| `xgboost` | Single XGBoost model | Fast fallback |
| `ensemble` | Stacked ensemble (XGB+RF+LGB) | Better accuracy, no spatial smoothing |
| `ensemble_gp` | Ensemble + GP correction | Best accuracy with uncertainty, slower |

## Training Data Schema

CSV columns:

- **Identifiers**: `state`, `district`, `crop_key`, `year`
- **Yields**: `wofost_yield` (simulated), `actual_yield` (ground truth)
- **Location**: `latitude`, `longitude`
- **Satellite**: `lai_mean`, `ndvi_mean`, `ndre_mean` (optional)
- **Weather**: `rainfall_total`, `temperature_mean`, `soil_moisture_mean` (optional)

## Monitoring

Prediction logs are written to `logs/predictions_YYYYMMDD.csv` with daily rotation.

Columns:
- timestamp, state, district, crop_key, year
- latitude, longitude
- wofost_yield, corrected_yield, correction_factor
- ics_ratio, model_version
- ci_lower, ci_upper, warnings

## Drift Detection

Use PSI (Population Stability Index) to detect feature drift:

```python
from src.monitoring.drift import calculate_psi, check_drift
import pandas as pd

reference_data = pd.read_csv("logs/predictions_20231101.csv")
current_data = pd.read_csv("logs/predictions_20231115.csv")

psi_scores = calculate_psi(
    reference_data,
    current_data,
    feature_cols=["latitude", "longitude", "wofost_yield"]
)

drift_report = check_drift(psi_scores, threshold=0.15)
print(drift_report)
```

## Docker Deployment

Build and run:

```bash
docker-compose up --build -d
```

View logs:

```bash
docker-compose logs -f
```

Update model without rebuild:

```bash
# Train new model locally
python run.py train --enable-gp

# Models are automatically picked up via volume mount
# Restart container
docker-compose restart
```

## Development

Run with auto-reload:

```bash
python run.py api --reload --log-level DEBUG
```

## Troubleshooting

**Model not loaded:**
- Ensure `models/ensemble.pkl` exists
- Run `python run.py train` first

**ICS ratios missing:**
- Check `data/ics_ratios/*.json` files exist
- Or set `model.type: "ensemble"` in config

**GPU not detected:**
- Install PyTorch with CUDA support
- Set `model.gp.use_gpu: true` in config

**Predictions too slow:**
- Use `model.type: "ensemble"` (without GP)
- Reduce `api.workers` if memory-constrained

## License

MIT
