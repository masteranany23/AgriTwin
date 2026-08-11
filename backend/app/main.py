"""
main.py — AgriTwin FastAPI Application Entry Point
===================================================

This file:
  1. Creates the FastAPI app instance with metadata from core/config.py
  2. Configures logging
  3. Adds CORS middleware
  4. Mounts all route routers with their URL prefixes
  5. Defines the /health check endpoint

How to run:
    cd /home/vini/Arena/AgriTwin
    source venv/bin/activate
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

API documentation (auto-generated):
    http://localhost:8000/docs     — Swagger UI (interactive)
    http://localhost:8000/redoc   — ReDoc (clean reference)

Documentation Files:
    docs/w_shape_grvi_protocol.md     — W-Shape GRVI Protocol (Smartphone LAI)
    docs/satellite_ndre_fetcher.md    — Sentinel-2 NDRE Automated Fetching
    docs/era5_land_integration.md     — ERA5-Land Weather & 4-Layer Soil Moisture
    docs/enkf_design.md               — Ensemble Kalman Filter Design
    docs/state_variables.md           — WOFOST State Variables Reference

Backend module structure:
    backend/app/
    ├── main.py                  ← You are here (app factory + router mounting)
    ├── core/
    │   ├── config.py            ← Centralized settings (env vars)
    │   └── exceptions.py        ← Custom exception hierarchy
    ├── api/
    │   ├── routes/
    │   │   ├── simulate.py      ← POST /simulate, GET /simulate/crops
    │   │   ├── simulations.py   ← GET /simulations, GET /simulations/{id}
    │   │   ├── fields.py        ← Field CRUD endpoints
    │   │   ├── scout_sessions.py ← W-Shape GRVI protocol (5 photos → LAI)
    │   │   ├── fusion.py        ← Data fusion pipeline (Module 3.3)
    │   │   └── interpolation.py ← Temporal interpolation
    │   └── schemas/
    │       └── simulate.py      ← Pydantic request/response models
    ├── services/
    │   ├── simulation_service.py ← Orchestrates WOFOST run
    │   ├── weather_service.py   ← Hybrid router (ERA5-Land + NASA POWER)
    │   └── soil_service.py      ← SoilGrids API + caching
    ├── simulation/
    │   └── engine.py            ← Core WOFOST run_simulation() function
    ├── assimilation/
    │   ├── filters/enkf.py      ← Ensemble Kalman Filter implementation
    │   └── api/                 ← Assimilation endpoints
    ├── satellite/
    │   ├── providers/           ← Sentinel-2, Sentinel-1 providers
    │   └── api/routes.py        ← Satellite LAI fetch endpoints
    └── scenario/
        └── api/                 ← Scenario sweep endpoints

Currently mounted routers:
    - simulate.router         → /simulate           (WOFOST simulation)
    - simulations_router      → /simulations        (Simulation history)
    - fields_router          → /fields              (Field CRUD)
    - scout_sessions_router  → /fields/{id}/scout-session  (W-Shape GRVI)
    - scenario_router        → /scenarios           (Sowing date sweeps)
    - observations_router    → /observations        (Field observations)
    - satellite_router       → /satellite           (Sentinel-2 NDRE fetch)
    - assimilation_router    → /assimilation        (EnKF data assimilation)
    - fusion_router          → /fusion              (Data fusion pipeline)
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.api.routes import simulate
from backend.app.api.routes.simulations import router as simulations_router
from backend.app.api.routes.fields import router as fields_router
from backend.app.api.routes.scout_sessions import router as scout_sessions_router
from backend.app.scenario.api.scenario_routes import router as scenario_router
from backend.app.assimilation.api.observation_routes import router as observations_router
from backend.app.assimilation.api.assimilation_routes import router as assimilation_router
from backend.app.satellite.api.routes import router as satellite_router
from backend.app.api.routes.fusion import router as fusion_router
from backend.app.api.routes.benchmark_routes import router as benchmark_router
from backend.app.db.session import create_tables

# ── Logging ───────────────────────────────────────────────────────────────────
# Configure once at startup. All loggers in the application inherit this config.
# Change LOG_LEVEL in core/config.py (or set AGRITWIN_LOG_LEVEL env var)
# to "DEBUG" for verbose PCSE internal messages.
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s │ %(name)-42s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("AgriTwin API starting — version %s", settings.APP_VERSION)


# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Simulation",
            "description": (
                "WOFOST 7.2 crop simulation endpoints. "
                "Run physics-based crop simulations using NASA POWER weather "
                "and SoilGrids soil data."
            ),
        },
        {
            "name": "Simulations",
            "description": (
                "Retrieve and manage stored simulation history. "
                "Query past runs, download time series, and delete records."
            ),
        },
        {
            "name": "Fields",
            "description": (
                "CRUD for Field records — GPS-located agricultural plots. "
                "Fields group simulation runs by physical location."
            ),
        },
        {
            "name": "Scout Sessions",
            "description": (
                "W-Shape GRVI Protocol — Farmer smartphone-based LAI monitoring. "
                "Upload 5 photos in W-shape pattern across field. Backend computes "
                "median GRVI (Green-Red Vegetation Index) and converts to LAI estimate. "
                "30% observation error ('Gentle Nudge') for EnKF assimilation. "
                "See docs/w_shape_grvi_protocol.md for full protocol."
            ),
        },
        {
            "name": "Observations",
            "description": (
                "Ingest and query field observations from any source: "
                "Sentinel-2 satellite, soil moisture sensors, weather stations, "
                "manual field scouts, and model-derived pseudo-observations. "
                "Observation records are the data foundation for future EnKF assimilation."
            ),
        },
        {
            "name": "Data Fusion (Module 3.3)",
            "description": (
                "Complete data fusion pipeline: temporal interpolation with cloud-gap detection, "
                "spatial alignment, confidence estimation, and multi-source Bayesian fusion. "
                "Combines satellite, weather, and ground observations into unified high-quality data streams."
            ),
        },
        {
            "name": "System",
            "description": "Health check and service metadata endpoints.",
        },
    ],
)


# ── Database startup ──────────────────────────────────────────────────────────
# Create all tables on startup (idempotent — safe to call on every boot).
# In production with PostgreSQL, replace this with Alembic migrations.
@app.on_event("startup")
def on_startup() -> None:
    """Initialise the database schema on server start when configured.

    In production, database migrations are managed exclusively by Alembic.
    Automatic table creation is enabled only in development/testing environments
    when AUTO_CREATE_TABLES is enabled.
    """
    if settings.AUTO_CREATE_TABLES:
        create_tables()
        logger.info("Database tables verified / created.")
    else:
        logger.info("Skipping automatic table creation; relying on Alembic migrations.")


# ── CORS middleware ───────────────────────────────────────────────────────────
# Allows the frontend (React/Next.js dashboard) to call this API from a browser.
# In development: allow all origins.
# In production: restrict to known origins in settings.CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Route mounting ────────────────────────────────────────────────────────────
# Each router handles a logical feature domain. The prefix defines the URL
# namespace. Routes inside each router file define the suffix.
#
# Currently mounted:
#   simulate.router      → POST /simulate, GET /simulate/crops
#   simulations_router   → GET /simulations, GET /simulations/{id}, DELETE /simulations/{id}
#   fields_router        → GET /fields, POST /fields, GET /fields/{id}, DELETE /fields/{id}
app.include_router(
    simulate.router,
    prefix="/simulate",
    tags=["Simulation"],
)
app.include_router(
    simulations_router,
    prefix="/simulations",
    tags=["Simulations"],
)
app.include_router(
    fields_router,
    prefix="/fields",
    tags=["Fields"],
)
app.include_router(
    scout_sessions_router,
    prefix="/fields",
    tags=["Scout Sessions"],
)
app.include_router(
    scenario_router,
    prefix="/scenarios",
    tags=["Scenarios"],
)
app.include_router(
    observations_router,
    prefix="/observations",
    tags=["Observations"],
)
app.include_router(
    satellite_router,
    prefix="/satellite",
    tags=["Satellite"],
)
app.include_router(
    assimilation_router,
    prefix="/assimilation",
    tags=["Assimilation"],
)
 
app.include_router(
    fusion_router,
    prefix="/fusion",
    tags=["Data Fusion (Module 3.3)"],
)

app.include_router(
    benchmark_router,
    prefix="/benchmark",
    tags=["Benchmarking"],
)


# ── Health check endpoint ─────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["System"],
    summary="Service health check",
    description=(
        "Returns the current status of the AgriTwin API service. "
        "Suitable for use by container orchestrators (Kubernetes, Docker Compose) "
        "as a liveness probe."
    ),
)
def health_check() -> dict:
    """Return service health and database connectivity status."""
    from backend.app.db.session import engine
    db_status = "unknown"
    try:
        with engine.connect() as conn:
            conn.execute(__import__('sqlalchemy').text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    return {
        "status": "ok",
        "service": "agritwin",
        "version": settings.APP_VERSION,
        "database": db_status,
    }
