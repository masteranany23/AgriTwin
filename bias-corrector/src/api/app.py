"""
FastAPI application for AgriTwin Bias Correction API.
"""
import logging
import time


from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import load_config, get_project_root
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from ..model.correction import CorrectionModel
from ..monitoring.logger import PredictionLogger


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global state
app_state = {
    "config": None,
    "model": None,
    "prediction_logger": None,
    "start_time": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting AgriTwin Bias Correction API...")
    app_state["start_time"] = time.time()
    
    try:
        # Load configuration
        config = load_config()
        app_state["config"] = config
        logger.info(f"Configuration loaded: {config.model['type']}")
        
        # Initialize correction model
        model = CorrectionModel(config)
        model.load()
        app_state["model"] = model
        logger.info(f"Model loaded: {config.model['version']}")
        
        # Initialize prediction logger
        if config.monitoring["log_predictions"]:
            log_dir = get_project_root() / config.monitoring["log_dir"]
            app_state["prediction_logger"] = PredictionLogger(
                log_dir=log_dir,
                buffer_size=config.monitoring["log_buffer_size"]
            )
            logger.info("Prediction logger initialized")
        
        logger.info("API startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    if app_state["prediction_logger"]:
        app_state["prediction_logger"].shutdown()


# Initialize FastAPI app
config = load_config()
app = FastAPI(
    title=config.api["title"],
    version=config.api["version"],
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "message": str(exc)
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Service health status.
    """
    config = app_state["config"]
    model = app_state["model"]
    
    uptime = time.time() - app_state["start_time"]
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        version=config.model["version"],
        model_loaded=model is not None,
        model_type=config.model["type"],
        ics_ratios_loaded=model.ics_ratios is not None if model else False,
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate bias-corrected yield prediction.
    
    Args:
        request: Prediction request with WOFOST yield and features.
        
    Returns:
        PredictionResponse: Corrected yield with confidence interval.
        
    Raises:
        HTTPException: If prediction fails.
    """
    model = app_state["model"]
    prediction_logger = app_state["prediction_logger"]
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Generate prediction
        logger.debug(f"Processing prediction for {request.state}/{request.district} {request.crop_key} {request.year}")
        result = model.predict(request)
        
        # Log prediction
        if prediction_logger:
            await prediction_logger.log_prediction(request, result)
        
        return result
        
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Generate batch bias-corrected predictions.
    
    Args:
        request: Batch prediction request.
        
    Returns:
        BatchPredictionResponse: Results for all predictions.
        
    Raises:
        HTTPException: If batch processing fails.
    """
    model = app_state["model"]
    prediction_logger = app_state["prediction_logger"]
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    
    logger.info(f"Processing batch of {len(request.predictions)} predictions")
    
    for pred_request in request.predictions:
        try:
            result = model.predict(pred_request)
            results.append(result)
            successful += 1
            
            # Log prediction
            if prediction_logger:
                await prediction_logger.log_prediction(pred_request, result)
                
        except Exception as e:
            logger.error(f"Failed prediction in batch: {e}")
            # Add failed prediction with warnings
            results.append(
                PredictionResponse(
                    original_yield=pred_request.wofost_yield,
                    corrected_yield=pred_request.wofost_yield,
                    correction_factor=1.0,
                    model_version=app_state["config"].model["version"],
                    warnings=[f"Prediction failed: {str(e)}"]
                )
            )
            failed += 1
    
    processing_time = time.time() - start_time
    
    logger.info(f"Batch complete: {successful}/{len(request.predictions)} successful ({processing_time:.2f}s)")
    
    return BatchPredictionResponse(
        predictions=results,
        total=len(request.predictions),
        successful=successful,
        failed=failed,
        processing_time_seconds=processing_time
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AgriTwin Bias Correction API",
        "version": app_state["config"].model["version"],
        "docs": "/docs",
        "health": "/health"
    }
