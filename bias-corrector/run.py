#!/usr/bin/env python
"""
CLI entrypoint for AgriTwin Bias Corrector.

Commands:
    extract-ics    Extract ICS ratios from PDFs
    train          Train the bias correction model
    api            Start the FastAPI server
    test           Run a test prediction
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.config import load_config, get_project_root, get_data_path, get_ics_path, get_model_path, get_gp_model_path
from src.data.loader import load_training_data
from src.data.ics_extractor import ICSExtractor
from src.model.ensemble import StackedEnsemble
from src.model.gp_correction import DeepGPCorrection
from src.monitoring.metrics import compute_metrics
from src.utils.helpers import setup_logging


logger = logging.getLogger(__name__)


def extract_ics(args):
    """Extract ICS ratios from PDFs."""
    logger.info("=== Extracting ICS Ratios ===")
    
    config = load_config(args.env)
    ics_dir = get_ics_path(config)
    
    # PDF directory
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else get_project_root() / "data" / "raw" / "des_pdfs"
    
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        logger.info("Please download DES PDFs manually and place them in the directory.")
        return
    
    # Extract
    extractor = ICSExtractor(output_dir=ics_dir)
    years = list(range(args.start_year, args.end_year + 1)) if args.start_year and args.end_year else None
    
    ratios = extractor.extract_from_pdfs(pdf_dir, years=years)
    
    # Save
    extractor.save_ics_ratios(ratios, filename="ics_historical.json")
    
    logger.info(f"✓ Extracted {len(ratios)} ICS ratios")


def train_model(args):
    """Train the bias correction model."""
    logger.info("=== Training Bias Correction Model ===")
    
    config = load_config(args.env)
    
    # Load training data
    data_path = get_data_path(config)
    logger.info(f"Loading data from {data_path}")
    
    X_train, y_train, X_test, y_test, metadata = load_training_data(data_path)
    
    # Train ensemble
    logger.info("Training stacked ensemble...")
    ensemble = StackedEnsemble(config.model.get("ensemble", {}))
    
    train_metrics = ensemble.train(
        X_train,
        y_train,
        years=pd.Series(metadata["year"][metadata["train_mask"]])
    )
    
    logger.info("Training metrics:")
    for key, value in train_metrics.items():
        logger.info(f"  {key}: {value:.3f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred_test = ensemble.predict(X_test)
    test_metrics = compute_metrics(y_test.values, y_pred_test)
    
    # Save ensemble
    model_path = get_model_path(config)
    ensemble.save(model_path)
    logger.info(f"✓ Ensemble saved to {model_path}")
    
    # Train GP if enabled
    if args.enable_gp or config.model.get("enable_gp", False):
        logger.info("Training GP correction layer...")
        
        # Get ensemble predictions on full training data
        X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
        
        ensemble_preds = ensemble.predict(X_full)
        
        gp_model = DeepGPCorrection(config.model.get("gp", {}))
        gp_model.train(
            ensemble_preds=ensemble_preds,
            true_yields=y_full.values,
            latitudes=metadata["latitude"],
            longitudes=metadata["longitude"],
            years=metadata["year"]
        )
        
        # Save GP model
        gp_path = get_gp_model_path(config)
        gp_model.save(gp_path)
        logger.info(f"✓ GP model saved to {gp_path}")
        
        # Evaluate GP correction
        gp_corrected, _ = gp_model.predict(
            ensemble_preds,
            metadata["latitude"],
            metadata["longitude"],
            metadata["year"]
        )
        gp_metrics = compute_metrics(y_full.values, gp_corrected)
        logger.info("GP-corrected metrics:")
        for key, value in gp_metrics.items():
            logger.info(f"  {key}: {value:.3f}")
    
    logger.info("✓ Training complete!")


def start_api(args):
    """Start the FastAPI server."""
    import uvicorn
    
    logger.info("=== Starting Bias Corrector API ===")
    
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level.lower()
    )


def test_prediction(args):
    """Run a test prediction."""
    logger.info("=== Running Test Prediction ===")
    
    from src.api.schemas import PredictionRequest, CropKey
    from src.model.correction import CorrectionModel
    
    config = load_config(args.env)
    
    # Load model
    model = CorrectionModel(config)
    model.load()
    logger.info("Model loaded")
    
    # Create test request
    test_request = PredictionRequest(
        state="Punjab",
        district="Ludhiana",
        crop_key=CropKey.WHEAT,
        year=2023,
        wofost_yield=4500.0,
        latitude=30.9,
        longitude=75.85,
        lai_mean=3.5,
        ndvi_mean=0.75,
        ndre_mean=0.25,
        rainfall_total=650.0,
        temperature_mean=22.5,
        soil_moisture_mean=0.28
    )
    
    logger.info(f"Test request: {test_request.state}/{test_request.district} {test_request.crop_key.value} {test_request.year}")
    logger.info(f"WOFOST yield: {test_request.wofost_yield} kg/ha")
    
    # Predict
    result = model.predict(test_request)
    
    logger.info("Result:")
    logger.info(f"  Original yield: {result.original_yield:.2f} kg/ha")
    logger.info(f"  Corrected yield: {result.corrected_yield:.2f} kg/ha")
    logger.info(f"  Correction factor: {result.correction_factor:.3f}")
    if result.ics_ratio:
        logger.info(f"  ICS ratio: {result.ics_ratio:.3f}")
    if result.confidence_interval:
        logger.info(f"  Confidence interval: [{result.confidence_interval.lower:.2f}, {result.confidence_interval.upper:.2f}]")
    if result.warnings:
        logger.warning(f"  Warnings: {result.warnings}")
    
    logger.info("✓ Test complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AgriTwin Bias Corrector CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--env",
        default="development",
        choices=["development", "production"],
        help="Environment configuration"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # extract-ics command
    extract_parser = subparsers.add_parser("extract-ics", help="Extract ICS ratios from PDFs")
    extract_parser.add_argument("--pdf-dir", help="Directory containing DES PDFs")
    extract_parser.add_argument("--start-year", type=int, help="Start year (inclusive)")
    extract_parser.add_argument("--end-year", type=int, help="End year (inclusive)")
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train bias correction model")
    train_parser.add_argument("--enable-gp", action="store_true", help="Enable GP correction layer")
    
    # api command
    api_parser = subparsers.add_parser("api", help="Start FastAPI server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    api_parser.add_argument("--port", type=int, default=8000, help="Port number")
    api_parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    
    # test command
    test_parser = subparsers.add_parser("test", help="Run test prediction")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command == "extract-ics":
        extract_ics(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "api":
        start_api(args)
    elif args.command == "test":
        test_prediction(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
