#!/usr/bin/env python3
"""
scripts/fetch_satellite_data.py — Scheduled Satellite Data Fetcher
===================================================================

CLI script for automated satellite data fetching.
Can be run as a cron job or scheduled task.

Usage:
    # Fetch for all fields in the last 30 days
    python scripts/fetch_satellite_data.py --days-back 30
    
    # Fetch for specific field
    python scripts/fetch_satellite_data.py --field-id 123e4567-e89b-12d3-a456-426614174000
    
    # Fetch with custom cloud cover threshold
    python scripts/fetch_satellite_data.py --days-back 30 --max-cloud-cover 0.3
    
    # Dry run (test without saving)
    python scripts/fetch_satellite_data.py --days-back 30 --dry-run

Cron Example (daily at 2 AM):
    0 2 * * * cd /path/to/AgriTwin && python scripts/fetch_satellite_data.py --days-back 7

Windows Task Scheduler:
    Program: python
    Arguments: C:\path\to\AgriTwin\scripts\fetch_satellite_data.py --days-back 7
    Start in: C:\path\to\AgriTwin
"""

import argparse
import logging
import sys
import uuid
import datetime as dt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session

from backend.app.db.session import get_db, create_tables
from backend.app.models.field import Field
from backend.app.services.satellite_fetcher import SatelliteFetcher


# ── Logging Configuration ─────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-5s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI Functions ─────────────────────────────────────────────────────────────

def fetch_for_all_fields(
    db: Session,
    days_back: int,
    max_cloud_cover: float,
    dry_run: bool = False,
) -> None:
    """Fetch satellite data for all fields in the database.
    
    Args:
        db: Database session
        days_back: Number of days to look back
        max_cloud_cover: Maximum cloud cover threshold
        dry_run: If True, simulate without saving
    """
    # Get all fields
    fields = db.query(Field).all()
    
    if not fields:
        logger.warning("No fields found in database")
        return
    
    logger.info("Found %d fields to process", len(fields))
    
    # Calculate date range
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=days_back)
    
    logger.info("Fetching data from %s to %s", start_date, end_date)
    
    # Initialize fetcher
    if dry_run:
        logger.info("DRY RUN MODE - No data will be saved")
        return
    
    fetcher = SatelliteFetcher(max_cloud_cover=max_cloud_cover)
    
    # Process each field
    total_observations = 0
    failed_fields = []
    
    for field in fields:
        try:
            if not field.boundary_geojson:
                logger.warning("Field %s (%s) has no boundary, skipping", field.id, field.name)
                continue
            
            logger.info("Processing field: %s (%s)", field.name, field.id)
            
            results = fetcher.fetch_for_field(field.id, start_date, end_date, db)
            
            total_observations += len(results)
            
            logger.info(
                "Field %s: %d observations created",
                field.name, len(results)
            )
            
        except Exception as e:
            logger.error("Error processing field %s: %s", field.name, e)
            failed_fields.append((field.name, str(e)))
    
    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Total fields processed: %d", len(fields))
    logger.info("Total observations created: %d", total_observations)
    logger.info("Failed fields: %d", len(failed_fields))
    
    if failed_fields:
        logger.warning("Failed fields:")
        for field_name, error in failed_fields:
            logger.warning("  - %s: %s", field_name, error)


def fetch_for_single_field(
    db: Session,
    field_id: uuid.UUID,
    days_back: int,
    max_cloud_cover: float,
    dry_run: bool = False,
) -> None:
    """Fetch satellite data for a single field.
    
    Args:
        db: Database session
        field_id: Field UUID
        days_back: Number of days to look back
        max_cloud_cover: Maximum cloud cover threshold
        dry_run: If True, simulate without saving
    """
    # Get field
    field = db.get(Field, field_id)
    if field is None:
        logger.error("Field %s not found", field_id)
        sys.exit(1)
    
    if not field.boundary_geojson:
        logger.error("Field %s has no boundary geometry", field_id)
        sys.exit(1)
    
    logger.info("Processing field: %s (%s)", field.name, field_id)
    
    # Calculate date range
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=days_back)
    
    logger.info("Fetching data from %s to %s", start_date, end_date)
    
    if dry_run:
        logger.info("DRY RUN MODE - No data will be saved")
        return
    
    # Fetch data
    fetcher = SatelliteFetcher(max_cloud_cover=max_cloud_cover)
    
    try:
        results = fetcher.fetch_for_field(field.id, start_date, end_date, db)
        
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info("Field: %s (%s)", field.name, field.id)
        logger.info("Observations created: %d", len(results))
        
        if results:
            logger.info("Sample observations:")
            for i, result in enumerate(results[:5], 1):
                logger.info(
                    "  %d. Date: %s, NDRE: %.3f, LAI: %.3f, Cloud: %.1f%%, Confidence: %.2f",
                    i, result['date'], result['ndre'], result['lai'],
                    result['cloud_cover'] * 100, result['confidence_score']
                )
        
    except Exception as e:
        logger.error("Error fetching data: %s", e, exc_info=True)
        sys.exit(1)


def test_sentinelhub_connection() -> bool:
    """Test SentinelHub API connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        from sentinelhub import SHConfig
        
        config = SHConfig()
        
        if not config.sh_client_id or not config.sh_client_secret:
            logger.error(
                "SentinelHub credentials not configured. "
                "Set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables."
            )
            return False
        
        logger.info("SentinelHub credentials found")
        logger.info("Instance ID: %s", config.instance_id or "Not set")
        
        return True
        
    except ImportError:
        logger.error("sentinelhub package not installed")
        return False
    except Exception as e:
        logger.error("Error testing SentinelHub connection: %s", e)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch Sentinel-2 satellite data for AgriTwin fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--field-id",
        type=str,
        help="Fetch data for a specific field UUID (if not provided, processes all fields)"
    )
    
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Number of days to look back (default: 30)"
    )
    
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=0.2,
        help="Maximum cloud cover threshold 0.0-1.0 (default: 0.2)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test run without saving data"
    )
    
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test SentinelHub API connection and exit"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Test connection if requested
    if args.test_connection:
        success = test_sentinelhub_connection()
        sys.exit(0 if success else 1)
    
    # Validate arguments
    if args.max_cloud_cover < 0.0 or args.max_cloud_cover > 1.0:
        logger.error("max_cloud_cover must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.days_back < 1:
        logger.error("days_back must be at least 1")
        sys.exit(1)
    
    # Initialize database
    logger.info("Initializing database connection...")
    create_tables()
    
    # Get database session
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        # Process based on arguments
        if args.field_id:
            # Parse UUID
            try:
                field_uuid = uuid.UUID(args.field_id)
            except ValueError:
                logger.error("Invalid field UUID: %s", args.field_id)
                sys.exit(1)
            
            fetch_for_single_field(
                db,
                field_uuid,
                args.days_back,
                args.max_cloud_cover,
                args.dry_run,
            )
        else:
            fetch_for_all_fields(
                db,
                args.days_back,
                args.max_cloud_cover,
                args.dry_run,
            )
    
    finally:
        db.close()
    
    logger.info("Satellite data fetch complete")


if __name__ == "__main__":
    main()
