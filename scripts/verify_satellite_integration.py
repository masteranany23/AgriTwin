#!/usr/bin/env python3
"""
scripts/verify_satellite_integration.py — Verify Satellite Integration
=======================================================================

This script verifies that the satellite fetcher is properly integrated
with all AgriTwin components.

Usage:
    python scripts/verify_satellite_integration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    tests = []
    
    # Core satellite fetcher
    try:
        from backend.app.services.satellite_fetcher import SatelliteFetcher
        tests.append(("✓", "SatelliteFetcher service"))
    except ImportError as e:
        tests.append(("✗", f"SatelliteFetcher service: {e}"))
    
    # API routes
    try:
        from backend.app.satellite.api.routes import router
        tests.append(("✓", "Satellite API routes"))
    except ImportError as e:
        tests.append(("✗", f"Satellite API routes: {e}"))
    
    # Data fusion integration
    try:
        from backend.app.services.data_fusion_pipeline import DataFusionPipeline
        tests.append(("✓", "DataFusionPipeline"))
    except ImportError as e:
        tests.append(("✗", f"DataFusionPipeline: {e}"))
    
    # Multi-source fusion
    try:
        from backend.app.services.multi_source_fusion_service import MultiSourceFusionService
        tests.append(("✓", "MultiSourceFusionService"))
    except ImportError as e:
        tests.append(("✗", f"MultiSourceFusionService: {e}"))
    
    # Observation models
    try:
        from backend.app.assimilation.models.observation import Observation, ObservationSource
        tests.append(("✓", "Observation models"))
    except ImportError as e:
        tests.append(("✗", f"Observation models: {e}"))
    
    # Observation repository
    try:
        from backend.app.assimilation.repositories.observation_repository import ObservationRepository
        tests.append(("✓", "ObservationRepository"))
    except ImportError as e:
        tests.append(("✗", f"ObservationRepository: {e}"))
    
    # Print results
    for status, message in tests:
        print(f"  {status} {message}")
    
    passed = sum(1 for status, _ in tests if status == "✓")
    total = len(tests)
    print(f"\nResult: {passed}/{total} imports successful")
    
    return passed == total


def test_sentinelhub_availability():
    """Test SentinelHub package availability."""
    print("\n" + "="*70)
    print("TEST 2: SentinelHub Package")
    print("="*70)
    
    try:
        import sentinelhub
        print(f"  ✓ sentinelhub package installed (version: {sentinelhub.__version__})")
        
        from sentinelhub import SHConfig
        config = SHConfig()
        
        if config.sh_client_id and config.sh_client_secret:
            print(f"  ✓ SentinelHub credentials configured")
            print(f"    Client ID: {config.sh_client_id[:10]}...")
            has_credentials = True
        else:
            print(f"  ⚠ SentinelHub credentials NOT configured")
            print(f"    Set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables")
            has_credentials = False
        
        return has_credentials
        
    except ImportError:
        print(f"  ✗ sentinelhub package NOT installed")
        print(f"    Install with: pip install sentinelhub")
        return False


def test_database_integration():
    """Test database integration."""
    print("\n" + "="*70)
    print("TEST 3: Database Integration")
    print("="*70)
    
    try:
        from backend.app.db.session import engine, get_db
        from backend.app.models.field import Field
        from backend.app.assimilation.models.observation import Observation
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(__import__('sqlalchemy').text("SELECT 1"))
        
        print(f"  ✓ Database connection successful")
        print(f"  ✓ Field model accessible")
        print(f"  ✓ Observation model accessible")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        return False


def test_api_routes_mounted():
    """Test that satellite routes are mounted in FastAPI app."""
    print("\n" + "="*70)
    print("TEST 4: API Routes Mounted")
    print("="*70)
    
    try:
        from backend.app.main import app
        
        # Check if satellite routes are in the app
        route_paths = [route.path for route in app.routes]
        
        expected_routes = [
            "/satellite/fetch/{field_id}",
            "/satellite/fetch-batch",
            "/satellite/status/{field_id}",
        ]
        
        tests = []
        for expected in expected_routes:
            if expected in route_paths:
                tests.append(("✓", f"Route mounted: {expected}"))
            else:
                tests.append(("✗", f"Route NOT found: {expected}"))
        
        for status, message in tests:
            print(f"  {status} {message}")
        
        passed = sum(1 for status, _ in tests if status == "✓")
        total = len(tests)
        
        return passed == total
        
    except Exception as e:
        print(f"  ✗ Error checking routes: {e}")
        return False


def test_data_fusion_connection():
    """Test that satellite fetcher connects to data fusion."""
    print("\n" + "="*70)
    print("TEST 5: Data Fusion Integration")
    print("="*70)
    
    try:
        from backend.app.services.satellite_fetcher import SatelliteFetcher
        from backend.app.services.data_fusion_pipeline import DataFusionPipeline
        
        # Check if SatelliteFetcher imports DataFusionPipeline
        import inspect
        source = inspect.getsource(SatelliteFetcher)
        
        tests = [
            ("DataFusionPipeline" in source, "DataFusionPipeline import"),
            ("_push_to_data_fusion" in source, "_push_to_data_fusion method"),
            ("add_observation" in source, "add_observation call"),
        ]
        
        for condition, description in tests:
            if condition:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description} NOT found")
        
        passed = sum(1 for condition, _ in tests if condition)
        return passed == len(tests)
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_confidence_scoring():
    """Test confidence scoring logic."""
    print("\n" + "="*70)
    print("TEST 6: Confidence Scoring")
    print("="*70)
    
    try:
        from backend.app.services.confidence_estimator import ConfidenceEstimator
        from backend.app.api.schemas.fusion import ConfidenceRequest, ObservationSource
        
        estimator = ConfidenceEstimator()
        
        # Test clear sky
        req_clear = ConfidenceRequest(
            source=ObservationSource.SENTINEL2,
            value=3.5,
            cloud_cover=0.05,
            viewing_angle=0,
            sensor_health=1.0,
            days_since_observation=0
        )
        resp_clear = estimator.compute_confidence(req_clear)
        
        # Test cloudy
        req_cloudy = ConfidenceRequest(
            source=ObservationSource.SENTINEL2,
            value=3.5,
            cloud_cover=0.8,
            viewing_angle=0,
            sensor_health=1.0,
            days_since_observation=0
        )
        resp_cloudy = estimator.compute_confidence(req_cloudy)
        
        print(f"  ✓ ConfidenceEstimator available")
        print(f"  ✓ Clear sky confidence: {resp_clear.confidence_score:.3f}")
        print(f"  ✓ Cloudy confidence: {resp_cloudy.confidence_score:.3f}")
        
        # Clear sky should have higher confidence
        if resp_clear.confidence_score > resp_cloudy.confidence_score:
            print(f"  ✓ Confidence scoring works correctly")
            return True
        else:
            print(f"  ✗ Confidence scoring error")
            return False
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_cli_script():
    """Test CLI script exists and is executable."""
    print("\n" + "="*70)
    print("TEST 7: CLI Script")
    print("="*70)
    
    cli_path = Path(__file__).parent / "fetch_satellite_data.py"
    
    if cli_path.exists():
        print(f"  ✓ CLI script exists: {cli_path.name}")
        
        # Check if it has main function
        with open(cli_path, 'r') as f:
            content = f.read()
            if 'def main()' in content:
                print(f"  ✓ main() function found")
            else:
                print(f"  ⚠ main() function not found")
            
            if 'argparse' in content:
                print(f"  ✓ Uses argparse for CLI")
            else:
                print(f"  ⚠ argparse not found")
        
        return True
    else:
        print(f"  ✗ CLI script not found at: {cli_path}")
        return False


def main():
    """Run all verification tests."""
    print("="*70)
    print("Satellite Integration Verification")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("SentinelHub Package", test_sentinelhub_availability()))
    results.append(("Database Integration", test_database_integration()))
    results.append(("API Routes", test_api_routes_mounted()))
    results.append(("Data Fusion Connection", test_data_fusion_connection()))
    results.append(("Confidence Scoring", test_confidence_scoring()))
    results.append(("CLI Script", test_cli_script()))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} — {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✅ All integration tests PASSED!")
        print("Satellite Fetcher is fully integrated and ready to use.")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed.")
        print("Review failed tests above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
