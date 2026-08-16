"""
tests/test_production_startup.py
=================================

Tests for production startup configuration and database table creation settings.
"""

import os
from unittest.mock import patch

from backend.app.core.config import Settings, settings
from backend.app.main import on_startup


def test_auto_create_tables_setting_defaults():
    """Verify AUTO_CREATE_TABLES defaults based on ENVIRONMENT."""
    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        os.environ.pop("AUTO_CREATE_TABLES", None)
        s = Settings()
        assert s.AUTO_CREATE_TABLES is False

    with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
        os.environ.pop("AUTO_CREATE_TABLES", None)
        s = Settings()
        assert s.AUTO_CREATE_TABLES is True


def test_auto_create_tables_override():
    """Verify AUTO_CREATE_TABLES environment variable override."""
    with patch.dict(os.environ, {"ENVIRONMENT": "production", "AUTO_CREATE_TABLES": "true"}):
        s = Settings()
        assert s.AUTO_CREATE_TABLES is True

    with patch.dict(os.environ, {"ENVIRONMENT": "development", "AUTO_CREATE_TABLES": "false"}):
        s = Settings()
        assert s.AUTO_CREATE_TABLES is False


def test_on_startup_production_skips_create_tables():
    """Verify on_startup skips table creation when ENVIRONMENT is production."""
    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        os.environ.pop("AUTO_CREATE_TABLES", None)
        with patch("backend.app.main.create_tables") as mock_create_tables:
            on_startup()
            mock_create_tables.assert_not_called()


def test_on_startup_dev_calls_create_tables():
    """Verify on_startup calls table creation when ENVIRONMENT is development."""
    with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
        os.environ.pop("AUTO_CREATE_TABLES", None)
        with patch("backend.app.main.create_tables") as mock_create_tables:
            on_startup()
            mock_create_tables.assert_called_once()
