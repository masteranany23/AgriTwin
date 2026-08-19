"""
ERA5-Land Source - Simplified wrapper for ERA5LandWeatherSource

This module provides a simplified interface to ERA5-Land weather data.
It's used by the weather service for fallback and hybrid strategies.
"""


import logging
from pathlib import Path
from typing import Optional

from .era5_land_weather_source import ERA5LandWeatherSource

logger = logging.getLogger(__name__)


class ERA5LandSource:
    """
    Simplified ERA5-Land source for integration with weather service.
    
    Usage:
        source = ERA5LandSource()
        if source.is_available():
            provider = source.fetch_weather(lat, lon, start, end)
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize ERA5-Land source.
        
        Args:
            data_dir: Path to ERA5 NetCDF files. If None, uses default location.
        """
        if data_dir is None:
            # Default to data/raw/era5_land in project root
            project_root = Path(__file__).parent.parent.parent.parent
            data_dir = project_root / "data" / "raw" / "era5_land"
        
        self.data_dir = Path(data_dir)
        self._source = None
        self._available = None
    
    def is_available(self) -> bool:
        """
        Check if ERA5-Land data is available.
        
        Returns:
            True if ERA5 NetCDF files exist and xarray is installed
        """
        if self._available is not None:
            return self._available
        
        # Check if data directory exists
        if not self.data_dir.exists():
            logger.debug(f"ERA5 data directory not found: {self.data_dir}")
            self._available = False
            return False
        
        # Check if any NetCDF files exist
        nc_files = list(self.data_dir.glob("*.nc"))
        if not nc_files:
            logger.debug(f"No NetCDF files found in {self.data_dir}")
            self._available = False
            return False
        
        # Check if xarray is available
        try:
            import xarray
            self._available = True
            logger.info(f"ERA5-Land available with {len(nc_files)} NetCDF files")
            return True
        except ImportError:
            logger.debug("xarray not installed - ERA5 unavailable")
            self._available = False
            return False
    
    def get_source(self) -> ERA5LandWeatherSource:
        """
        Get ERA5LandWeatherSource instance.
        
        Returns:
            ERA5LandWeatherSource
        
        Raises:
            RuntimeError: If ERA5 data not available
        """
        if not self.is_available():
            raise RuntimeError(
                f"ERA5-Land data not available. "
                f"Expected NetCDF files in: {self.data_dir}"
            )
        
        if self._source is None:
            self._source = ERA5LandWeatherSource(str(self.data_dir))
        
        return self._source
    
    def fetch_weather(self, latitude, longitude, start_date, end_date, elevation=10.0):
        """
        Fetch weather data using ERA5-Land.
        
        Args:
            latitude: Site latitude
            longitude: Site longitude
            start_date: Start date
            end_date: End date
            elevation: Site elevation (optional)
        
        Returns:
            WeatherDataProvider
        
        Raises:
            RuntimeError: If ERA5 not available or fetch fails
        """
        source = self.get_source()
        return source.get_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            elevation=elevation
        )


# Singleton instance for global use
_era5_source = None


def get_era5_source(data_dir: Optional[str] = None) -> ERA5LandSource:
    """
    Get singleton ERA5LandSource instance.
    
    Args:
        data_dir: Optional custom data directory
    
    Returns:
        ERA5LandSource instance
    """
    global _era5_source
    if _era5_source is None:
        _era5_source = ERA5LandSource(data_dir)
    return _era5_source


def is_era5_available() -> bool:
    """
    Quick check if ERA5-Land is available.
    
    Returns:
        True if ERA5 data and dependencies available
    """
    source = get_era5_source()
    return source.is_available()
