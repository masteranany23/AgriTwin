"""
ERA5-Land Weather Service
=========================

Service layer for ERA5-Land weather data integration.
Provides high-level API for fetching ERA5-Land weather with fallback to NASA POWER.

Features:
- Automatic source selection (ERA5 → NASA POWER fallback)
- Caching support
- Error handling and retry logic
- Integration with WOFOST simulation engine
"""


import logging
import datetime as dt
from typing import Optional


from pcse.base import WeatherDataProvider

from backend.app.data_sources.era5_land_source import ERA5LandSource
from backend.app.services.weather_service import WeatherService

logger = logging.getLogger(__name__)


class ERA5LandWeatherService:
    """
    High-level service for ERA5-Land weather data with NASA POWER fallback.
    
    This service implements a hybrid strategy:
    1. Try ERA5-Land first (higher resolution, more accurate)
    2. Fall back to NASA POWER if ERA5 unavailable or fails
    
    Usage:
        service = ERA5LandWeatherService()
        provider = service.get_weather(
            latitude=30.9,
            longitude=75.85,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )
    """
    
    def __init__(
        self,
        era5_data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        prefer_era5: bool = True
    ):
        """
        Initialize ERA5-Land weather service.
        
        Args:
            era5_data_dir: Directory with ERA5-Land NetCDF files
            cache_dir: Cache directory for weather data
            prefer_era5: If True, try ERA5 first; if False, use NASA POWER only
        """
        self.prefer_era5 = prefer_era5
        
        # Initialize ERA5 source
        self.era5_source = None
        if prefer_era5:
            try:
                self.era5_source = ERA5LandSource(era5_data_dir)
                if self.era5_source.is_available():
                    logger.info("ERA5-Land source initialized and available")
                else:
                    logger.info("ERA5-Land data not available, will use NASA POWER")
                    self.era5_source = None
            except Exception as e:
                logger.warning(f"Failed to initialize ERA5 source: {e}")
                self.era5_source = None
        
        # Initialize NASA POWER fallback
        self.nasa_service = WeatherService(cache_dir=cache_dir)
        logger.info("NASA POWER fallback initialized")
    
    def get_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: dt.date,
        end_date: dt.date,
        elevation: float = 10.0,
        force_nasa: bool = False
    ) -> tuple[WeatherDataProvider, str]:
        """
        Get weather data with automatic source selection.
        
        Args:
            latitude: Site latitude (-90 to 90)
            longitude: Site longitude (-180 to 180)
            start_date: First date (inclusive)
            end_date: Last date (inclusive)
            elevation: Site elevation [m]
            force_nasa: If True, skip ERA5 and use NASA POWER directly
        
        Returns:
            Tuple of (WeatherDataProvider, source_name)
            source_name is "ERA5-Land" or "NASA POWER"
        
        Raises:
            ValueError: Invalid coordinates or date range
            RuntimeError: Both ERA5 and NASA POWER failed
        """
        # Validate inputs
        if not (-90 <= latitude <= 90):
            raise ValueError(f"Latitude must be in [-90, 90], got {latitude}")
        if not (-180 <= longitude <= 180):
            raise ValueError(f"Longitude must be in [-180, 180], got {longitude}")
        if start_date >= end_date:
            raise ValueError(f"start_date must be < end_date")
        
        logger.info(
            f"Fetching weather for ({latitude:.4f}, {longitude:.4f}) "
            f"from {start_date} to {end_date}"
        )
        
        # Try ERA5-Land first (if available and not forced to use NASA)
        if not force_nasa and self.era5_source is not None:
            try:
                logger.info("Attempting to fetch from ERA5-Land...")
                provider = self.era5_source.fetch_weather(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=start_date,
                    end_date=end_date,
                    elevation=elevation
                )
                logger.info("✓ Successfully fetched weather from ERA5-Land")
                return provider, "ERA5-Land"
                
            except FileNotFoundError as e:
                logger.warning(f"ERA5 NetCDF files not found: {e}")
                logger.info("Falling back to NASA POWER...")
            except Exception as e:
                logger.error(f"ERA5 fetch failed: {e}")
                logger.info("Falling back to NASA POWER...")
        
        # Fallback to NASA POWER
        try:
            logger.info("Fetching from NASA POWER...")
            provider = self.nasa_service.get_weather_provider(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
                force_update=False
            )
            logger.info("✓ Successfully fetched weather from NASA POWER")
            return provider, "NASA POWER"
            
        except Exception as e:
            logger.error(f"NASA POWER fetch failed: {e}")
            raise RuntimeError(
                f"Failed to fetch weather from both ERA5-Land and NASA POWER: {e}"
            )
    
    def get_weather_provider(
        self,
        latitude: float,
        longitude: float,
        start_date: dt.date,
        end_date: dt.date,
        **kwargs
    ) -> WeatherDataProvider:
        """
        Get weather provider (compatible with existing code).
        
        This method signature matches WeatherService.get_weather_provider()
        for backward compatibility.
        
        Args:
            latitude: Site latitude
            longitude: Site longitude
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments (elevation, force_update, etc.)
        
        Returns:
            WeatherDataProvider
        """
        provider, source = self.get_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            elevation=kwargs.get('elevation', 10.0),
            force_nasa=kwargs.get('force_nasa', False)
        )
        return provider
    
    def is_era5_available(self) -> bool:
        """
        Check if ERA5-Land is currently available.
        
        Returns:
            True if ERA5 can be used
        """
        return self.era5_source is not None
    
    def get_source_info(self) -> dict:
        """
        Get information about available weather sources.
        
        Returns:
            Dict with source availability and status
        """
        info = {
            'era5_available': self.is_era5_available(),
            'nasa_available': True,  # NASA POWER always available
            'preferred_source': 'ERA5-Land' if self.is_era5_available() else 'NASA POWER',
        }
        
        if self.era5_source:
            info['era5_data_dir'] = str(self.era5_source.data_dir)
        
        return info


class HybridWeatherService:
    """
    Hybrid weather service with intelligent source selection.
    
    Automatically chooses the best available weather source:
    - ERA5-Land for high-resolution needs (crop modeling)
    - NASA POWER as reliable fallback
    
    This is the recommended service for production use.
    """
    
    def __init__(self):
        """Initialize hybrid weather service."""
        self.service = ERA5LandWeatherService(prefer_era5=True)
        logger.info(f"Hybrid weather service initialized")
        logger.info(f"Source info: {self.service.get_source_info()}")
    
    def get_weather_provider(
        self,
        latitude: float,
        longitude: float,
        start_date: dt.date,
        end_date: dt.date,
        **kwargs
    ) -> WeatherDataProvider:
        """
        Get weather provider from best available source.
        
        Args:
            latitude: Site latitude
            longitude: Site longitude
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments
        
        Returns:
            WeatherDataProvider
        """
        return self.service.get_weather_provider(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )


# Singleton instance for global use
_hybrid_service = None


def get_hybrid_weather_service() -> HybridWeatherService:
    """
    Get singleton hybrid weather service instance.
    
    Returns:
        HybridWeatherService
    """
    global _hybrid_service
    if _hybrid_service is None:
        _hybrid_service = HybridWeatherService()
    return _hybrid_service


def get_weather_for_simulation(
    latitude: float,
    longitude: float,
    start_date: dt.date,
    end_date: dt.date,
    elevation: float = 10.0
) -> tuple[WeatherDataProvider, str]:
    """
    Convenience function to get weather for simulation.
    
    Uses hybrid service with automatic ERA5/NASA fallback.
    
    Args:
        latitude: Site latitude
        longitude: Site longitude
        start_date: Start date
        end_date: End date
        elevation: Site elevation [m]
    
    Returns:
        Tuple of (WeatherDataProvider, source_name)
    """
    service = ERA5LandWeatherService()
    return service.get_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        elevation=elevation
    )


# Example usage
if __name__ == "__main__":
    """Test the ERA5 land service."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create service
    service = ERA5LandWeatherService()
    
    # Check availability
    info = service.get_source_info()
    print("\n" + "="*60)
    print("Weather Source Information:")
    print("="*60)
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Test fetch
    try:
        provider, source = service.get_weather(
            latitude=30.9,
            longitude=75.85,
            start_date=dt.date(2020, 1, 1),
            end_date=dt.date(2020, 1, 31),
            elevation=250.0
        )
        
        print(f"✓ Success!")
        print(f"  Source: {source}")
        print(f"  Days: {(provider.last_date - provider.first_date).days + 1}")
        print(f"  Date range: {provider.first_date} to {provider.last_date}")
        print(f"  Elevation: {provider.elevation}m")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
