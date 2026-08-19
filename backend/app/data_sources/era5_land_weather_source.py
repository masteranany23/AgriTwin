"""
ERA5-Land Weather Source for AgriTwin
======================================

Implements WeatherSource interface using ERA5-Land reanalysis data.
ERA5-Land provides high-resolution (0.1° ~11km) weather and soil data.

Features:
- 4-layer soil moisture (0-7cm, 7-28cm, 28-100cm, 100-289cm)
- Hourly data aggregated to daily
- Higher spatial resolution than NASA POWER
- Requires pre-downloaded NetCDF files

Data source: Copernicus Climate Data Store (CDS)
https://cds.climate.copernicus.eu/
"""


import logging
import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    xr = None

from pcse.base import WeatherDataProvider, WeatherDataContainer
from pcse.util import ea_from_tdew, reference_ET
from pcse.exceptions import PCSEError

from .weather_source import WeatherSource

logger = logging.getLogger(__name__)

# Default Angstrom coefficients for ERA5
DEFAULT_ANGSTROM_A = 0.25
DEFAULT_ANGSTROM_B = 0.50


class ERA5LandWeatherSource(WeatherSource):
    """
    Weather source using ERA5-Land reanalysis NetCDF files.
    
    Usage:
        source = ERA5LandWeatherSource(data_dir="path/to/era5/netcdf/files")
        provider = source.get_weather(lat=30.9, lon=75.85,
                                      start_date=date(2020, 1, 1),
                                      end_date=date(2020, 12, 31))
    
    NetCDF file naming convention:
        era5_land_{year}.nc or era5_land_combined.nc
    
    Required variables in NetCDF:
        - t2m: 2m temperature [K]
        - tp: Total precipitation [m]
        - ssrd: Surface solar radiation downwards [J/m²]
        - u10, v10: 10m wind components [m/s]
        - d2m: 2m dewpoint temperature [K]
        - swvl1, swvl2, swvl3, swvl4: Soil moisture [m³/m³]
    """
    
    def __init__(self, data_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize ERA5-Land weather source.
        
        Args:
            data_dir: Directory containing ERA5-Land NetCDF files
            cache_dir: Optional cache directory (not used currently)
        """
        if not HAS_XARRAY:
            raise ImportError(
                "xarray is required for ERA5LandWeatherSource. "
                "Install with: pip install xarray netCDF4"
            )
        
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"ERA5 data directory not found: {data_dir}")
        
        self.cache_dir = cache_dir
        logger.info(f"Initialized ERA5LandWeatherSource with data_dir={data_dir}")
    
    def get_source_name(self) -> str:
        """Return source name for logging."""
        return "ERA5-Land"
    
    def get_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: dt.date,
        end_date: dt.date,
        *,
        elevation: float = 10.0,
    ) -> WeatherDataProvider:
        """
        Fetch ERA5-Land weather data for location and time range.
        
        Args:
            latitude: Site latitude (-90 to 90)
            longitude: Site longitude (-180 to 180)
            start_date: First date (inclusive)
            end_date: Last date (inclusive)
            elevation: Site elevation [m] (used for ET calculation)
        
        Returns:
            WeatherDataProvider ready for WOFOST
        
        Raises:
            FileNotFoundError: If ERA5 NetCDF file not found
            ValueError: If invalid coordinates or date range
            RuntimeError: If data extraction fails
        """
        # Validate inputs
        if not (-90 <= latitude <= 90):
            raise ValueError(f"Latitude must be in [-90, 90], got {latitude}")
        if not (-180 <= longitude <= 180):
            raise ValueError(f"Longitude must be in [-180, 180], got {longitude}")
        if start_date >= end_date:
            raise ValueError(f"start_date must be < end_date")
        
        logger.info(
            f"Fetching ERA5-Land weather for ({latitude:.4f}, {longitude:.4f}) "
            f"from {start_date} to {end_date}"
        )
        
        # Extract data from NetCDF
        df_era5 = self._extract_era5_data(latitude, longitude, start_date, end_date)
        
        # Convert to PCSE format
        df_pcse = self._convert_to_pcse_format(df_era5, latitude, longitude, elevation)
        
        # Estimate Angstrom coefficients
        angst_a, angst_b = self._estimate_angstrom(df_pcse)
        
        # Build WeatherDataProvider
        provider = self._build_provider(
            df_pcse, latitude, longitude, elevation, angst_a, angst_b
        )
        
        logger.info(
            f"ERA5-Land weather loaded: {provider.first_date} to {provider.last_date} "
            f"({(provider.last_date - provider.first_date).days + 1} days)"
        )
        
        return provider
    
    def _find_netcdf_file(self, year: int) -> Optional[Path]:
        """
        Find NetCDF file for given year.
        
        Looks for:
        1. era5_land_{year}.nc
        2. era5_land_combined.nc (multi-year file)
        
        Args:
            year: Year to find data for
        
        Returns:
            Path to NetCDF file or None if not found
        """
        # Try year-specific file first
        year_file = self.data_dir / f"era5_land_{year}.nc"
        if year_file.exists():
            return year_file
        
        # Try combined file
        combined_file = self.data_dir / "era5_land_combined.nc"
        if combined_file.exists():
            return combined_file
        
        return None
    
    def _extract_era5_data(
        self,
        latitude: float,
        longitude: float,
        start_date: dt.date,
        end_date: dt.date,
    ) -> pd.DataFrame:
        """
        Extract ERA5-Land data from NetCDF files.
        
        Args:
            latitude: Site latitude
            longitude: Site longitude
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with ERA5 variables
        
        Raises:
            FileNotFoundError: If NetCDF file not found
            RuntimeError: If data extraction fails
        """
        # Determine which years we need
        years = range(start_date.year, end_date.year + 1)
        
        # Collect data from all needed files
        all_data = []
        
        for year in years:
            nc_file = self._find_netcdf_file(year)
            
            if nc_file is None:
                raise FileNotFoundError(
                    f"ERA5-Land NetCDF file not found for year {year}. "
                    f"Expected: {self.data_dir}/era5_land_{year}.nc or "
                    f"{self.data_dir}/era5_land_combined.nc"
                )
            
            logger.debug(f"Reading ERA5 data from {nc_file}")
            
            try:
                # Open dataset
                ds = xr.open_dataset(nc_file)
                
                # Select nearest grid point
                ds_point = ds.sel(
                    latitude=latitude,
                    longitude=longitude,
                    method='nearest'
                )
                
                # Filter to date range
                ds_point = ds_point.sel(
                    time=slice(
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                )
                
                # Convert to DataFrame
                df = ds_point.to_dataframe().reset_index()
                all_data.append(df)
                
                ds.close()
                
            except Exception as e:
                raise RuntimeError(f"Failed to extract data from {nc_file}: {e}")
        
        # Combine data from all years
        if not all_data:
            raise RuntimeError("No data extracted from ERA5 files")
        
        df_combined = pd.concat(all_data, ignore_index=True)
        
        # Aggregate hourly to daily
        df_daily = self._aggregate_to_daily(df_combined)
        
        return df_daily
    
    def _aggregate_to_daily(self, df_hourly: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate hourly ERA5 data to daily values.
        
        Aggregation rules:
        - Temperature: min, max, mean
        - Precipitation: sum
        - Radiation: sum
        - Wind: mean of components, then compute speed
        - Dewpoint: mean
        - Soil moisture: mean
        
        Args:
            df_hourly: Hourly ERA5 data
        
        Returns:
            Daily aggregated DataFrame
        """
        df_hourly['date'] = pd.to_datetime(df_hourly['time']).dt.date
        
        # Aggregate by date
        daily_agg = df_hourly.groupby('date').agg({
            't2m': ['min', 'max', 'mean'],  # Temperature
            'tp': 'sum',                     # Precipitation
            'ssrd': 'sum',                   # Radiation
            'u10': 'mean',                   # Wind U component
            'v10': 'mean',                   # Wind V component
            'd2m': 'mean',                   # Dewpoint
            'swvl1': 'mean',                 # Soil moisture layer 1
            'swvl2': 'mean',                 # Soil moisture layer 2
            'swvl3': 'mean',                 # Soil moisture layer 3
            'swvl4': 'mean',                 # Soil moisture layer 4
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = [
            'date', 
            't2m_min', 't2m_max', 't2m_mean',
            'tp_sum',
            'ssrd_sum',
            'u10_mean', 'v10_mean',
            'd2m_mean',
            'swvl1_mean', 'swvl2_mean', 'swvl3_mean', 'swvl4_mean'
        ]
        
        # Compute wind speed from components
        daily_agg['wind_speed'] = np.sqrt(
            daily_agg['u10_mean']**2 + daily_agg['v10_mean']**2
        )
        
        return daily_agg
    
    def _convert_to_pcse_format(
        self,
        df_era5: pd.DataFrame,
        latitude: float,
        longitude: float,
        elevation: float,
    ) -> pd.DataFrame:
        """
        Convert ERA5 DataFrame to PCSE-compatible format.
        
        Unit conversions:
        - Temperature: K → °C (subtract 273.15)
        - Precipitation: m → cm (multiply by 100)
        - Radiation: J/m² → J/m²/day (already daily sum)
        - Wind: m/s → m/s (no conversion)
        - Dewpoint: K → °C (subtract 273.15)
        - Soil moisture: m³/m³ → m³/m³ (no conversion)
        
        Args:
            df_era5: ERA5 daily data
            latitude: Site latitude
            longitude: Site longitude
            elevation: Site elevation
        
        Returns:
            PCSE-compatible DataFrame
        """
        df_pcse = pd.DataFrame({
            # Date
            'DAY': pd.to_datetime(df_era5['date']),
            
            # Temperature: K → °C
            'TMIN': df_era5['t2m_min'] - 273.15,
            'TMAX': df_era5['t2m_max'] - 273.15,
            'TEMP': df_era5['t2m_mean'] - 273.15,
            
            # Radiation: J/m²/day (already correct unit)
            'IRRAD': df_era5['ssrd_sum'],
            
            # Precipitation: m → cm
            'RAIN': df_era5['tp_sum'] * 100,
            
            # Wind: m/s (already correct)
            'WIND': df_era5['wind_speed'],
            
            # Dewpoint: K → °C, then convert to vapor pressure
            'TDEW': df_era5['d2m_mean'] - 273.15,
            
            # Soil moisture: m³/m³ (already correct)
            'SOIL_MOISTURE_L1': df_era5['swvl1_mean'],
            'SOIL_MOISTURE_L2': df_era5['swvl2_mean'],
            'SOIL_MOISTURE_L3': df_era5['swvl3_mean'],
            'SOIL_MOISTURE_L4': df_era5['swvl4_mean'],
            
            # Site info
            'LAT': latitude,
            'LON': longitude,
            'ELEV': elevation,
        })
        
        # Convert dewpoint to vapor pressure (hPa)
        df_pcse['VAP'] = df_pcse['TDEW'].apply(
            lambda tdew: ea_from_tdew(tdew) * 10.0  # kPa → hPa
        )
        
        # Convert date to Python date objects
        df_pcse['DAY'] = df_pcse['DAY'].dt.date
        
        # Drop rows with missing data
        n_before = len(df_pcse)
        df_pcse = df_pcse.dropna()
        n_dropped = n_before - len(df_pcse)
        
        if n_dropped > 0:
            logger.warning(f"Dropped {n_dropped} days with missing ERA5 data")
        
        return df_pcse
    
    def _estimate_angstrom(self, df_pcse: pd.DataFrame) -> tuple:
        """
        Estimate Angstrom A/B coefficients.
        
        For ERA5, we use conservative default values since we don't have
        top-of-atmosphere radiation to estimate from.
        
        Args:
            df_pcse: PCSE-format DataFrame
        
        Returns:
            (angst_a, angst_b) tuple
        """
        # Use default coefficients for ERA5
        # These are typical values for mid-latitudes
        return DEFAULT_ANGSTROM_A, DEFAULT_ANGSTROM_B
    
    def _build_provider(
        self,
        df_pcse: pd.DataFrame,
        latitude: float,
        longitude: float,
        elevation: float,
        angst_a: float,
        angst_b: float,
    ) -> WeatherDataProvider:
        """
        Build WeatherDataProvider from processed DataFrame.
        
        Args:
            df_pcse: PCSE-format DataFrame
            latitude: Site latitude
            longitude: Site longitude
            elevation: Site elevation
            angst_a: Angstrom A coefficient
            angst_b: Angstrom B coefficient
        
        Returns:
            WeatherDataProvider ready for WOFOST
        """
        provider = WeatherDataProvider()
        provider.latitude = latitude
        provider.longitude = longitude
        provider.elevation = elevation
        provider.angstA = angst_a
        provider.angstB = angst_b
        provider.ETmodel = "PM"  # Penman-Monteith
        provider.description = [
            f"ERA5-Land weather for ({latitude:.4f}, {longitude:.4f})"
        ]
        
        records = df_pcse.to_dict(orient='records')
        n_errors = 0
        
        for rec in records:
            # Compute reference evapotranspiration
            try:
                e0_mm, es0_mm, et0_mm = reference_ET(
                    rec['DAY'], rec['LAT'], rec['ELEV'],
                    rec['TMIN'], rec['TMAX'], rec['IRRAD'],
                    rec['VAP'], rec['WIND'],
                    angst_a, angst_b, "PM"
                )
            except (ValueError, PCSEError) as e:
                logger.warning(f"ET calculation failed for {rec['DAY']}: {e}")
                n_errors += 1
                continue
            
            # Convert ET from mm/day → cm/day
            rec['E0'] = e0_mm / 10.0
            rec['ES0'] = es0_mm / 10.0
            rec['ET0'] = et0_mm / 10.0
            
            # Create and store weather container
            wdc = WeatherDataContainer(**rec)
            provider._store_WeatherDataContainer(wdc, wdc.DAY)
        
        if n_errors > 0:
            logger.warning(f"Skipped {n_errors} days due to ET calculation errors")
        
        return provider
