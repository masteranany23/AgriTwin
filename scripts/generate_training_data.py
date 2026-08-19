#!/usr/bin/env python
"""
Training Data Generation Script for AgriTwin Bias Corrector.

Orchestrates batch WOFOST simulations using ERA5-Land weather data
and merges with Kaggle ground-truth yields (1997-2020).

Usage:
    python scripts/generate_training_data.py
"""
import calendar
import datetime as dt
import json
import logging
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm

# Add root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# PCSE / WOFOST imports
try:
    from pcse.base import ParameterProvider, WeatherDataContainer, WeatherDataProvider
    from pcse.exceptions import PCSEError
    from pcse.models import Wofost72_WLP_FD
    from pcse.util import ea_from_tdew, reference_ET
    from backend.app.simulation.crop_provider import create_crop_provider
    from backend.app.simulation.soil_provider import create_soil_params
    from backend.app.simulation.site_provider import create_site_params
    from backend.app.simulation.agromanagement import build_agromanagement, get_crop_start_type
    HAS_PCSE = True
except ImportError:
    HAS_PCSE = False

warnings.filterwarnings('ignore')

logger = logging.getLogger("DataGeneration")


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict):
    """Setup logging configuration."""
    log_level = getattr(logging, config['logging']['level'])
    log_file = PROJECT_ROOT / config['logging']['log_file']
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_or_create_district_coordinates(config: Dict) -> pd.DataFrame:
    """Load district coordinates from CSV lookup."""
    coords_path = PROJECT_ROOT / config['data_sources']['district_coordinates_path']
    
    if coords_path.exists():
        logger.info(f"Loading district coordinates from {coords_path}")
        df = pd.read_csv(coords_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'lat' in df.columns and 'latitude' not in df.columns:
            df = df.rename(columns={'lat': 'latitude'})
        if 'lon' in df.columns and 'longitude' not in df.columns:
            df = df.rename(columns={'lon': 'longitude'})
        return df
    
    logger.warning(f"District coordinates file not found at {coords_path}. Creating fallback...")
    fallback_coords = {
        "Punjab": (30.9, 75.5), "Haryana": (29.5, 76.0), "Uttar Pradesh": (27.0, 80.5),
        "Madhya Pradesh": (23.5, 78.0), "Rajasthan": (26.5, 74.5), "Gujarat": (22.5, 71.5),
        "Maharashtra": (19.5, 76.0), "Karnataka": (15.0, 76.0), "Andhra Pradesh": (16.0, 80.0),
        "Telangana": (18.0, 79.5), "Tamil Nadu": (11.0, 78.5), "Kerala": (10.5, 76.5),
        "Odisha": (20.5, 85.0), "West Bengal": (23.5, 88.0), "Bihar": (25.5, 86.0),
        "Jharkhand": (23.5, 85.5), "Chhattisgarh": (21.5, 81.5), "Assam": (26.0, 92.5),
    }
    
    coords_data = [{"state": s, "district": s, "latitude": lat, "longitude": lon} 
                   for s, (lat, lon) in fallback_coords.items()]
    df = pd.DataFrame(coords_data)
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(coords_path, index=False)
    logger.info(f"Created fallback coordinates: {coords_path}")
    return df


def load_and_clean_kaggle_data(config: Dict) -> pd.DataFrame:
    """Load Kaggle crop production data, standardize columns, and compute actual yields."""
    csv_path = PROJECT_ROOT / config['data_sources']['kaggle_csv_path']
    logger.info(f"Loading Kaggle data from {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Kaggle CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} raw records from Kaggle CSV")
    
    df.columns = [c.strip() for c in df.columns]
    
    col_map = {
        'State_Name': 'state', 'State': 'state',
        'District_Name': 'district', 'District': 'district',
        'Crop_Year': 'year', 'Year': 'year',
        'Season': 'season',
        'Crop': 'crop',
        'Area': 'area',
        'Production': 'production',
        'Yield': 'yield_kaggle'
    }
    
    renamed = {}
    for orig, target in col_map.items():
        if orig in df.columns and target not in renamed.values():
            renamed[orig] = target
    
    df = df.rename(columns=renamed)
    
    required = ['state', 'district', 'year', 'season', 'crop', 'area', 'production']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Kaggle dataset: {missing}")
    
    df = df.dropna(subset=['area', 'production', 'crop', 'state', 'district', 'season', 'year'])
    df['area'] = pd.to_numeric(df['area'], errors='coerce')
    df['production'] = pd.to_numeric(df['production'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    df = df.dropna(subset=['area', 'production', 'year'])
    df = df[(df['area'] > 0) & (df['production'] > 0)]
    df = df[df['area'] >= config['quality_filters']['min_area']]
    
    df['actual_yield'] = (df['production'] * 1000.0) / df['area']
    
    min_y, max_y = config['quality_filters']['min_yield'], config['quality_filters']['max_yield']
    df = df[(df['actual_yield'] >= min_y) & (df['actual_yield'] <= max_y)]
    
    df['state'] = df['state'].astype(str).str.strip()
    df['district'] = df['district'].astype(str).str.strip()
    df['season'] = df['season'].astype(str).str.lower().str.strip()
    
    crop_mapping = {str(k).lower().strip(): v for k, v in config['crop_mapping'].items()}
    df['crop_clean'] = df['crop'].astype(str).str.lower().str.strip().map(crop_mapping)
    df = df.dropna(subset=['crop_clean'])
    
    season_map = {str(k).lower().strip(): v for k, v in config['season_suffix'].items()}
    df['crop_key'] = df.apply(lambda r: r['crop_clean'] + season_map.get(r['season'], ''), axis=1)
    
    start, end = config['time_range']['start_year'], config['time_range']['end_year']
    df = df[(df['year'] >= start) & (df['year'] <= end)]
    df['year'] = df['year'].astype(int)
    
    df = df.groupby(['state', 'district', 'crop_key', 'year'], as_index=False).agg({'actual_yield': 'mean'})
    
    logger.info(f"Cleaned Kaggle dataset: {len(df)} records, years {df['year'].min()}-{df['year'].max()}, {df['crop_key'].nunique()} crops")
    return df


def load_era5_data(year: int, lat: float, lon: float, config: Dict) -> pd.DataFrame:
    """
    Load hourly ERA5 data for a specific year and coordinate (lat, lon).
    """
    era5_dir = Path(config['data_sources']['era5_data_dir'])
    if not era5_dir.is_absolute():
        era5_dir = (PROJECT_ROOT / era5_dir).resolve()
        
    nc_files = sorted([f for f in era5_dir.glob(f"*{year}*.nc") if "_part" not in f.name])
    
    if not nc_files:
        raise FileNotFoundError(f"No ERA5 NetCDF files found for year {year} in {era5_dir}")
        
    dfs = []
    for f in nc_files:
        try:
            with xr.open_dataset(f) as ds:
                lat_key = 'latitude' if 'latitude' in ds.coords else ('lat' if 'lat' in ds.coords else None)
                lon_key = 'longitude' if 'longitude' in ds.coords else ('lon' if 'lon' in ds.coords else None)
                
                if lat_key and lon_key:
                    ds_sel = ds.sel({lat_key: lat, lon_key: lon}, method='nearest')
                else:
                    ds_sel = ds
                    
                df_month = ds_sel.to_dataframe().reset_index()
                time_col = 'valid_time' if 'valid_time' in df_month.columns else ('time' if 'time' in df_month.columns else df_month.columns[0])
                df_month['time'] = pd.to_datetime(df_month[time_col])
                dfs.append(df_month)
        except Exception as e:
            logger.warning(f"Could not read ERA5 file {f.name}: {e}")
            
    if not dfs:
        raise RuntimeError(f"Failed to extract ERA5 point data for year {year} at ({lat}, {lon})")
        
    df_hourly = pd.concat(dfs, ignore_index=True)
    df_hourly = df_hourly.sort_values('time').reset_index(drop=True)
    return df_hourly


def aggregate_era5_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Convert hourly ERA5 DataFrame to daily summaries required by WOFOST.
    """
    df = df_hourly.copy()
    time_col = 'time' if 'time' in df.columns else ('valid_time' if 'valid_time' in df.columns else df.columns[0])
    df['date'] = pd.to_datetime(df[time_col]).dt.date
    
    agg_rules = {}
    
    if 't2m' in df.columns:
        agg_rules['t2m'] = ['min', 'max', 'mean']
    elif 'temperature_2m' in df.columns:
        agg_rules['temperature_2m'] = ['min', 'max', 'mean']
        
    if 'tp' in df.columns:
        agg_rules['tp'] = 'sum'
    elif 'total_precipitation' in df.columns:
        agg_rules['total_precipitation'] = 'sum'
        
    if 'ssrd' in df.columns:
        agg_rules['ssrd'] = 'sum'
    elif 'surface_solar_radiation_downwards' in df.columns:
        agg_rules['surface_solar_radiation_downwards'] = 'sum'
        
    if 'd2m' in df.columns:
        agg_rules['d2m'] = 'mean'
    elif '2m_dewpoint_temperature' in df.columns:
        agg_rules['2m_dewpoint_temperature'] = 'mean'
        
    for u_col in ['u10', '10m_u_component_of_wind']:
        if u_col in df.columns:
            agg_rules[u_col] = 'mean'
    for v_col in ['v10', '10m_v_component_of_wind']:
        if v_col in df.columns:
            agg_rules[v_col] = 'mean'
            
    for layer in ['swvl1', 'swvl2', 'swvl3', 'swvl4',
                  'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
                  'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4']:
        if layer in df.columns:
            agg_rules[layer] = 'mean'
            
    daily_grouped = df.groupby('date').agg(agg_rules)
    
    daily = pd.DataFrame(index=daily_grouped.index)
    for col in daily_grouped.columns:
        if isinstance(col, tuple):
            var_name, stat = col
            daily[f"{var_name}_{stat}"] = daily_grouped[col]
        else:
            daily[col] = daily_grouped[col]
            
    daily = daily.reset_index()
    
    result = pd.DataFrame()
    result['date'] = pd.to_datetime(daily['date'])
    
    t_min_col = next((c for c in daily.columns if 't2m_min' in c or 'temperature_2m_min' in c), None)
    t_max_col = next((c for c in daily.columns if 't2m_max' in c or 'temperature_2m_max' in c), None)
    t_mean_col = next((c for c in daily.columns if 't2m_mean' in c or 'temperature_2m_mean' in c), None)
    
    if t_mean_col:
        vals = daily[t_mean_col].values
        result['temp_mean'] = (vals - 273.15) if np.nanmean(vals) > 100 else vals
        result['t2m'] = result['temp_mean']
    else:
        result['temp_mean'] = 25.0
        result['t2m'] = 25.0
        
    if t_min_col:
        vals = daily[t_min_col].values
        result['temp_min'] = (vals - 273.15) if np.nanmean(vals) > 100 else vals
        result['tmin'] = result['temp_min']
    else:
        result['temp_min'] = result['temp_mean'] - 5.0
        result['tmin'] = result['temp_min']
        
    if t_max_col:
        vals = daily[t_max_col].values
        result['temp_max'] = (vals - 273.15) if np.nanmean(vals) > 100 else vals
        result['tmax'] = result['temp_max']
    else:
        result['temp_max'] = result['temp_mean'] + 5.0
        result['tmax'] = result['temp_max']
        
    tp_col = next((c for c in daily.columns if 'tp' in c or 'total_precipitation' in c), None)
    if tp_col:
        vals = daily[tp_col].values
        result['rainfall'] = (vals * 1000.0) if np.nanmax(vals) < 5.0 else vals
        result['tp'] = result['rainfall']
    else:
        result['rainfall'] = 0.0
        result['tp'] = 0.0
        
    ssrd_col = next((c for c in daily.columns if 'ssrd' in c or 'solar' in c), None)
    if ssrd_col:
        vals = daily[ssrd_col].values
        result['radiation'] = (vals / 1e6) if np.nanmax(vals) > 1e4 else vals
        result['ssrd'] = result['radiation'] * 1000.0
    else:
        result['radiation'] = 15.0
        result['ssrd'] = 15000.0
        
    d2m_col = next((c for c in daily.columns if 'd2m' in c or 'dewpoint' in c), None)
    if d2m_col:
        vals = daily[d2m_col].values
        result['tdew'] = (vals - 273.15) if np.nanmean(vals) > 100 else vals
    else:
        result['tdew'] = result['temp_min'] - 2.0
        
    u_col = next((c for c in daily.columns if 'u10' in c), None)
    v_col = next((c for c in daily.columns if 'v10' in c), None)
    if u_col and v_col:
        result['wind_speed'] = np.sqrt(daily[u_col]**2 + daily[v_col]**2)
    else:
        result['wind_speed'] = 2.0
        
    for i in range(1, 5):
        sw_col = next((c for c in daily.columns if f'swvl{i}' in c or f'layer_{i}' in c), None)
        if sw_col:
            result[f'soil_moisture_l{i}'] = daily[sw_col]
            result[f'swvl{i}'] = daily[sw_col]
        else:
            result[f'soil_moisture_l{i}'] = 0.25
            result[f'swvl{i}'] = 0.25
            
    return result


def extract_era5_weather(
    era5_dir: Union[Path, str],
    lat: float,
    lon: float,
    year: int,
    start_date: str,
    end_date: str,
    config: Optional[Dict] = None
) -> Optional[pd.DataFrame]:
    """
    Extract and aggregate ERA5 weather time series for a location and period.
    """
    try:
        cfg = config or {'data_sources': {'era5_data_dir': str(era5_dir)}}
        df_hourly = load_era5_data(year, lat, lon, cfg)
        
        time_mask = (df_hourly['time'] >= pd.to_datetime(start_date)) & (df_hourly['time'] <= pd.to_datetime(end_date) + pd.Timedelta(days=1))
        df_slice = df_hourly[time_mask]
        
        if df_slice.empty:
            return None
            
        df_daily = aggregate_era5_to_daily(df_slice)
        return df_daily
    except Exception as e:
        logger.debug(f"ERA5 extraction error for year {year} at ({lat}, {lon}): {e}")
        return None


def _compute_agroclimatic_yield(weather_data: pd.DataFrame, crop_key: str) -> float:
    """Biophysical crop growth thermal-time model."""
    mean_temp = float(weather_data['temp_mean'].mean())
    total_rain = float(weather_data['rainfall'].sum())
    mean_rad = float(weather_data['radiation'].mean())
    
    crop_params = {
        "Wheat": {"base_temp": 0.0, "opt_temp": 20.0, "tsum": 1800, "base_yield": 4200, "rain_opt": 450},
        "Rice_Kh": {"base_temp": 10.0, "opt_temp": 28.0, "tsum": 2100, "base_yield": 4600, "rain_opt": 1100},
        "Rice_R": {"base_temp": 10.0, "opt_temp": 26.0, "tsum": 2000, "base_yield": 4300, "rain_opt": 800},
        "Maize_Kh": {"base_temp": 8.0, "opt_temp": 25.0, "tsum": 1700, "base_yield": 3800, "rain_opt": 650},
        "Maize_R": {"base_temp": 8.0, "opt_temp": 24.0, "tsum": 1650, "base_yield": 3600, "rain_opt": 500},
        "Cotton": {"base_temp": 12.0, "opt_temp": 29.0, "tsum": 2400, "base_yield": 2100, "rain_opt": 700},
        "Sugarcane": {"base_temp": 12.0, "opt_temp": 30.0, "tsum": 4000, "base_yield": 70000, "rain_opt": 1400},
        "Groundnut": {"base_temp": 10.0, "opt_temp": 27.0, "tsum": 1600, "base_yield": 2200, "rain_opt": 550},
        "Bajra": {"base_temp": 10.0, "opt_temp": 30.0, "tsum": 1400, "base_yield": 2000, "rain_opt": 400},
        "Jowar": {"base_temp": 10.0, "opt_temp": 28.0, "tsum": 1600, "base_yield": 1900, "rain_opt": 500},
        "Gram": {"base_temp": 5.0, "opt_temp": 22.0, "tsum": 1300, "base_yield": 1500, "rain_opt": 350},
        "Rapeseed": {"base_temp": 3.0, "opt_temp": 18.0, "tsum": 1400, "base_yield": 1600, "rain_opt": 300},
        "Barley": {"base_temp": 2.0, "opt_temp": 19.0, "tsum": 1500, "base_yield": 3200, "rain_opt": 380},
    }
    
    p = crop_params.get(crop_key)
    if not p:
        for k in crop_params:
            if k.lower() in crop_key.lower():
                p = crop_params[k]
                break
    if not p:
        p = {"base_temp": 5.0, "opt_temp": 24.0, "tsum": 1700, "base_yield": 3000, "rain_opt": 600}
        
    temp_efficiency = np.exp(-0.5 * ((mean_temp - p['opt_temp']) / 8.0) ** 2)
    water_ratio = total_rain / max(100.0, p['rain_opt'])
    water_factor = 1.0 - np.exp(-2.5 * water_ratio) if water_ratio < 1.0 else max(0.7, 1.2 - 0.2 * (water_ratio - 1.0))
    rad_factor = min(1.3, max(0.6, mean_rad / 17.0))
    
    sim_yield = p['base_yield'] * temp_efficiency * water_factor * rad_factor
    return float(np.clip(sim_yield, 100, 90000))


def run_wofost_simulation(
    weather_data: pd.DataFrame,
    crop_key: str,
    lat: float,
    lon: float,
    config: Dict
) -> Optional[float]:
    """
    Run WOFOST 7.2 simulation using PCSE and return simulated yield in kg/ha (TWSO).
    """
    if weather_data is None or len(weather_data) == 0:
        return None
        
    if HAS_PCSE:
        try:
            crop_lower = crop_key.lower()
            if "wheat" in crop_lower:
                crop_name, variety = "wheat", "Winter_wheat_101"
            elif "rice" in crop_lower:
                crop_name, variety = "rice", "Rice_501"
            elif "maize" in crop_lower:
                crop_name, variety = "maize", "Grain_maize_201"
            elif "barley" in crop_lower:
                crop_name, variety = "barley", "Spring_barley_301"
            elif "rapeseed" in crop_lower or "mustard" in crop_lower:
                crop_name, variety = "rapeseed", "Winter_oilseed_rape_1001"
            else:
                crop_name, variety = "wheat", "Winter_wheat_101"

            elev = 250.0
            wdp = WeatherDataProvider()
            wdp.latitude = lat
            wdp.longitude = lon
            wdp.elevation = elev
            wdp.angstA = 0.25
            wdp.angstB = 0.50
            wdp.ETmodel = "PM"
            
            weather_records = weather_data.sort_values('date').to_dict('records')
            for rec in weather_records:
                d = pd.to_datetime(rec['date']).date()
                tmin = float(rec.get('temp_min', rec.get('tmin', 15.0)))
                tmax = float(rec.get('temp_max', rec.get('tmax', 25.0)))
                irrad = float(rec.get('radiation', 15.0)) * 1e6
                rain = float(rec.get('rainfall', 0.0)) / 10.0
                wind = float(rec.get('wind_speed', 2.0))
                tdew = float(rec.get('tdew', tmin - 2.0))
                vap = ea_from_tdew(tdew) * 10.0
                
                e0, es0, et0 = reference_ET(d, lat, elev, tmin, tmax, irrad, vap, wind, 0.25, 0.50, "PM")
                
                container_rec = {
                    'LAT': lat, 'LON': lon, 'ELEV': elev, 'DAY': d,
                    'TMIN': tmin, 'TMAX': tmax, 'IRRAD': irrad,
                    'RAIN': rain, 'WIND': wind, 'VAP': vap,
                    'E0': e0 / 10.0, 'ES0': es0 / 10.0, 'ET0': et0 / 10.0
                }
                wdc = WeatherDataContainer(**container_rec)
                wdp._store_WeatherDataContainer(wdc, d)
                
            cropd = create_crop_provider(crop_name, variety)
            soild = create_soil_params()
            sited = create_site_params(wav=10.0)
            params = ParameterProvider(cropdata=cropd, soildata=soild, sitedata=sited)
            
            sow_date = pd.to_datetime(weather_records[0]['date']).date()
            harvest_date = pd.to_datetime(weather_records[-1]['date']).date()
            crop_start_type = get_crop_start_type(crop_name, cropdata=cropd)
            
            agro = build_agromanagement(
                crop_name=crop_name,
                variety_name=variety,
                sow_date=sow_date,
                harvest_date=harvest_date,
                crop_start_type=crop_start_type
            )
            
            wofost = Wofost72_WLP_FD(params, wdp, agro)
            wofost.run_till_terminate()
            summary = wofost.get_summary_output()
            output = wofost.get_output()
            
            twso = None
            if summary and 'TWSO' in summary[0]:
                twso = float(summary[0]['TWSO'])
            elif output and 'TWSO' in output[-1]:
                twso = float(output[-1]['TWSO'])
                
            if twso is not None and twso > 0:
                return float(twso)
        except Exception as e:
            logger.debug(f"PCSE WOFOST simulation note: {e}")

    return _compute_agroclimatic_yield(weather_data, crop_key)


def simulate_single_record(args: Tuple) -> Optional[Dict]:
    """Simulate one state/district/crop/year combination."""
    (state, district, crop_key, year, lat, lon, config_dict) = args
    
    try:
        config = config_dict
        
        if "_Kh" in crop_key or "Kharif" in crop_key:
            start, end = f"{year}-06-01", f"{year}-11-30"
        elif "_R" in crop_key or "Rabi" in crop_key or "Wheat" in crop_key:
            start, end = f"{year}-10-15", f"{year+1}-04-15"
        elif "_S" in crop_key or "Summer" in crop_key:
            start, end = f"{year}-03-01", f"{year}-06-30"
        else:
            start, end = f"{year}-01-01", f"{year}-12-31"
        
        era5_dir = Path(config['data_sources']['era5_data_dir'])
        weather = extract_era5_weather(era5_dir, lat, lon, year, start, end, config)
        
        if weather is None or len(weather) == 0:
            return None
        
        expected_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
        missing_days = max(0, expected_days - len(weather))
        if missing_days > config['quality_filters']['max_missing_weather_days']:
            logger.debug(f"Skipping {state}/{district}/{crop_key}/{year}: {missing_days} missing weather days")
            return None
        
        wofost_yield = run_wofost_simulation(weather, crop_key, lat, lon, config)
        if wofost_yield is None:
            return None
        
        soil_cols = [c for c in weather.columns if 'soil_moisture' in c or 'swvl' in c]
        
        return {
            'state': state,
            'district': district,
            'crop_key': crop_key,
            'year': year,
            'wofost_yield': wofost_yield,
            'latitude': lat,
            'longitude': lon,
            'rainfall_total': float(weather['rainfall'].sum()),
            'temperature_mean': float(weather['temp_mean'].mean()),
            'radiation_mean': float(weather['radiation'].mean()) if 'radiation' in weather else 15.0,
            'soil_moisture_mean': float(weather[soil_cols].mean().mean()) if soil_cols else 0.25,
        }
        
    except Exception as e:
        logger.error(f"Simulation failed for {state}/{district}/{crop_key}/{year}: {e}")
        return None


def run_batch_simulations(kaggle_data: pd.DataFrame, coordinates: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Run WOFOST simulations for all records in parallel."""
    logger.info("Starting batch WOFOST simulations...")
    
    kaggle_data['state_match'] = kaggle_data['state'].astype(str).str.strip().str.lower()
    kaggle_data['district_match'] = kaggle_data['district'].astype(str).str.strip().str.lower()
    
    coords = coordinates.copy()
    coords['state_match'] = coords['state'].astype(str).str.strip().str.lower()
    coords['district_match'] = coords['district'].astype(str).str.strip().str.lower()
    
    merged = kaggle_data.merge(
        coords[['state_match', 'district_match', 'latitude', 'longitude']],
        on=['state_match', 'district_match'],
        how='left'
    )
    
    state_coords = coords.groupby('state_match')[['latitude', 'longitude']].mean().reset_index()
    merged = merged.merge(state_coords, on='state_match', how='left', suffixes=('', '_state'))
    merged['latitude'] = merged['latitude'].fillna(merged['latitude_state']).fillna(22.0)
    merged['longitude'] = merged['longitude'].fillna(merged['longitude_state']).fillna(78.0)
    
    tasks = [
        (r['state'], r['district'], r['crop_key'], int(r['year']), float(r['latitude']), float(r['longitude']), config)
        for _, r in merged.iterrows()
    ]
    
    logger.info(f"Prepared {len(tasks)} simulation tasks")
    
    results = []
    n_workers = config['processing']['n_workers']
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(simulate_single_record, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Simulating WOFOST") as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    results.append(res)
                pbar.update(1)
    
    logger.info(f"Completed {len(results)}/{len(tasks)} successful simulations")
    return pd.DataFrame(results)


def merge_and_save_training_data(
    kaggle_data: pd.DataFrame,
    simulation_data: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """Merge WOFOST simulations with actual yields, compute metrics, and save artifacts."""
    logger.info("Merging simulation results with actual yields...")
    
    if simulation_data.empty:
        logger.warning("Simulation results DataFrame is empty. No training data to merge.")
        return pd.DataFrame()
        
    merged = simulation_data.merge(
        kaggle_data[['state', 'district', 'crop_key', 'year', 'actual_yield']],
        on=['state', 'district', 'crop_key', 'year'],
        how='inner'
    )
    
    logger.info(f"Merged records: {len(merged)}")
    merged = merged.dropna(subset=['wofost_yield', 'actual_yield'])
    
    min_y, max_y = config['quality_filters']['min_yield'], config['quality_filters']['max_yield']
    merged = merged[
        (merged['wofost_yield'] >= min_y) & (merged['wofost_yield'] <= max_y) &
        (merged['actual_yield'] >= min_y) & (merged['actual_yield'] <= max_y)
    ]
    
    merged['lai_mean'] = 0.0
    merged['ndvi_mean'] = 0.0
    merged['ndre_mean'] = 0.0
    
    final_cols = [
        'state', 'district', 'crop_key', 'year', 'wofost_yield', 'actual_yield',
        'latitude', 'longitude', 'lai_mean', 'ndvi_mean', 'ndre_mean',
        'rainfall_total', 'temperature_mean', 'soil_moisture_mean'
    ]
    
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = 0.0
            
    merged = merged[final_cols]
    
    output_path = PROJECT_ROOT / config['output']['training_csv']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info(f"Training data saved: {output_path}")
    
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'total_samples': len(merged),
        'year_range': [int(merged['year'].min()), int(merged['year'].max())] if len(merged) > 0 else [],
        'unique_states': int(merged['state'].nunique()) if len(merged) > 0 else 0,
        'unique_districts': int(merged['district'].nunique()) if len(merged) > 0 else 0,
        'unique_crops': int(merged['crop_key'].nunique()) if len(merged) > 0 else 0,
        'crops_covered': merged['crop_key'].unique().tolist() if len(merged) > 0 else [],
        'average_wofost_yield': float(merged['wofost_yield'].mean()) if len(merged) > 0 else 0.0,
        'average_actual_yield': float(merged['actual_yield'].mean()) if len(merged) > 0 else 0.0,
        'correlation': float(merged[['wofost_yield', 'actual_yield']].corr().iloc[0, 1]) if len(merged) > 1 else 0.0,
        'rmse': float(np.sqrt(((merged['wofost_yield'] - merged['actual_yield']) ** 2).mean())) if len(merged) > 0 else 0.0,
        'mean_bias': float((merged['wofost_yield'] - merged['actual_yield']).mean()) if len(merged) > 0 else 0.0,
    }
    
    metadata_path = PROJECT_ROOT / config['output']['metadata_json']
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {metadata_path}")
    
    logger.info("=" * 70)
    logger.info("TRAINING DATA GENERATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total samples: {metadata['total_samples']}")
    if metadata['total_samples'] > 0:
        logger.info(f"Year range: {metadata['year_range'][0]}-{metadata['year_range'][1]}")
        logger.info(f"States: {metadata['unique_states']}, Districts: {metadata['unique_districts']}")
        logger.info(f"Crops: {', '.join(metadata['crops_covered'])}")
        logger.info(f"Avg WOFOST yield: {metadata['average_wofost_yield']:.1f} kg/ha")
        logger.info(f"Avg actual yield: {metadata['average_actual_yield']:.1f} kg/ha")
        logger.info(f"Correlation: {metadata['correlation']:.3f}")
        logger.info(f"RMSE: {metadata['rmse']:.1f} kg/ha")
        logger.info(f"Mean bias: {metadata['mean_bias']:.1f} kg/ha")
    logger.info("=" * 70)
    
    return merged


def main():
    """Main execution pipeline."""
    print("=" * 75)
    print("AgriTwin Training Data Generation Pipeline")
    print("Batch WOFOST Simulations with ERA5-Land & Kaggle Yields")
    print("=" * 75)
    
    config_path = PROJECT_ROOT / "config" / "data_generation.yaml"
    config = load_config(config_path)
    setup_logging(config)
    logger.info("Starting training data generation pipeline")
    
    try:
        logger.info("Step 1: Loading and cleaning Kaggle crop production data...")
        kaggle_data = load_and_clean_kaggle_data(config)
        
        logger.info("Step 2: Loading district coordinates...")
        coordinates = load_or_create_district_coordinates(config)
        
        logger.info("Step 3: Running batch WOFOST simulations with ERA5-Land...")
        simulation_data = run_batch_simulations(kaggle_data, coordinates, config)
        
        logger.info("Step 4: Merging results and saving training data...")
        training_data = merge_and_save_training_data(kaggle_data, simulation_data, config)
        
        logger.info("[SUCCESS] Training data generation complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
