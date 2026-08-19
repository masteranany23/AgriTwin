#!/usr/bin/env python
"""
AgriTwin ERA5-Land Reanalysis Downloader (Multi-Year Support).

Downloads 10 key biophysical meteorological and soil variables for India (0.1° resolution)
from Copernicus Climate Data Store (CDS).

Features:
- Multi-Year batch processing (e.g. 2018, 2019, 2020)
- 15-day chunking to strictly stay under the CDS 12,000 cost-unit limit per request
- Automatic license verification & acceptance
- Automatic xarray merging into clean monthly NetCDF files
- Safe resumption (skips already completed monthly files)
- Automatic cleanup of intermediate chunk parts

Usage:
    python scripts/download_era5_daily.py                 # Default: [2018, 2019]
    python scripts/download_era5_daily.py --years 2018 2019
"""

import argparse
import calendar
import os
import sys
from pathlib import Path
import cdsapi

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

# Ensure correct project root and data directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "era5_land"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Auto-locate and configure .cdsapirc from project root or user home
rc_candidate = PROJECT_ROOT / ".cdsapirc"
if rc_candidate.exists():
    os.environ["CDSAPI_RC"] = str(rc_candidate)

# India bounding box (North, West, South, East)
INDIA_BBOX = [35.5, 66.0, 6.5, 90.0]

# All 10 variables required for AgriTwin (WOFOST crop simulation & ML yield models)
VARIABLES = [
    "2m_temperature",                     # Temperature (min/max/mean)
    "2m_dewpoint_temperature",            # Dewpoint / Vapor pressure
    "total_precipitation",                # Daily rainfall
    "surface_solar_radiation_downwards",  # Solar radiation
    "10m_u_component_of_wind",            # Wind U component
    "10m_v_component_of_wind",            # Wind V component
    "volumetric_soil_water_layer_1",      # Soil moisture layer 1 (0-7cm)
    "volumetric_soil_water_layer_2",      # Soil moisture layer 2 (7-28cm)
    "volumetric_soil_water_layer_3",      # Soil moisture layer 3 (28-100cm)
    "volumetric_soil_water_layer_4",      # Soil moisture layer 4 (100-289cm)
]

ALL_HOURS = [f"{h:02d}:00" for h in range(24)]


def ensure_licences_accepted(client):
    """Auto-verify and accept required Copernicus dataset licenses if needed."""
    try:
        accepted = [lic.get("id") for lic in client.client.get_accepted_licences()]
        required_licences = [
            ("licence-to-use-copernicus-products", 12),
            ("cc-by", 1),
            ("terms-of-use-cds", 11),
        ]
        for lic_id, rev in required_licences:
            if lic_id not in accepted:
                print(f"📋 Accepting required licence '{lic_id}' (rev {rev})...")
                client.client.accept_licence(lic_id, rev)
    except Exception as e:
        print(f"⚠️ Licence check note: {e}")


def download_year(year: int, client: cdsapi.Client):
    """Download and merge all 12 monthly files for a given year."""
    print(f"\n" + "=" * 65)
    print(f"🌾 Processing Year {year}")
    print("=" * 65)
    
    total_months = list(range(1, 13))
    successful_months = []
    failed_months = []

    for month in total_months:
        _, last_day = calendar.monthrange(year, month)
        output_file = DATA_DIR / f"era5_land_{year}_{month:02d}.nc"

        if output_file.exists() and output_file.stat().st_size > 0:
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"✅ {year}-{month:02d} already exists ({output_file.name}, {size_mb:.2f} MB). Skipping.")
            successful_months.append(month)
            continue

        print(f"\n⏳ Processing {year}-{month:02d} (1 to {last_day} days)...")

        # Split into 2 chunks (1-15, 16-end) to strictly stay within CDS request cost limits (limit = 12,000)
        day_chunks = [
            [f"{d:02d}" for d in range(1, 16)],
            [f"{d:02d}" for d in range(16, last_day + 1)],
        ]

        chunk_files = []
        month_success = True

        for i, days in enumerate(day_chunks, start=1):
            chunk_file = DATA_DIR / f"era5_land_{year}_{month:02d}_part{i}.nc"
            chunk_files.append(chunk_file)

            if chunk_file.exists() and chunk_file.stat().st_size > 0:
                size_mb = chunk_file.stat().st_size / (1024 * 1024)
                print(f"  🔹 Part {i} (Days {days[0]}–{days[-1]}) already downloaded ({size_mb:.2f} MB).")
                continue

            print(f"  🔹 Requesting Part {i} (Days {days[0]}–{days[-1]})...")
            try:
                client.retrieve(
                    "reanalysis-era5-land",
                    {
                        "variable": VARIABLES,
                        "year": str(year),
                        "month": f"{month:02d}",
                        "day": days,
                        "time": ALL_HOURS,
                        "area": INDIA_BBOX,
                        "data_format": "netcdf",
                        "download_format": "unarchived",
                    },
                    str(chunk_file),
                )
                if chunk_file.exists() and chunk_file.stat().st_size > 0:
                    size_mb = chunk_file.stat().st_size / (1024 * 1024)
                    print(f"  ✅ Part {i} downloaded ({chunk_file.name}, {size_mb:.2f} MB)")
                else:
                    print(f"  ❌ Part {i} file was empty or missing.")
                    month_success = False
                    break
            except Exception as e:
                print(f"  ❌ Error downloading Part {i}: {e}")
                month_success = False
                break

        if not month_success:
            print(f"⚠️ Skipping merge for {year}-{month:02d} due to download error.")
            failed_months.append(month)
            continue

        # Merge parts into the final monthly NetCDF file
        if HAS_XARRAY:
            try:
                print(f"  🔄 Merging {len(chunk_files)} parts into {output_file.name}...")
                datasets = [xr.open_dataset(f) for f in chunk_files]
                time_dim = "valid_time" if "valid_time" in datasets[0].dims else "time"
                merged = xr.concat(datasets, dim=time_dim)
                merged.to_netcdf(output_file)
                
                # Close and clean up temporary parts
                for ds in datasets:
                    ds.close()
                for f in chunk_files:
                    f.unlink(missing_ok=True)

                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"  ✅ Successfully merged and created {output_file.name} ({size_mb:.2f} MB)")
                successful_months.append(month)
            except Exception as e:
                print(f"  ⚠️ Merge error ({e}). Parts retained separately.")
                failed_months.append(month)
        else:
            print(f"  ✅ Parts downloaded successfully: {[f.name for f in chunk_files]}")
            successful_months.append(month)

    print("\n" + "-" * 65)
    if len(successful_months) == len(total_months) and len(failed_months) == 0:
        print(f"🎉 Complete for Year {year}! All 12 months verified in {DATA_DIR}")
    else:
        print(f"⚠️ Year {year} Incomplete: {len(successful_months)}/12 months successful. Failed: {failed_months}")


def main():
    parser = argparse.ArgumentParser(description="Download ERA5-Land Reanalysis Data for India")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2018, 2019],
        help="Years to download (default: 2018 2019)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print(f"🌾 AgriTwin Multi-Year ERA5-Land Downloader")
    print(f"📅 Target Years: {args.years}")
    print(f"📁 Output Directory: {DATA_DIR}")
    print(f"🗺️  Bounding Box (India): {INDIA_BBOX}")
    print(f"📊 Variables ({len(VARIABLES)}): {', '.join(VARIABLES[:4])} ...")
    print("=" * 65)

    # Initialize CDS API client
    client = cdsapi.Client()
    ensure_licences_accepted(client)

    for year in args.years:
        download_year(year, client)

    print("\n" + "=" * 65)
    print("✨ All requested years processed!")
    print("=" * 65)


if __name__ == "__main__":
    main()
