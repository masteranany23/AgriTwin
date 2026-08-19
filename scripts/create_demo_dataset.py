#!/usr/bin/env python
"""
Quick Demo Dataset Generator for AgriTwin Bias Corrector.

Creates a small training dataset from Kaggle crop_production.csv
with synthetic WOFOST yields for demo purposes.

Usage:
    python scripts/create_demo_dataset.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent

def create_demo_dataset():
    """Create demo training dataset from Kaggle data."""
    print("=" * 80)
    print("Creating Demo Dataset for AgriTwin Bias Corrector")
    print("=" * 80)
    
    # Load Kaggle data
    kaggle_path = PROJECT_ROOT / "backend" / "data" / "kaggle" / "crop_production.csv"
    print(f"\n1. Loading Kaggle data from: {kaggle_path}")
    
    if not kaggle_path.exists():
        print(f"ERROR: Kaggle CSV not found at {kaggle_path}")
        return
    
    df = pd.read_csv(kaggle_path)
    print(f"   Loaded {len(df)} records")
    
    # Clean and filter
    print("\n2. Cleaning and filtering data...")
    df = df.dropna(subset=['Area', 'Production'])
    df = df[(df['Area'] > 0) & (df['Production'] > 0)]
    df = df[df['Area'] >= 10]  # Min 10 hectares
    
    # Compute actual yield (tonnes to kg/ha)
    df['actual_yield'] = (df['Production'] * 1000) / df['Area']
    df = df[(df['actual_yield'] >= 100) & (df['actual_yield'] <= 15000)]
    
    # Standardize column names
    df = df.rename(columns={
        'State_Name': 'state',
        'District_Name': 'district',
        'Crop_Year': 'year',
        'Season': 'season',
        'Crop': 'crop'
    })
    
    # Filter to common crops
    crop_mapping = {
        'rice': 'Rice',
        'wheat': 'Wheat',
        'maize': 'Maize',
        'cotton': 'Cotton',
        'sugarcane': 'Sugarcane',
        'groundnut': 'Groundnut'
    }
    df['crop'] = df['crop'].str.lower().str.strip()
    df = df[df['crop'].isin(crop_mapping.keys())]
    df['crop'] = df['crop'].map(crop_mapping)
    
    # Filter to recent years with data
    df = df[(df['year'] >= 2010) & (df['year'] <= 2020)]
    
    # Add season suffix
    season_map = {
        'kharif': '_Kh',
        'rabi': '_R',
        'whole year': '',
        'annual': ''
    }
    df['season'] = df['season'].str.lower().str.strip()
    df['crop_key'] = df.apply(
        lambda r: r['crop'] + season_map.get(r['season'], ''),
        axis=1
    )
    
    # Aggregate duplicates
    df = df.groupby(['state', 'district', 'crop_key', 'year'], as_index=False).agg({
        'actual_yield': 'mean'
    })
    
    print(f"   After cleaning: {len(df)} records")
    print(f"   Years: {df['year'].min()}-{df['year'].max()}")
    print(f"   Crops: {df['crop_key'].unique().tolist()}")
    
    # Sample for demo (limit to manageable size)
    print("\n3. Sampling records for demo dataset...")
    np.random.seed(42)
    
    # Sample 200 records across different states/crops
    if len(df) > 200:
        df = df.sample(n=200, random_state=42)
    
    print(f"   Demo sample size: {len(df)} records")
    
    # Add mock coordinates (approximate state centers)
    state_coords = {
        "Andhra Pradesh": (15.9129, 79.7400),
        "Assam": (26.2006, 92.9376),
        "Bihar": (25.0961, 85.3131),
        "Chhattisgarh": (21.2787, 81.8661),
        "Gujarat": (22.2587, 71.1924),
        "Haryana": (29.0588, 76.0856),
        "Karnataka": (15.3173, 75.7139),
        "Kerala": (10.8505, 76.2711),
        "Madhya Pradesh": (22.9734, 78.6569),
        "Maharashtra": (19.7515, 75.7139),
        "Odisha": (20.9517, 85.0985),
        "Punjab": (31.1471, 75.3412),
        "Rajasthan": (27.0238, 74.2179),
        "Tamil Nadu": (11.1271, 78.6569),
        "Telangana": (18.1124, 79.0193),
        "Uttar Pradesh": (26.8467, 80.9462),
        "West Bengal": (22.9868, 87.8550),
    }
    
    df['latitude'] = df['state'].map(lambda s: state_coords.get(s, (20.0, 78.0))[0])
    df['longitude'] = df['state'].map(lambda s: state_coords.get(s, (20.0, 78.0))[1])
    
    # Add small random jitter to coordinates for district variation
    df['latitude'] = df['latitude'] + np.random.uniform(-1, 1, size=len(df))
    df['longitude'] = df['longitude'] + np.random.uniform(-1, 1, size=len(df))
    
    # Generate synthetic WOFOST yields
    print("\n4. Generating synthetic WOFOST yields...")
    
    # Base yields by crop (kg/ha)
    base_yields = {
        'Wheat': 4000,
        'Rice': 4500,
        'Rice_Kh': 4200,
        'Rice_R': 4300,
        'Maize': 3500,
        'Maize_Kh': 3600,
        'Cotton': 2000,
        'Cotton_Kh': 2100,
        'Sugarcane': 70000,
        'Groundnut': 1800,
        'Groundnut_Kh': 1900
    }
    
    def generate_wofost_yield(row):
        """Generate synthetic WOFOST yield with realistic bias."""
        actual = row['actual_yield']
        crop = row['crop_key']
        base = base_yields.get(crop, 3000)
        
        # Add systematic bias (WOFOST tends to overestimate)
        bias_factor = np.random.uniform(1.05, 1.25)
        
        # Add noise
        noise = np.random.normal(0, actual * 0.1)
        
        # Generate yield close to actual but with bias
        wofost = actual * bias_factor + noise
        
        # Ensure reasonable range
        wofost = max(actual * 0.8, min(wofost, actual * 1.5))
        
        return wofost
    
    df['wofost_yield'] = df.apply(generate_wofost_yield, axis=1)
    
    # Add mock satellite features
    print("\n5. Adding mock satellite and weather features...")
    
    # LAI: Leaf Area Index (0-8, typical range 1-5 for crops)
    df['lai_mean'] = np.random.uniform(1.5, 5.0, size=len(df))
    
    # NDVI: Normalized Difference Vegetation Index (0-1, healthy crops ~0.6-0.9)
    df['ndvi_mean'] = np.random.uniform(0.6, 0.85, size=len(df))
    
    # NDRE: Normalized Difference Red Edge (0-1, correlates with LAI)
    df['ndre_mean'] = np.random.uniform(0.2, 0.4, size=len(df))
    
    # Rainfall: Total seasonal rainfall (mm, typical 400-1200 for India)
    df['rainfall_total'] = np.random.uniform(400, 1200, size=len(df))
    
    # Temperature: Mean temperature (°C, typical 20-30 for India)
    df['temperature_mean'] = np.random.uniform(20, 30, size=len(df))
    
    # Soil moisture: Mean soil moisture (m³/m³, typical 0.15-0.35)
    df['soil_moisture_mean'] = np.random.uniform(0.15, 0.35, size=len(df))
    
    # Reorder columns
    final_cols = [
        'state', 'district', 'crop_key', 'year',
        'wofost_yield', 'actual_yield',
        'latitude', 'longitude',
        'lai_mean', 'ndvi_mean', 'ndre_mean',
        'rainfall_total', 'temperature_mean', 'soil_moisture_mean'
    ]
    
    df = df[final_cols]
    
    # Save CSV
    output_path = PROJECT_ROOT / "bias-corrector" / "data" / "training" / "train.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n6. Training data saved: {output_path}")
    
    # Generate metadata
    metadata = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'data_source': 'Kaggle crop_production.csv with synthetic WOFOST yields',
        'is_demo': True,
        'total_samples': len(df),
        'year_range': [int(df['year'].min()), int(df['year'].max())],
        'unique_states': int(df['state'].nunique()),
        'unique_districts': int(df['district'].nunique()),
        'unique_crops': int(df['crop_key'].nunique()),
        'crops_covered': sorted(df['crop_key'].unique().tolist()),
        'average_wofost_yield': float(df['wofost_yield'].mean()),
        'average_actual_yield': float(df['actual_yield'].mean()),
        'correlation': float(df[['wofost_yield', 'actual_yield']].corr().iloc[0, 1]),
        'rmse': float(np.sqrt(((df['wofost_yield'] - df['actual_yield']) ** 2).mean())),
        'mean_bias': float((df['wofost_yield'] - df['actual_yield']).mean()),
        'mean_bias_percent': float(((df['wofost_yield'] - df['actual_yield']) / df['actual_yield']).mean() * 100),
    }
    
    metadata_path = PROJECT_ROOT / "bias-corrector" / "data" / "training" / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved: {metadata_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("DEMO DATASET SUMMARY")
    print("=" * 80)
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Year range: {metadata['year_range'][0]}-{metadata['year_range'][1]}")
    print(f"States: {metadata['unique_states']}, Districts: {metadata['unique_districts']}")
    print(f"Crops: {', '.join(metadata['crops_covered'])}")
    print(f"\nAvg WOFOST yield: {metadata['average_wofost_yield']:.1f} kg/ha")
    print(f"Avg actual yield: {metadata['average_actual_yield']:.1f} kg/ha")
    print(f"Mean bias: {metadata['mean_bias']:.1f} kg/ha ({metadata['mean_bias_percent']:.1f}%)")
    print(f"\nCorrelation: {metadata['correlation']:.3f}")
    print(f"RMSE: {metadata['rmse']:.1f} kg/ha")
    print("=" * 80)
    print("\n✓ Demo dataset ready for training!")
    print(f"   Next: python bias-corrector/run.py train")
    

if __name__ == "__main__":
    create_demo_dataset()
