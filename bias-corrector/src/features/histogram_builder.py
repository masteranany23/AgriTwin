"""
Sentinel-2 histogram feature extraction.
"""
import logging
from typing import Dict

import numpy as np


logger = logging.getLogger(__name__)


def build_histograms(
    sentinel2_tile: np.ndarray,
    bins: int = 32,
    value_range: tuple = (0.0, 1.0)
) -> np.ndarray:
    """
    Build histograms for multi-spectral Sentinel-2 tile.
    
    Args:
        sentinel2_tile: Multi-spectral image array (H, W, 12).
            Bands: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12.
        bins: Number of bins per histogram.
        value_range: Value range for histogram bins (min, max).
        
    Returns:
        Histogram features (bins, 12) - one histogram per band.
    """
    if sentinel2_tile.ndim != 3:
        raise ValueError(f"Expected 3D array (H, W, C), got shape {sentinel2_tile.shape}")
    
    height, width, n_bands = sentinel2_tile.shape
    
    if n_bands != 12:
        logger.warning(f"Expected 12 bands, got {n_bands}")
    
    histograms = np.zeros((bins, n_bands), dtype=np.float32)
    
    for band_idx in range(n_bands):
        band_data = sentinel2_tile[:, :, band_idx].flatten()
        
        # Remove invalid values (NaN, inf)
        band_data = band_data[np.isfinite(band_data)]
        
        if len(band_data) == 0:
            logger.warning(f"Band {band_idx} has no valid data")
            continue
        
        # Compute histogram
        hist, _ = np.histogram(
            band_data,
            bins=bins,
            range=value_range
        )
        
        # Normalize to probability distribution
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        
        histograms[:, band_idx] = hist
    
    return histograms


def process_tile_to_features(
    tile: np.ndarray,
    latitude: float,
    longitude: float,
    date: str,
    bins: int = 32
) -> Dict[str, any]:
    """
    Process Sentinel-2 tile to feature dictionary.
    
    Args:
        tile: Sentinel-2 tile (H, W, 12).
        latitude: Latitude of tile center.
        longitude: Longitude of tile center.
        date: Observation date (YYYY-MM-DD).
        bins: Number of histogram bins.
        
    Returns:
        Feature dictionary with histograms and metadata.
    """
    # Build histograms
    histograms = build_histograms(tile, bins=bins)
    
    # Flatten histograms to 1D feature vector
    hist_features = histograms.flatten()  # (bins * 12,)
    
    # Additional spectral indices (optional)
    # Extract mean per band as supplementary features
    band_means = np.nanmean(tile, axis=(0, 1))  # (12,)
    
    features = {
        "histogram_features": hist_features,
        "band_means": band_means,
        "latitude": latitude,
        "longitude": longitude,
        "date": date,
        "n_bins": bins,
        "n_bands": tile.shape[2] if tile.ndim == 3 else 0
    }
    
    return features


def extract_ndvi(tile: np.ndarray, nir_idx: int = 7, red_idx: int = 3) -> float:
    """
    Extract mean NDVI from Sentinel-2 tile.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        tile: Sentinel-2 tile (H, W, 12).
        nir_idx: Index of NIR band (default: 7 for B8).
        red_idx: Index of Red band (default: 3 for B4).
        
    Returns:
        Mean NDVI value.
    """
    nir = tile[:, :, nir_idx]
    red = tile[:, :, red_idx]
    
    # Compute NDVI
    ndvi = (nir - red) / (nir + red + 1e-6)
    
    # Remove invalid values
    ndvi = ndvi[np.isfinite(ndvi)]
    
    if len(ndvi) == 0:
        return 0.0
    
    return float(np.mean(ndvi))


def extract_ndre(
    tile: np.ndarray,
    nir_idx: int = 7,
    rededge_idx: int = 5
) -> float:
    """
    Extract mean NDRE (Normalized Difference Red Edge) from Sentinel-2 tile.
    
    NDRE = (NIR - RedEdge) / (NIR + RedEdge)
    
    Args:
        tile: Sentinel-2 tile (H, W, 12).
        nir_idx: Index of NIR band (default: 7 for B8).
        rededge_idx: Index of RedEdge band (default: 5 for B6).
        
    Returns:
        Mean NDRE value.
    """
    nir = tile[:, :, nir_idx]
    rededge = tile[:, :, rededge_idx]
    
    # Compute NDRE
    ndre = (nir - rededge) / (nir + rededge + 1e-6)
    
    # Remove invalid values
    ndre = ndre[np.isfinite(ndre)]
    
    if len(ndre) == 0:
        return 0.0
    
    return float(np.mean(ndre))
