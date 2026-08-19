"""
Spatial Alignment Service
=========================

Resamples observations from different resolutions (10m, 9km, point) to a unified field grid.

Research Alignment:
- Enables meaningful fusion of Sentinel-2 (10m), ERA5-Land (9km), and Farmer Photos (point).
- Ensures the EnKF receives data that is spatially representative of the entire field.
- Critical for smallholder farms (2.5 acres) where 9km ERA5 pixels cover thousands of fields.
"""

import logging

from typing import Optional, Tuple
from uuid import UUID
from shapely.geometry import Point, Polygon, box



from backend.app.models.field import Field
from backend.app.api.schemas.fusion import (
    SpatialAlignmentRequest,
    SpatialAlignmentResponse,
    SpatialResolution
)

logger = logging.getLogger(__name__)


class SpatialAlignmentService:
    """
    Aligns multi-resolution observations to a common field grid.
    """
    
    def __init__(self, db_session):
        self.db = db_session
        # Define approximate cell sizes in degrees for different resolutions
        # (1 degree latitude ≈ 111 km)
        self.resolution_to_deg = {
            "HIGH": 0.00009,    # ~10m
            "MEDIUM": 0.001,    # ~100m
            "LOW": 0.08,        # ~9km
            "POINT": 0.0        # Exact point
        }
    
    def _get_field_boundary(self, field_id: UUID) -> Optional[Polygon]:
        """Fetch or approximate field boundary."""
        field = self.db.query(Field).filter(Field.id == field_id).first()
        if not field:
            return None
        
        # If field has a GeoJSON boundary, use it
        if field.boundary_geojson:
            from shapely.geometry import shape
            return shape(field.boundary_geojson)
        
        # Fallback: Create a small square buffer around the field center
        # For Indian smallholder farms (~2.5 acres ≈ 100m x 100m)
        center_lat = field.latitude
        center_lon = field.longitude
        # Approx 0.0009 degrees ≈ 100 meters
        delta = 0.00045
        return box(
            center_lon - delta, 
            center_lat - delta,
            center_lon + delta, 
            center_lat + delta
        )
    
    def _resample_point_to_grid(self, point: Point, grid_resolution: float) -> Tuple[float, float]:
        """
        Snap a GPS point to the nearest grid cell center.
        This ensures the farmer photo is assigned to a specific grid cell.
        """
        # Simple snapping: round to nearest grid resolution
        # In production, you'd use a proper rasterization library like rasterio
        grid_x = round(point.x / grid_resolution) * grid_resolution
        grid_y = round(point.y / grid_resolution) * grid_resolution
        return (grid_x, grid_y)
    
    def _aggregate_low_res_to_field(self, value: float, source_resolution: float, field_area: float) -> float:
        """
        Downsamples a coarse pixel (e.g., 9km ERA5) to field level.
        If the field is small (2.5 acres), we simply take the pixel's value
        as it represents the broader regional weather.
        """
        # For ERA5-Land (9km), the field is just a tiny fraction of the pixel.
        # We accept the pixel's average value for the field.
        # A more sophisticated approach would use bilinear interpolation.
        return value
    
    def align_observations(self, request: SpatialAlignmentRequest) -> SpatialAlignmentResponse:
        """
        Align all observations to a unified grid.
        """
        field_boundary = self._get_field_boundary(request.field_id)
        if not field_boundary:
            return SpatialAlignmentResponse(
                field_id=request.field_id,
                aligned_observations=[],
                grid_metadata={"error": "Field boundary not found"},
                message="Failed to align: Field not found."
            )
        
        target_res = request.target_resolution
        grid_metadata = {
            "target_resolution_meters": target_res,
            "field_boundary": field_boundary.bounds,
            "field_area_sq_m": field_boundary.area
        }
        
        aligned_obs = []
        
        for obs in request.observations:
            # Parse observation metadata
            lat = obs.get("latitude")
            lon = obs.get("longitude")
            resolution = obs.get("resolution", "MEDIUM")
            value = obs.get("value")
            variable = obs.get("variable", "LAI")
            source = obs.get("source", "UNKNOWN")
            
            # Create a shapely point for the observation
            obs_point = Point(lon, lat)
            
            # Check if observation falls within field boundary (or is close)
            if not field_boundary.contains(obs_point) and obs_point.distance(field_boundary) > 0.001:
                # If the observation is outside the field, skip or warn
                # For ERA5 (9km), it might be slightly outside but still representative
                if resolution != "LOW":
                    logger.warning(f"Observation {source} at ({lat},{lon}) is outside field boundary. Skipping.")
                    continue
            
            # Resample based on resolution
            if resolution == "POINT":
                # Farmer photo: snap to nearest grid cell
                grid_lon, grid_lat = self._resample_point_to_grid(obs_point, target_res)
                obs["snapped_latitude"] = grid_lat
                obs["snapped_longitude"] = grid_lon
                obs["grid_cell"] = f"cell_{grid_lat}_{grid_lon}"
            elif resolution == "LOW":
                # ERA5-Land: Keep as-is (field is too small to resolve)
                obs["snapped_latitude"] = lat
                obs["snapped_longitude"] = lon
                obs["field_representative"] = True
            else:
                # MEDIUM/HIGH (Sentinel): Keep as-is or resample to target grid
                grid_lon, grid_lat = self._resample_point_to_grid(obs_point, target_res)
                obs["snapped_latitude"] = grid_lat
                obs["snapped_longitude"] = grid_lon
                obs["grid_cell"] = f"cell_{grid_lat}_{grid_lon}"
            
            aligned_obs.append(obs)
        
        return SpatialAlignmentResponse(
            field_id=request.field_id,
            aligned_observations=aligned_obs,
            grid_metadata=grid_metadata,
            message=f"Aligned {len(aligned_obs)} observations to {target_res}m grid."
        )
