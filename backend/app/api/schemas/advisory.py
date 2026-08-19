"""
api/schemas/advisory.py — Farmer Advisory & Decision Support Schemas
=====================================================================

Pydantic schemas for:
  1. Crop Recommendation & Profit Engine (Farmer Need 1)
  2. Daily Field Advisory & Actionable Alerts (Farmer Needs 4, 5, 7)
  3. Farmer Summary Dashboard / WhatsApp Card (Farmer Needs 3, 6, UI)
"""

import datetime
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Crop Recommendation ────────────────────────────────────────────────────────

class CropRecommendationRequest(BaseModel):
    """Request schema for crop recommendation."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Field latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Field longitude")
    season: str = Field("rabi", description="Target season: 'kharif', 'rabi', or 'zaid'")
    land_area_acres: float = Field(2.5, gt=0, description="Farm land size in acres (default 2.5 acres)")
    field_id: Optional[uuid.UUID] = Field(None, description="Optional registered field UUID")


class CropOption(BaseModel):
    """Economic and agronomic profile of a recommended crop."""
    crop_name: str = Field(..., description="Crop common name (e.g., 'Wheat', 'Mustard', 'Rice')")
    variety_name: str = Field(..., description="Recommended variety (e.g., 'HD-2967', 'Pusa Bold')")
    season: str = Field(..., description="Season: 'kharif', 'rabi', 'zaid'")
    suitability_score: float = Field(..., ge=0.0, le=1.0, description="Soil & climate suitability (0-1)")
    suitability_label: str = Field(..., description="'Highly Suitable', 'Suitable', 'Moderate'")
    
    # Yield projections
    expected_yield_kg_ha: float = Field(..., description="Expected yield in kg/hectare")
    expected_yield_quintal_acre: float = Field(..., description="Expected yield in Quintals/acre")
    
    # Financials (₹ in INR)
    msp_inr_per_quintal: float = Field(..., description="Government Minimum Support Price (₹/Quintal)")
    estimated_cost_per_acre_inr: float = Field(..., description="Estimated cost of cultivation (seed, fert, water) ₹/acre")
    gross_revenue_per_acre_inr: float = Field(..., description="Expected gross revenue ₹/acre")
    net_profit_per_acre_inr: float = Field(..., description="Expected net profit ₹/acre")
    total_net_profit_inr: float = Field(..., description="Total projected profit for the farmer's land area")
    
    # Agronomic guidance
    optimal_sowing_window: str = Field(..., description="Recommended sowing date range")
    key_advantage_en: str = Field(..., description="Key reason in English")
    key_advantage_hi: str = Field(..., description="Key reason in Hindi")


class CropRecommendationResponse(BaseModel):
    """Response containing ranked crop recommendations."""
    latitude: float
    longitude: float
    season: str
    land_area_acres: float
    top_recommendation: CropOption
    ranked_options: List[CropOption]
    summary_message_en: str
    summary_message_hi: str


# ── Daily Field Advisory & Alerts ──────────────────────────────────────────────

class AdvisoryItem(BaseModel):
    """A single actionable advice or alert for the farmer."""
    category: str = Field(..., description="'irrigation', 'nitrogen', 'pest_stress', 'weather', 'general'")
    severity: str = Field("info", description="'info', 'warning', 'critical'")
    icon: str = Field("💡", description="Emoji icon (💧, 🚨, 🌧️, 🌾, ☀️)")
    title_en: str = Field(..., description="Title in English")
    title_hi: str = Field(..., description="Title in Hindi")
    action_en: str = Field(..., description="Concrete action recommendation in English")
    action_hi: str = Field(..., description="Concrete action recommendation in Hindi")
    timing: str = Field("immediate", description="'immediate', 'within_3_days', 'routine'")
    confidence_score: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in advisory")


class FieldAdvisoryResponse(BaseModel):
    """Daily advisory summary for a field."""
    field_id: uuid.UUID
    date: datetime.date
    crop: Optional[str] = None
    variety: Optional[str] = None
    growth_stage: Optional[str] = None
    dvs: Optional[float] = None
    soil_moisture_status: str = Field(..., description="'Optimal', 'Low', 'Critical Deficit', 'Excess'")
    nitrogen_status: str = Field(..., description="'Adequate', 'Mild Stress', 'Severe Chlorosis'")
    advisories: List[AdvisoryItem]
    weather_forecast_summary_en: str
    weather_forecast_summary_hi: str


# ── Farmer Summary Dashboard ───────────────────────────────────────────────────

class FarmerSummaryResponse(BaseModel):
    """Comprehensive summary card for mobile dashboard / WhatsApp bot."""
    field_id: uuid.UUID
    field_name: str
    crop: str
    variety: str
    sowing_date: Optional[datetime.date] = None
    days_after_sowing: int
    current_stage: str
    
    # Expected yield with uncertainty (Farmer Needs 3 & 6)
    expected_yield_kg_ha: float
    expected_yield_quintal_acre: float
    confidence_interval_kg_ha: float = Field(..., description="± error margin in kg/ha")
    confidence_percentage: float = Field(..., ge=0, le=100, description="Model confidence %")
    
    # Historical comparison
    historical_comparison_text_en: str
    historical_comparison_text_hi: str
    yield_change_vs_last_year_pct: float
    
    # Key active advisories
    active_alerts: List[AdvisoryItem]
    
    # Formatted display cards (ready for WhatsApp / SMS / UI display)
    card_text_en: str
    card_text_hi: str
