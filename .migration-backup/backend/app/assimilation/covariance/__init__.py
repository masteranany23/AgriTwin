"""
backend/app/assimilation/covariance/__init__.py
=================================================

Observation error covariance abstractions for EnKF data assimilation.
"""

from backend.app.assimilation.covariance.observation_covariance import ObservationCovariance

__all__ = ["ObservationCovariance"]
