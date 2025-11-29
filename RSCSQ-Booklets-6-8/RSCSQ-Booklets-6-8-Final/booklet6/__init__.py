"""
RSCS-Q Booklet 6: Drift Surface Mapping & Entropy Governance
============================================================

This module provides drift detection, entropy governance, and
surface topology tracking for the RSCS-Q cognitive architecture.

Main Components:
    - EVIScore: Entropic Validity Index measurement
    - MDSScore: Model Drift Score measurement
    - EntropyGovernor: Aperture enforcement with graduated response
    - DriftSurface: Topology tracking across dimensions
    - B6Interface: Unified interface for B6 functionality

Example:
    >>> from src.booklet6 import B6Interface
    >>> interface = B6Interface("my-system")
    >>> evi = interface.compute_evi(predicted, actual)
    >>> print(f"EVI valid: {evi.is_valid()}")
"""

from .drift_surface import (
    # Constants
    EVI_THRESHOLD,
    MDS_WARNING,
    MDS_CRITICAL,
    ENTROPY_MIN,
    ENTROPY_MAX,
    
    # Enums
    DriftSeverity,
    GovernorAction,
    
    # Classes
    EVIScore,
    MDSScore,
    EntropyGovernor,
    DriftSurface,
    B6Interface,
    
    # Test runner
    run_tests,
)

__all__ = [
    'EVI_THRESHOLD',
    'MDS_WARNING', 
    'MDS_CRITICAL',
    'ENTROPY_MIN',
    'ENTROPY_MAX',
    'DriftSeverity',
    'GovernorAction',
    'EVIScore',
    'MDSScore',
    'EntropyGovernor',
    'DriftSurface',
    'B6Interface',
    'run_tests',
]

__version__ = '1.0.0'
