"""
RSCS-Q Booklet 7: Reflective Autonomy & Swarm Intelligence
==========================================================

This module provides reflective swarm coherence, meta-kernel
bridge integration, and activation profile management.

Main Components:
    - SwarmCoherence: Hash-based consensus protocol
    - SwarmMember: Individual swarm participant
    - ActivationProfile: Task-appropriate autonomy levels
    - MetaKernelBridge: B6→B7→B8 integration layer
    - ReflexiveOverride: Bounded self-modification

Example:
    >>> from src.booklet7 import MetaKernelBridge, ActivationLevel
    >>> bridge = MetaKernelBridge("my-system")
    >>> bridge.receive_b6_metrics(evi=0.7, mds=0.2)
    >>> print(f"Activation: {bridge.current_profile.level.name}")
"""

from .meta_kernel_bridge import (
    # Constants
    QUORUM_THRESHOLD,
    FORK_TIMEOUT,
    COHERENCE_MIN,
    
    # Enums
    ActivationLevel,
    SwarmState,
    
    # Classes
    SwarmMember,
    SwarmCoherence,
    ActivationProfile,
    MetaKernelBridge,
    OverrideRequest,
    ReflexiveOverride,
    
    # Test runner
    run_tests,
)

__all__ = [
    'QUORUM_THRESHOLD',
    'FORK_TIMEOUT',
    'COHERENCE_MIN',
    'ActivationLevel',
    'SwarmState',
    'SwarmMember',
    'SwarmCoherence',
    'ActivationProfile',
    'MetaKernelBridge',
    'OverrideRequest',
    'ReflexiveOverride',
    'run_tests',
]

__version__ = '1.0.0'
