"""
RSCS-Q Booklet 7: Reflective Autonomy & Swarm Intelligence
==========================================================

This module provides reflective swarm coherence, meta-kernel
bridge integration, and activation profile management.

Components:
- Swarm Coherence: Hash-based agreement protocol
- Meta-Kernel Bridge: B6→B7→B8 integration layer
- Activation Profiles: Task-appropriate autonomy levels
- Reflexive Override: Bounded self-modification

Author: Entropica Research Collective
Version: 1.0

Note: This module bridges B6 (Drift/Entropy) with B8 (Self-Modeling).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum, auto
import hashlib
import json


# =============================================================================
# CONSTANTS
# =============================================================================

QUORUM_THRESHOLD = 0.67   # Minimum quorum for consensus
FORK_TIMEOUT = 3          # Ticks to resolve fork
COHERENCE_MIN = 0.6       # Minimum acceptable coherence


# =============================================================================
# ENUMS
# =============================================================================

class ActivationLevel(Enum):
    """Autonomy activation levels"""
    DORMANT = 0      # Minimal activity, human approval required
    GUARDED = 1      # Conservative, frequent checkpoints
    ACTIVE = 2       # Normal operation, standard oversight
    AUTONOMOUS = 3   # Full autonomy within bounds


class SwarmState(Enum):
    """Swarm consensus state"""
    COHERENT = auto()
    DIVERGING = auto()
    FORKED = auto()
    RECOVERING = auto()


# =============================================================================
# SWARM COHERENCE
# =============================================================================

@dataclass
class SwarmMember:
    """Individual swarm member"""
    member_id: str
    state_hash: str = ""
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    reputation: float = 1.0
    
    def update_hash(self, state: Dict[str, Any]) -> str:
        """Update state hash"""
        canonical = json.dumps(state, sort_keys=True)
        self.state_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        self.last_heartbeat = datetime.utcnow()
        return self.state_hash


class SwarmCoherence:
    """
    Manages hash-based swarm agreement.
    
    Coherence κ = (1/N) Σ 1[hash_i = mode(hashes)]
    """
    
    def __init__(self, swarm_id: str):
        self.swarm_id = swarm_id
        self.members: Dict[str, SwarmMember] = {}
        self.state = SwarmState.COHERENT
        self.fork_tick = 0
    
    def add_member(self, member: SwarmMember) -> None:
        """Add member to swarm"""
        self.members[member.member_id] = member
    
    def compute_coherence(self) -> float:
        """Compute current coherence score"""
        if not self.members:
            return 1.0
        
        hashes = [m.state_hash for m in self.members.values() if m.state_hash]
        if not hashes:
            return 1.0
        
        # Find mode hash
        from collections import Counter
        hash_counts = Counter(hashes)
        mode_hash, mode_count = hash_counts.most_common(1)[0]
        
        coherence = mode_count / len(hashes)
        
        # Update state
        if coherence >= QUORUM_THRESHOLD:
            self.state = SwarmState.COHERENT
            self.fork_tick = 0
        elif coherence >= COHERENCE_MIN:
            self.state = SwarmState.DIVERGING
        else:
            self.state = SwarmState.FORKED
            self.fork_tick += 1
        
        return coherence
    
    def get_consensus_hash(self) -> Optional[str]:
        """Get consensus hash if quorum achieved"""
        if not self.members:
            return None
        
        hashes = [m.state_hash for m in self.members.values() if m.state_hash]
        if not hashes:
            return None
        
        from collections import Counter
        hash_counts = Counter(hashes)
        mode_hash, mode_count = hash_counts.most_common(1)[0]
        
        if mode_count / len(hashes) >= QUORUM_THRESHOLD:
            return mode_hash
        return None
    
    def needs_recovery(self) -> bool:
        """Check if fork recovery needed"""
        return self.fork_tick >= FORK_TIMEOUT
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'swarm_id': self.swarm_id,
            'members': len(self.members),
            'coherence': self.compute_coherence(),
            'state': self.state.name,
            'fork_tick': self.fork_tick,
            'consensus_hash': self.get_consensus_hash()
        }


# =============================================================================
# ACTIVATION PROFILES
# =============================================================================

@dataclass
class ActivationProfile:
    """Defines autonomy behavior at each level"""
    level: ActivationLevel
    checkpoint_interval: int        # Ticks between checkpoints
    approval_required: Set[str]     # Action types requiring approval
    entropy_bounds: tuple           # (min, max) entropy aperture
    drift_tolerance: float          # Max MDS before escalation
    
    @classmethod
    def dormant(cls) -> 'ActivationProfile':
        return cls(
            level=ActivationLevel.DORMANT,
            checkpoint_interval=1,
            approval_required={'*'},  # All actions
            entropy_bounds=(0.1, 0.3),
            drift_tolerance=0.1
        )
    
    @classmethod
    def guarded(cls) -> 'ActivationProfile':
        return cls(
            level=ActivationLevel.GUARDED,
            checkpoint_interval=5,
            approval_required={'mutation', 'spawn', 'unbind'},
            entropy_bounds=(0.2, 0.5),
            drift_tolerance=0.25
        )
    
    @classmethod
    def active(cls) -> 'ActivationProfile':
        return cls(
            level=ActivationLevel.ACTIVE,
            checkpoint_interval=10,
            approval_required={'unbind'},
            entropy_bounds=(0.3, 0.7),
            drift_tolerance=0.4
        )
    
    @classmethod
    def autonomous(cls) -> 'ActivationProfile':
        return cls(
            level=ActivationLevel.AUTONOMOUS,
            checkpoint_interval=20,
            approval_required=set(),  # No approval needed
            entropy_bounds=(0.4, 0.9),
            drift_tolerance=0.5
        )


# =============================================================================
# META-KERNEL BRIDGE
# =============================================================================

class MetaKernelBridge:
    """
    Integration layer between B6, B7, and B8.
    
    Inherits EVI/MDS from B6, exports activation profiles,
    and feeds into B8 self-modeling.
    """
    
    def __init__(self, bridge_id: str):
        self.bridge_id = bridge_id
        self.current_profile = ActivationProfile.guarded()
        self.evi_cache: Optional[float] = None
        self.mds_cache: Optional[float] = None
        self.swarm = SwarmCoherence(f"{bridge_id}-SWARM")
        self.history: List[Dict[str, Any]] = []
    
    def receive_b6_metrics(self, evi: float, mds: float) -> None:
        """Receive metrics from B6"""
        self.evi_cache = evi
        self.mds_cache = mds
        
        # Auto-adjust activation based on metrics
        self._adjust_activation()
    
    def _adjust_activation(self) -> None:
        """Adjust activation level based on metrics"""
        if self.evi_cache is None or self.mds_cache is None:
            return
        
        # High drift → reduce autonomy
        if self.mds_cache > 0.5:
            self.set_activation(ActivationLevel.DORMANT)
        elif self.mds_cache > 0.35:
            self.set_activation(ActivationLevel.GUARDED)
        elif self.evi_cache > 0.6:
            self.set_activation(ActivationLevel.ACTIVE)
    
    def set_activation(self, level: ActivationLevel) -> None:
        """Set activation level"""
        profiles = {
            ActivationLevel.DORMANT: ActivationProfile.dormant,
            ActivationLevel.GUARDED: ActivationProfile.guarded,
            ActivationLevel.ACTIVE: ActivationProfile.active,
            ActivationLevel.AUTONOMOUS: ActivationProfile.autonomous
        }
        self.current_profile = profiles[level]()
        
        self.history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'level': level.name,
            'evi': self.evi_cache,
            'mds': self.mds_cache
        })
    
    def export_for_b8(self) -> Dict[str, Any]:
        """Export state for B8 self-modeling"""
        return {
            'bridge_id': self.bridge_id,
            'activation': self.current_profile.level.name,
            'evi': self.evi_cache,
            'mds': self.mds_cache,
            'swarm': self.swarm.get_status(),
            'entropy_bounds': self.current_profile.entropy_bounds,
            'drift_tolerance': self.current_profile.drift_tolerance
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'bridge_id': self.bridge_id,
            'activation': self.current_profile.level.name,
            'checkpoint_interval': self.current_profile.checkpoint_interval,
            'evi': self.evi_cache,
            'mds': self.mds_cache,
            'swarm_coherence': self.swarm.compute_coherence()
        }


# =============================================================================
# REFLEXIVE OVERRIDE
# =============================================================================

@dataclass
class OverrideRequest:
    """Request for reflexive self-modification"""
    request_id: str
    target: str
    modification: Dict[str, Any]
    justification: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    approved: bool = False
    executed: bool = False


class ReflexiveOverride:
    """
    Bounded self-modification with audit trail.
    
    All overrides are logged and require justification.
    Certain modifications may require approval based on profile.
    """
    
    def __init__(self, override_id: str, profile: ActivationProfile):
        self.override_id = override_id
        self.profile = profile
        self.pending: List[OverrideRequest] = []
        self.executed: List[OverrideRequest] = []
    
    def request(
        self,
        target: str,
        modification: Dict[str, Any],
        justification: str
    ) -> OverrideRequest:
        """Request a self-modification"""
        req = OverrideRequest(
            request_id=f"OR-{len(self.pending) + len(self.executed):04d}",
            target=target,
            modification=modification,
            justification=justification
        )
        
        # Check if auto-approved based on profile
        mod_type = modification.get('type', 'unknown')
        if '*' not in self.profile.approval_required and \
           mod_type not in self.profile.approval_required:
            req.approved = True
        
        self.pending.append(req)
        return req
    
    def approve(self, request_id: str) -> bool:
        """Approve a pending request"""
        for req in self.pending:
            if req.request_id == request_id:
                req.approved = True
                return True
        return False
    
    def execute(self, request_id: str) -> bool:
        """Execute an approved request"""
        for i, req in enumerate(self.pending):
            if req.request_id == request_id and req.approved:
                req.executed = True
                self.executed.append(self.pending.pop(i))
                return True
        return False
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get full audit log"""
        return [
            {
                'request_id': r.request_id,
                'target': r.target,
                'modification': r.modification,
                'justification': r.justification,
                'approved': r.approved,
                'executed': r.executed,
                'timestamp': r.timestamp.isoformat()
            }
            for r in self.pending + self.executed
        ]


# =============================================================================
# TESTS
# =============================================================================

def run_tests():
    """Run B7 component tests"""
    import unittest
    
    class TestB7Components(unittest.TestCase):
        
        def test_swarm_coherence(self):
            swarm = SwarmCoherence("TEST-SWARM")
            
            # Add members with same hash
            for i in range(5):
                m = SwarmMember(f"M-{i}")
                m.update_hash({'value': 42})
                swarm.add_member(m)
            
            coherence = swarm.compute_coherence()
            self.assertEqual(coherence, 1.0)
            self.assertEqual(swarm.state, SwarmState.COHERENT)
        
        def test_swarm_divergence(self):
            swarm = SwarmCoherence("TEST-SWARM")
            
            # Add members with different hashes
            for i in range(5):
                m = SwarmMember(f"M-{i}")
                m.update_hash({'value': i})  # Different values
                swarm.add_member(m)
            
            coherence = swarm.compute_coherence()
            self.assertLess(coherence, QUORUM_THRESHOLD)
        
        def test_activation_profiles(self):
            dormant = ActivationProfile.dormant()
            self.assertEqual(dormant.level, ActivationLevel.DORMANT)
            self.assertIn('*', dormant.approval_required)
            
            autonomous = ActivationProfile.autonomous()
            self.assertEqual(autonomous.level, ActivationLevel.AUTONOMOUS)
            self.assertEqual(len(autonomous.approval_required), 0)
        
        def test_meta_kernel_bridge(self):
            bridge = MetaKernelBridge("TEST-BRIDGE")
            
            # Send good metrics
            bridge.receive_b6_metrics(evi=0.7, mds=0.2)
            self.assertEqual(
                bridge.current_profile.level,
                ActivationLevel.ACTIVE
            )
            
            # Send concerning metrics
            bridge.receive_b6_metrics(evi=0.5, mds=0.6)
            self.assertEqual(
                bridge.current_profile.level,
                ActivationLevel.DORMANT
            )
        
        def test_reflexive_override(self):
            profile = ActivationProfile.active()
            override = ReflexiveOverride("TEST-OR", profile)
            
            # Request that doesn't need approval
            req = override.request(
                target="config.parameter",
                modification={'type': 'adjust', 'value': 0.5},
                justification="Optimization"
            )
            self.assertTrue(req.approved)
            
            # Execute
            success = override.execute(req.request_id)
            self.assertTrue(success)
            
            # Check audit
            log = override.get_audit_log()
            self.assertEqual(len(log), 1)
            self.assertTrue(log[0]['executed'])
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestB7Components)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nB7 Tests: {result.testsRun} run, {len(result.failures)} failed")
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
