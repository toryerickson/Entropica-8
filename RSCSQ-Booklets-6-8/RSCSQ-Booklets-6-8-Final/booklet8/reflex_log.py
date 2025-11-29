"""
RSCS-Q Booklet 8: ReflexLog and Observer System
===============================================

Audit trail and observer integration for self-modeling systems.

This module implements:
- ReflexLog: Append-only audit log with hash chaining
- ObserverMesh: Multi-observer quorum system
- CollapseEvent: Observer-conditioned state collapse

Author: Entropica Research Collective
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum, auto
from datetime import datetime
from collections import deque
import hashlib
import json


# =============================================================================
# ENUMS
# =============================================================================

class EventType(Enum):
    """Types of reflex log events"""
    ACTION_PROPOSAL = auto()
    COLLAPSE = auto()
    REFLECTION = auto()
    MODIFICATION = auto()
    REPAIR = auto()
    ALERT = auto()
    QUORUM = auto()
    ESCALATION = auto()


class ObserverPhase(Enum):
    """Observer phases"""
    DORMANT = 0
    OBSERVING = 1
    VOTING = 2
    COLLAPSED = 3
    DIVERGENT = 4


class CollapseValidity(Enum):
    """Validity of a collapse event"""
    VALID = auto()
    INVALID_QUORUM = auto()
    MUTEX_VIOLATION = auto()
    AUDIT_FAILURE = auto()


# =============================================================================
# REFLEX LOG EVENT
# =============================================================================

@dataclass
class ReflexLogEvent:
    """
    A single event in the ReflexLog.
    
    Schema matches the specification:
    ts, capsule_id, observer_phase, event_type, quorum, 
    constraints_hash, entropy, delta_state_hash, lineage_ptr,
    rci, psr, shy
    """
    ts: datetime
    capsule_id: str
    observer_phase: int
    event_type: EventType
    
    # Quorum and constraints
    quorum: int = 0
    constraints_hash: str = ""
    
    # State
    entropy: float = 0.0
    delta_state_hash: str = ""
    lineage_ptr: str = ""
    
    # Metrics (RCI/PSR/SHY)
    rci: float = 0.0  # Reflex Coherence Index
    psr: float = 0.0  # Plan Stability Ratio
    shy: float = 0.0  # Shock Hygiene
    
    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Hash chain
    prev_hash: str = ""
    event_hash: str = ""
    
    def compute_hash(self) -> str:
        """Compute event hash for chain"""
        content = json.dumps({
            'ts': self.ts.isoformat(),
            'capsule_id': self.capsule_id,
            'event_type': self.event_type.name,
            'quorum': self.quorum,
            'constraints_hash': self.constraints_hash,
            'prev_hash': self.prev_hash
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ts': self.ts.isoformat(),
            'capsule_id': self.capsule_id,
            'observer_phase': self.observer_phase,
            'event_type': self.event_type.name,
            'quorum': self.quorum,
            'constraints_hash': self.constraints_hash,
            'entropy': self.entropy,
            'delta_state_hash': self.delta_state_hash,
            'lineage_ptr': self.lineage_ptr,
            'rci': self.rci,
            'psr': self.psr,
            'shy': self.shy,
            'prev_hash': self.prev_hash,
            'event_hash': self.event_hash
        }
    
    def to_json(self) -> str:
        """Convert to JSON format matching specification"""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# REFLEX LOG
# =============================================================================

class ReflexLog:
    """
    Append-only, hash-chained audit log.
    
    Implements the ReflexLog specification:
    - Every action proposal and collapse must emit a row
    - Hash chain for tamper evidence
    - Retention window management
    """
    
    def __init__(
        self, 
        log_id: str,
        retention_window: int = 1000
    ):
        self.log_id = log_id
        self.retention_window = retention_window
        self.events: deque = deque(maxlen=retention_window)
        self.last_hash: str = "GENESIS"
        self.event_count: int = 0
        
        # Indices for fast lookup
        self._capsule_index: Dict[str, List[int]] = {}
        self._type_index: Dict[EventType, List[int]] = {}
    
    def emit(
        self,
        capsule_id: str,
        event_type: EventType,
        observer_phase: int = 0,
        quorum: int = 0,
        constraints_hash: str = "",
        entropy: float = 0.0,
        delta_state_hash: str = "",
        lineage_ptr: str = "",
        rci: float = 0.0,
        psr: float = 0.0,
        shy: float = 0.0,
        context: Dict[str, Any] = None
    ) -> ReflexLogEvent:
        """
        Emit a new event to the log.
        
        Returns the created event.
        """
        event = ReflexLogEvent(
            ts=datetime.utcnow(),
            capsule_id=capsule_id,
            observer_phase=observer_phase,
            event_type=event_type,
            quorum=quorum,
            constraints_hash=constraints_hash,
            entropy=entropy,
            delta_state_hash=delta_state_hash,
            lineage_ptr=lineage_ptr,
            rci=rci,
            psr=psr,
            shy=shy,
            context=context or {},
            prev_hash=self.last_hash
        )
        
        # Compute and set hash
        event.event_hash = event.compute_hash()
        self.last_hash = event.event_hash
        
        # Add to log
        self.events.append(event)
        event_idx = self.event_count
        self.event_count += 1
        
        # Update indices
        if capsule_id not in self._capsule_index:
            self._capsule_index[capsule_id] = []
        self._capsule_index[capsule_id].append(event_idx)
        
        if event_type not in self._type_index:
            self._type_index[event_type] = []
        self._type_index[event_type].append(event_idx)
        
        return event
    
    def emit_action_proposal(
        self,
        capsule_id: str,
        constraints_hash: str,
        rci: float,
        psr: float,
        shy: float,
        **kwargs
    ) -> ReflexLogEvent:
        """Emit an action proposal event"""
        return self.emit(
            capsule_id=capsule_id,
            event_type=EventType.ACTION_PROPOSAL,
            constraints_hash=constraints_hash,
            rci=rci,
            psr=psr,
            shy=shy,
            **kwargs
        )
    
    def emit_collapse(
        self,
        capsule_id: str,
        observer_phase: int,
        quorum: int,
        delta_state_hash: str,
        **kwargs
    ) -> ReflexLogEvent:
        """Emit a collapse event"""
        return self.emit(
            capsule_id=capsule_id,
            event_type=EventType.COLLAPSE,
            observer_phase=observer_phase,
            quorum=quorum,
            delta_state_hash=delta_state_hash,
            **kwargs
        )
    
    def get_events_for_capsule(self, capsule_id: str) -> List[ReflexLogEvent]:
        """Get all events for a capsule"""
        if capsule_id not in self._capsule_index:
            return []
        
        events = []
        for idx in self._capsule_index[capsule_id]:
            # Handle deque rotation
            actual_idx = idx - (self.event_count - len(self.events))
            if 0 <= actual_idx < len(self.events):
                events.append(self.events[actual_idx])
        return events
    
    def get_events_by_type(self, event_type: EventType) -> List[ReflexLogEvent]:
        """Get all events of a specific type"""
        if event_type not in self._type_index:
            return []
        
        events = []
        for idx in self._type_index[event_type]:
            actual_idx = idx - (self.event_count - len(self.events))
            if 0 <= actual_idx < len(self.events):
                events.append(self.events[actual_idx])
        return events
    
    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """
        Verify hash chain integrity.
        
        Returns (valid, first_invalid_index or None).
        """
        if len(self.events) == 0:
            return True, None
        
        prev_hash = "GENESIS"
        for i, event in enumerate(self.events):
            if event.prev_hash != prev_hash:
                return False, i
            
            computed = event.compute_hash()
            if event.event_hash != computed:
                return False, i
            
            prev_hash = event.event_hash
        
        return True, None
    
    def export_csv(self) -> str:
        """Export log to CSV format"""
        lines = ["ts,capsule_id,observer_phase,event_type,quorum,constraints_hash,entropy,delta_state_hash,lineage_ptr,rci,psr,shy"]
        
        for event in self.events:
            line = f"{event.ts.isoformat()},{event.capsule_id},{event.observer_phase},{event.event_type.name},{event.quorum},{event.constraints_hash},{event.entropy},{event.delta_state_hash},{event.lineage_ptr},{event.rci},{event.psr},{event.shy}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get log statistics"""
        type_counts = {}
        for event in self.events:
            type_name = event.event_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        valid, invalid_idx = self.verify_chain()
        
        return {
            'log_id': self.log_id,
            'event_count': len(self.events),
            'total_emitted': self.event_count,
            'retention_window': self.retention_window,
            'type_counts': type_counts,
            'chain_valid': valid,
            'first_invalid': invalid_idx,
            'last_hash': self.last_hash
        }


# =============================================================================
# OBSERVER MESH
# =============================================================================

@dataclass
class Observer:
    """An observer in the mesh"""
    observer_id: str
    phase: ObserverPhase = ObserverPhase.DORMANT
    weight: float = 1.0
    last_vote: Optional[datetime] = None
    vote_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def vote(self, event_id: str, approve: bool) -> Dict[str, Any]:
        """Cast a vote"""
        self.phase = ObserverPhase.VOTING
        vote = {
            'event_id': event_id,
            'approve': approve,
            'weight': self.weight,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.vote_history.append(vote)
        self.last_vote = datetime.utcnow()
        return vote


class ObserverMesh:
    """
    Multi-observer quorum system.
    
    Implements observer-conditioned collapse from specification:
    - Quorum >= q_min required for valid collapse
    - No mutex violations
    - Reversible audit map required
    """
    
    def __init__(
        self,
        mesh_id: str,
        quorum_threshold: float = 0.67  # 2/3 majority
    ):
        self.mesh_id = mesh_id
        self.quorum_threshold = quorum_threshold
        self.observers: Dict[str, Observer] = {}
        self.mutex_set: Set[str] = set()  # Active mutex predicates
        self.collapse_history: List[Dict[str, Any]] = []
    
    def add_observer(self, observer: Observer) -> None:
        """Add an observer to the mesh"""
        self.observers[observer.observer_id] = observer
    
    def remove_observer(self, observer_id: str) -> bool:
        """Remove an observer"""
        if observer_id in self.observers:
            del self.observers[observer_id]
            return True
        return False
    
    def set_mutex(self, predicate_id: str) -> None:
        """Set a mutex predicate"""
        self.mutex_set.add(predicate_id)
    
    def clear_mutex(self, predicate_id: str) -> None:
        """Clear a mutex predicate"""
        self.mutex_set.discard(predicate_id)
    
    def request_votes(self, event_id: str) -> Dict[str, bool]:
        """Request votes from all observers"""
        votes = {}
        for obs_id, observer in self.observers.items():
            observer.phase = ObserverPhase.OBSERVING
        
        # Simulate voting (in real system, would be async)
        for obs_id, observer in self.observers.items():
            # Default: approve if no mutex violations
            approve = len(self.mutex_set) == 0
            vote = observer.vote(event_id, approve)
            votes[obs_id] = vote['approve']
        
        return votes
    
    def compute_quorum(self, votes: Dict[str, bool]) -> Tuple[int, float]:
        """
        Compute quorum from votes.
        
        Returns (approve_count, weighted_ratio).
        """
        total_weight = sum(self.observers[oid].weight for oid in votes.keys() if oid in self.observers)
        approve_weight = sum(
            self.observers[oid].weight 
            for oid, approved in votes.items() 
            if approved and oid in self.observers
        )
        
        approve_count = sum(1 for v in votes.values() if v)
        ratio = approve_weight / total_weight if total_weight > 0 else 0.0
        
        return approve_count, ratio
    
    def validate_collapse(
        self,
        event_id: str,
        pre_state_hash: str,
        post_state_hash: str
    ) -> Tuple[CollapseValidity, Dict[str, Any]]:
        """
        Validate a collapse event.
        
        Checks:
        1. Observer quorum >= threshold
        2. No mutex violations
        3. Reversible audit map (pre->post hashes)
        """
        # Get votes
        votes = self.request_votes(event_id)
        approve_count, ratio = self.compute_quorum(votes)
        
        # Check quorum
        if ratio < self.quorum_threshold:
            return CollapseValidity.INVALID_QUORUM, {
                'ratio': ratio,
                'threshold': self.quorum_threshold,
                'approve_count': approve_count
            }
        
        # Check mutex
        if len(self.mutex_set) > 0:
            return CollapseValidity.MUTEX_VIOLATION, {
                'mutex_predicates': list(self.mutex_set)
            }
        
        # Verify audit map exists
        if not pre_state_hash or not post_state_hash:
            return CollapseValidity.AUDIT_FAILURE, {
                'pre_hash': pre_state_hash,
                'post_hash': post_state_hash
            }
        
        # Record collapse
        collapse_record = {
            'event_id': event_id,
            'timestamp': datetime.utcnow().isoformat(),
            'quorum': approve_count,
            'ratio': ratio,
            'pre_state': pre_state_hash,
            'post_state': post_state_hash,
            'votes': votes
        }
        self.collapse_history.append(collapse_record)
        
        # Update observer phases
        for observer in self.observers.values():
            observer.phase = ObserverPhase.COLLAPSED
        
        return CollapseValidity.VALID, collapse_record
    
    def generate_quorum_certificate(self, collapse_record: Dict[str, Any]) -> str:
        """Generate a signed quorum certificate"""
        cert = {
            'mesh_id': self.mesh_id,
            'event_id': collapse_record.get('event_id'),
            'timestamp': collapse_record.get('timestamp'),
            'quorum': collapse_record.get('quorum'),
            'ratio': collapse_record.get('ratio'),
            'observers': list(self.observers.keys())
        }
        
        # Sign with hash
        cert_hash = hashlib.sha256(json.dumps(cert, sort_keys=True).encode()).hexdigest()
        cert['signature'] = cert_hash
        
        return json.dumps(cert, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics"""
        return {
            'mesh_id': self.mesh_id,
            'observer_count': len(self.observers),
            'quorum_threshold': self.quorum_threshold,
            'active_mutex': list(self.mutex_set),
            'collapse_count': len(self.collapse_history),
            'total_weight': sum(o.weight for o in self.observers.values())
        }


# =============================================================================
# RUNTIME METRICS
# =============================================================================

@dataclass
class RuntimeMetrics:
    """
    Runtime metrics (RCI/PSR/SHY) from specification.
    
    - RCI (Reflex Coherence Index): Rolling coherence vs. anchor rubric
    - PSR (Plan Stability Ratio): Ratio of plan steps preserved
    - SHY (Shock Hygiene): Normalized surprise outside envelope
    """
    rci: float = 0.65  # Gate: >= 0.65
    psr: float = 35.0  # Gate: >= 35
    shy: float = 0.08  # Gate: within nominal band
    
    # Thresholds
    rci_gate: float = 0.65
    psr_gate: float = 35.0
    shy_nominal_band: Tuple[float, float] = (0.0, 0.15)
    
    # History for trend analysis
    rci_history: List[float] = field(default_factory=list)
    psr_history: List[float] = field(default_factory=list)
    shy_history: List[float] = field(default_factory=list)
    
    def update(self, rci: float, psr: float, shy: float) -> None:
        """Update metrics"""
        self.rci = rci
        self.psr = psr
        self.shy = shy
        
        self.rci_history.append(rci)
        self.psr_history.append(psr)
        self.shy_history.append(shy)
        
        # Keep last 100
        if len(self.rci_history) > 100:
            self.rci_history = self.rci_history[-100:]
            self.psr_history = self.psr_history[-100:]
            self.shy_history = self.shy_history[-100:]
    
    def check_gates(self) -> Tuple[bool, List[str]]:
        """Check if all gates pass"""
        violations = []
        
        if self.rci < self.rci_gate:
            violations.append(f"RCI={self.rci:.3f} < {self.rci_gate}")
        
        if self.psr < self.psr_gate:
            violations.append(f"PSR={self.psr:.1f} < {self.psr_gate}")
        
        if not (self.shy_nominal_band[0] <= self.shy <= self.shy_nominal_band[1]):
            violations.append(f"SHY={self.shy:.3f} outside {self.shy_nominal_band}")
        
        return len(violations) == 0, violations
    
    def compute_trends(self) -> Dict[str, str]:
        """Compute metric trends"""
        def trend(history: List[float]) -> str:
            if len(history) < 5:
                return "insufficient_data"
            recent = history[-5:]
            earlier = history[-10:-5] if len(history) >= 10 else history[:5]
            
            recent_avg = np.mean(recent)
            earlier_avg = np.mean(earlier)
            
            if recent_avg > earlier_avg * 1.05:
                return "improving"
            elif recent_avg < earlier_avg * 0.95:
                return "declining"
            else:
                return "stable"
        
        return {
            'rci_trend': trend(self.rci_history),
            'psr_trend': trend(self.psr_history),
            'shy_trend': trend(self.shy_history)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        passed, violations = self.check_gates()
        return {
            'rci': self.rci,
            'psr': self.psr,
            'shy': self.shy,
            'gates_passed': passed,
            'violations': violations,
            'trends': self.compute_trends()
        }


# =============================================================================
# DSL PREDICATES
# =============================================================================

def reflex_log_valid(log: ReflexLog) -> bool:
    """DSL predicate: Check if reflex log chain is valid"""
    valid, _ = log.verify_chain()
    return valid


def quorum_reached(mesh: ObserverMesh, votes: Dict[str, bool]) -> bool:
    """DSL predicate: Check if quorum is reached"""
    _, ratio = mesh.compute_quorum(votes)
    return ratio >= mesh.quorum_threshold


def metrics_gates_passed(metrics: RuntimeMetrics) -> bool:
    """DSL predicate: Check if all metric gates pass"""
    passed, _ = metrics.check_gates()
    return passed


def collapse_valid(mesh: ObserverMesh, event_id: str, pre: str, post: str) -> bool:
    """DSL predicate: Check if collapse is valid"""
    validity, _ = mesh.validate_collapse(event_id, pre, post)
    return validity == CollapseValidity.VALID


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BOOKLET 8: REFLEXLOG AND OBSERVER SYSTEM DEMO")
    print("=" * 70)
    print()
    
    # Create ReflexLog
    log = ReflexLog("LOG-001")
    print(f"Created ReflexLog: {log.log_id}")
    
    # Emit events
    event1 = log.emit_action_proposal(
        capsule_id="CAP-001",
        constraints_hash="abc123",
        rci=0.72,
        psr=45,
        shy=0.06
    )
    print(f"Emitted action proposal: {event1.event_hash[:16]}")
    
    event2 = log.emit_collapse(
        capsule_id="CAP-001",
        observer_phase=3,
        quorum=5,
        delta_state_hash="def456"
    )
    print(f"Emitted collapse: {event2.event_hash[:16]}")
    print()
    
    # Verify chain
    valid, invalid_idx = log.verify_chain()
    print(f"Chain valid: {valid}")
    print()
    
    # Create observer mesh
    mesh = ObserverMesh("MESH-001")
    for i in range(5):
        mesh.add_observer(Observer(f"OBS-{i}", weight=1.0))
    print(f"Created mesh with {len(mesh.observers)} observers")
    
    # Validate collapse
    validity, record = mesh.validate_collapse(
        event_id="EVT-001",
        pre_state_hash="pre123",
        post_state_hash="post456"
    )
    print(f"Collapse validity: {validity.name}")
    print()
    
    # Check runtime metrics
    metrics = RuntimeMetrics()
    metrics.update(0.72, 45, 0.06)
    passed, violations = metrics.check_gates()
    print(f"Metrics gates passed: {passed}")
    print(f"Metrics: {metrics.to_dict()}")
    print()
    
    # Export log
    print("Log statistics:")
    print(f"  {log.get_statistics()}")
    
    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

