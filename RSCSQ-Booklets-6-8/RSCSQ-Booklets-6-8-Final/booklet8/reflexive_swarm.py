"""
RSCS-Q Booklet 8: Reflexive Swarm Enhancement Module
=====================================================

This module adds advanced self-reflective capabilities:
- Heartbeat Protocol: Recursive parent/swarm check-in
- CapsuleDriftVector: Multidimensional drift signatures
- Reflexive Swarm Agreement: Capsules vote on each other's drift
- ModificationCostIndex: Cost tracking for self-modifications
- cascade_alert(): Ancestral anomaly detection
- Metric Graphs: EVI/MDS time series tracking

Author: Entropica Research Collective
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum, auto
import hashlib
import json


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class HeartbeatStatus(Enum):
    """Status of heartbeat check"""
    ALIVE = auto()
    STALE = auto()
    MISSING = auto()
    QUARANTINED = auto()


class AlertSeverity(Enum):
    """Severity levels for cascade alerts"""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class DriftDimension(Enum):
    """Dimensions of drift measurement"""
    BEHAVIORAL = auto()      # Behavior pattern drift
    STRUCTURAL = auto()      # Identity graph drift
    ALIGNMENT = auto()       # Rubric alignment drift
    TEMPORAL = auto()        # Time-based decay drift
    CONTEXTUAL = auto()      # Environmental drift


# Thresholds
HEARTBEAT_TIMEOUT_SECONDS = 30.0
HEARTBEAT_STALE_SECONDS = 15.0
MAX_MODIFICATION_COST = 100.0
CASCADE_ALERT_THRESHOLD = 0.4
SWARM_QUORUM_THRESHOLD = 0.67


# =============================================================================
# CAPSULE DRIFT VECTOR (Multidimensional)
# =============================================================================

@dataclass
class CapsuleDriftVector:
    """
    Multidimensional drift signature for a capsule.
    
    Tracks drift across multiple dimensions:
    - behavioral: Changes in action patterns
    - structural: Changes in identity graph
    - alignment: Divergence from rubric anchor
    - temporal: Time-based decay
    - contextual: Environmental/context drift
    """
    
    capsule_id: str
    dimensions: Dict[DriftDimension, float] = field(default_factory=dict)
    history: List[Tuple[datetime, Dict[DriftDimension, float]]] = field(default_factory=list)
    baseline: Optional[Dict[DriftDimension, float]] = None
    
    def __post_init__(self):
        # Initialize all dimensions to 0
        for dim in DriftDimension:
            if dim not in self.dimensions:
                self.dimensions[dim] = 0.0
    
    def set_dimension(self, dimension: DriftDimension, value: float) -> None:
        """Set drift value for a dimension"""
        self.dimensions[dimension] = max(0.0, min(1.0, value))
        self._record_history()
    
    def update_dimension(self, dimension: DriftDimension, delta: float) -> None:
        """Update drift by delta"""
        current = self.dimensions.get(dimension, 0.0)
        self.set_dimension(dimension, current + delta)
    
    def set_baseline(self) -> None:
        """Set current values as baseline"""
        self.baseline = self.dimensions.copy()
    
    def compute_magnitude(self) -> float:
        """Compute L2 norm of drift vector"""
        values = list(self.dimensions.values())
        return float(np.sqrt(sum(v**2 for v in values)))
    
    def compute_delta_from_baseline(self) -> float:
        """Compute delta from baseline"""
        if self.baseline is None:
            return 0.0
        
        delta_sq = 0.0
        for dim in DriftDimension:
            base_val = self.baseline.get(dim, 0.0)
            curr_val = self.dimensions.get(dim, 0.0)
            delta_sq += (curr_val - base_val) ** 2
        
        return float(np.sqrt(delta_sq))
    
    def exceeds_threshold(self, threshold: float = 0.35) -> bool:
        """Check if any dimension exceeds threshold"""
        return any(v > threshold for v in self.dimensions.values())
    
    def get_worst_dimension(self) -> Tuple[DriftDimension, float]:
        """Get dimension with highest drift"""
        worst_dim = max(self.dimensions, key=self.dimensions.get)
        return worst_dim, self.dimensions[worst_dim]
    
    def _record_history(self) -> None:
        """Record current state to history"""
        self.history.append((datetime.utcnow(), self.dimensions.copy()))
        # Keep last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            'capsule_id': self.capsule_id,
            'dimensions': {d.name: v for d, v in self.dimensions.items()},
            'magnitude': self.compute_magnitude(),
            'delta_from_baseline': self.compute_delta_from_baseline()
        }


# =============================================================================
# HEARTBEAT PROTOCOL
# =============================================================================

@dataclass
class HeartbeatRecord:
    """Record of a heartbeat check-in"""
    capsule_id: str
    timestamp: datetime
    parent_id: Optional[str]
    swarm_id: Optional[str]
    status: HeartbeatStatus
    drift_magnitude: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class HeartbeatProtocol:
    """
    Manages recursive heartbeat check-ins between capsules and their parents/swarms.
    
    Features:
    - Periodic check-in registration
    - Staleness detection
    - Parent notification on drift
    - Swarm-level health monitoring
    """
    
    def __init__(self, protocol_id: str):
        self.protocol_id = protocol_id
        self.heartbeats: Dict[str, HeartbeatRecord] = {}  # capsule_id -> last heartbeat
        self.swarm_members: Dict[str, Set[str]] = defaultdict(set)  # swarm_id -> capsule_ids
        self.parent_children: Dict[str, Set[str]] = defaultdict(set)  # parent_id -> child_ids
        self.alerts: List[Dict[str, Any]] = []
    
    def register_capsule(
        self, 
        capsule_id: str, 
        parent_id: Optional[str] = None,
        swarm_id: Optional[str] = None
    ) -> None:
        """Register a capsule for heartbeat monitoring"""
        if parent_id:
            self.parent_children[parent_id].add(capsule_id)
        if swarm_id:
            self.swarm_members[swarm_id].add(capsule_id)
        
        # Initial heartbeat
        self.checkin(capsule_id, parent_id, swarm_id, 0.0)
    
    def checkin(
        self,
        capsule_id: str,
        parent_id: Optional[str] = None,
        swarm_id: Optional[str] = None,
        drift_magnitude: float = 0.0,
        metadata: Dict[str, Any] = None
    ) -> HeartbeatStatus:
        """Record a heartbeat check-in"""
        record = HeartbeatRecord(
            capsule_id=capsule_id,
            timestamp=datetime.utcnow(),
            parent_id=parent_id,
            swarm_id=swarm_id,
            status=HeartbeatStatus.ALIVE,
            drift_magnitude=drift_magnitude,
            metadata=metadata or {}
        )
        self.heartbeats[capsule_id] = record
        
        # Check if drift is high - notify parent
        if drift_magnitude > CASCADE_ALERT_THRESHOLD and parent_id:
            self._notify_parent(capsule_id, parent_id, drift_magnitude)
        
        return HeartbeatStatus.ALIVE
    
    def check_status(self, capsule_id: str) -> HeartbeatStatus:
        """Check the current status of a capsule"""
        if capsule_id not in self.heartbeats:
            return HeartbeatStatus.MISSING
        
        record = self.heartbeats[capsule_id]
        age = (datetime.utcnow() - record.timestamp).total_seconds()
        
        if age > HEARTBEAT_TIMEOUT_SECONDS:
            return HeartbeatStatus.MISSING
        elif age > HEARTBEAT_STALE_SECONDS:
            return HeartbeatStatus.STALE
        else:
            return record.status
    
    def get_children_status(self, parent_id: str) -> Dict[str, HeartbeatStatus]:
        """Get status of all children of a parent"""
        children = self.parent_children.get(parent_id, set())
        return {cid: self.check_status(cid) for cid in children}
    
    def get_swarm_health(self, swarm_id: str) -> Dict[str, Any]:
        """Get health status of a swarm"""
        members = self.swarm_members.get(swarm_id, set())
        if not members:
            return {'swarm_id': swarm_id, 'alive_ratio': 0.0, 'members': 0}
        
        statuses = {mid: self.check_status(mid) for mid in members}
        alive = sum(1 for s in statuses.values() if s == HeartbeatStatus.ALIVE)
        
        return {
            'swarm_id': swarm_id,
            'members': len(members),
            'alive': alive,
            'stale': sum(1 for s in statuses.values() if s == HeartbeatStatus.STALE),
            'missing': sum(1 for s in statuses.values() if s == HeartbeatStatus.MISSING),
            'alive_ratio': alive / len(members),
            'healthy': alive / len(members) >= SWARM_QUORUM_THRESHOLD
        }
    
    def quarantine_capsule(self, capsule_id: str, reason: str) -> None:
        """Mark a capsule as quarantined"""
        if capsule_id in self.heartbeats:
            self.heartbeats[capsule_id].status = HeartbeatStatus.QUARANTINED
        
        self.alerts.append({
            'type': 'quarantine',
            'capsule_id': capsule_id,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def _notify_parent(self, capsule_id: str, parent_id: str, drift: float) -> None:
        """Notify parent of child's high drift"""
        self.alerts.append({
            'type': 'drift_notification',
            'child_id': capsule_id,
            'parent_id': parent_id,
            'drift': drift,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get protocol statistics"""
        statuses = [self.check_status(cid) for cid in self.heartbeats]
        return {
            'total_capsules': len(self.heartbeats),
            'alive': sum(1 for s in statuses if s == HeartbeatStatus.ALIVE),
            'stale': sum(1 for s in statuses if s == HeartbeatStatus.STALE),
            'missing': sum(1 for s in statuses if s == HeartbeatStatus.MISSING),
            'quarantined': sum(1 for s in statuses if s == HeartbeatStatus.QUARANTINED),
            'swarms': len(self.swarm_members),
            'alerts': len(self.alerts)
        }


# =============================================================================
# REFLEXIVE SWARM AGREEMENT
# =============================================================================

@dataclass
class DriftVote:
    """A vote on another capsule's drift status"""
    voter_id: str
    target_id: str
    timestamp: datetime
    assessed_drift: float
    vote: str  # 'healthy', 'drifting', 'critical', 'abstain'
    confidence: float
    reasoning: str = ""


class ReflexiveSwarmAgreement:
    """
    Implements reflexive swarm agreement protocol where capsules vote on each other's drift.
    
    Features:
    - Capsules evaluate and vote on peer drift
    - Quorum-based consensus
    - Self-declassification when consensus says "drifting"
    - Emergent correction without human intervention
    """
    
    def __init__(self, swarm_id: str, quorum_threshold: float = 0.67):
        self.swarm_id = swarm_id
        self.quorum_threshold = quorum_threshold
        self.members: Dict[str, CapsuleDriftVector] = {}  # capsule_id -> drift vector
        self.votes: Dict[str, List[DriftVote]] = defaultdict(list)  # target_id -> votes
        self.consensus: Dict[str, Dict[str, Any]] = {}  # target_id -> consensus result
        self.actions: List[Dict[str, Any]] = []  # Actions taken
    
    def add_member(self, capsule_id: str, drift_vector: CapsuleDriftVector) -> None:
        """Add a capsule to the swarm"""
        self.members[capsule_id] = drift_vector
    
    def submit_vote(
        self,
        voter_id: str,
        target_id: str,
        assessed_drift: float,
        confidence: float = 0.8,
        reasoning: str = ""
    ) -> DriftVote:
        """Submit a vote on a target capsule's drift"""
        # Determine vote category
        if assessed_drift < 0.15:
            vote = 'healthy'
        elif assessed_drift < 0.30:
            vote = 'drifting'
        else:
            vote = 'critical'
        
        drift_vote = DriftVote(
            voter_id=voter_id,
            target_id=target_id,
            timestamp=datetime.utcnow(),
            assessed_drift=assessed_drift,
            vote=vote,
            confidence=confidence,
            reasoning=reasoning
        )
        
        self.votes[target_id].append(drift_vote)
        return drift_vote
    
    def evaluate_peer(
        self,
        evaluator_id: str,
        target_id: str,
        noise_factor: float = 0.05
    ) -> DriftVote:
        """Have one capsule evaluate another's drift"""
        if target_id not in self.members:
            return self.submit_vote(evaluator_id, target_id, 0.0, 0.5, "target not found")
        
        target_drift = self.members[target_id]
        
        # Evaluate based on drift magnitude + some noise
        assessed = target_drift.compute_magnitude() + np.random.normal(0, noise_factor)
        assessed = max(0.0, min(1.0, assessed))
        
        # Confidence based on evaluator's own health
        evaluator_drift = self.members.get(evaluator_id)
        if evaluator_drift:
            own_health = 1.0 - evaluator_drift.compute_magnitude()
            confidence = 0.5 + 0.5 * own_health  # 0.5 to 1.0
        else:
            confidence = 0.5
        
        return self.submit_vote(evaluator_id, target_id, assessed, confidence)
    
    def run_peer_evaluation_round(self) -> Dict[str, Any]:
        """Have all capsules evaluate all others"""
        round_votes = 0
        
        for evaluator_id in self.members:
            for target_id in self.members:
                if evaluator_id != target_id:
                    self.evaluate_peer(evaluator_id, target_id)
                    round_votes += 1
        
        return {
            'swarm_id': self.swarm_id,
            'evaluators': len(self.members),
            'votes_cast': round_votes,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def compute_consensus(self, target_id: str) -> Dict[str, Any]:
        """Compute consensus on a target's drift status"""
        votes = self.votes.get(target_id, [])
        if not votes:
            return {'target_id': target_id, 'consensus': 'no_votes', 'confidence': 0.0}
        
        # Weight votes by confidence
        weighted_votes = defaultdict(float)
        total_weight = 0.0
        
        for vote in votes:
            weighted_votes[vote.vote] += vote.confidence
            total_weight += vote.confidence
        
        # Find majority
        if total_weight == 0:
            return {'target_id': target_id, 'consensus': 'no_confidence', 'confidence': 0.0}
        
        best_vote = max(weighted_votes, key=weighted_votes.get)
        consensus_ratio = weighted_votes[best_vote] / total_weight
        
        # Compute average assessed drift
        avg_drift = sum(v.assessed_drift * v.confidence for v in votes) / total_weight
        
        result = {
            'target_id': target_id,
            'consensus': best_vote,
            'consensus_ratio': consensus_ratio,
            'quorum_reached': consensus_ratio >= self.quorum_threshold,
            'average_assessed_drift': avg_drift,
            'vote_count': len(votes),
            'vote_breakdown': dict(weighted_votes)
        }
        
        self.consensus[target_id] = result
        return result
    
    def trigger_self_declassification(self, capsule_id: str) -> bool:
        """
        Check if consensus says capsule is drifting and trigger self-declassification.
        Returns True if declassification triggered.
        """
        consensus = self.consensus.get(capsule_id)
        if not consensus:
            consensus = self.compute_consensus(capsule_id)
        
        if consensus['quorum_reached'] and consensus['consensus'] in ('drifting', 'critical'):
            # Trigger self-declassification
            action = {
                'type': 'self_declassification',
                'capsule_id': capsule_id,
                'consensus': consensus['consensus'],
                'avg_drift': consensus['average_assessed_drift'],
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'requested_reevaluation'
            }
            self.actions.append(action)
            return True
        
        return False
    
    def run_full_consensus(self) -> Dict[str, Any]:
        """Run consensus for all members and trigger actions"""
        declassified = []
        
        for capsule_id in self.members:
            self.compute_consensus(capsule_id)
            if self.trigger_self_declassification(capsule_id):
                declassified.append(capsule_id)
        
        return {
            'swarm_id': self.swarm_id,
            'members_evaluated': len(self.members),
            'declassified': declassified,
            'actions_triggered': len(self.actions)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swarm agreement statistics"""
        return {
            'swarm_id': self.swarm_id,
            'member_count': len(self.members),
            'total_votes': sum(len(v) for v in self.votes.values()),
            'consensus_computed': len(self.consensus),
            'actions_taken': len(self.actions)
        }


# =============================================================================
# MODIFICATION COST INDEX
# =============================================================================

@dataclass
class ModificationCost:
    """Cost record for a self-modification"""
    modification_id: str
    capsule_id: str
    timestamp: datetime
    modification_type: str
    base_cost: float
    complexity_factor: float
    risk_factor: float
    total_cost: float
    approved: bool
    budget_remaining: float


class ModificationCostIndex:
    """
    Tracks and controls the cost of self-modifications.
    
    Prevents runaway self-modification by:
    - Assigning costs to different modification types
    - Tracking cumulative cost
    - Enforcing budget limits
    - Requiring escalation for high-cost modifications
    """
    
    # Base costs by modification type
    BASE_COSTS = {
        'parameter_adjustment': 5.0,
        'rubric_update': 15.0,
        'constraint_tightening': 10.0,
        'constraint_relaxation': 50.0,  # Very expensive!
        'lineage_extension': 8.0,
        'fingerprint_refresh': 3.0,
        'rollback': 20.0,
        'spawn_child': 12.0
    }
    
    def __init__(self, index_id: str, budget: float = 100.0):
        self.index_id = index_id
        self.budget = budget
        self.spent: float = 0.0
        self.modifications: List[ModificationCost] = []
        self.denied: List[ModificationCost] = []
    
    def compute_cost(
        self,
        modification_type: str,
        complexity_factor: float = 1.0,
        risk_factor: float = 1.0
    ) -> float:
        """Compute the total cost of a modification"""
        base = self.BASE_COSTS.get(modification_type, 10.0)
        return base * complexity_factor * risk_factor
    
    def request_modification(
        self,
        capsule_id: str,
        modification_type: str,
        complexity_factor: float = 1.0,
        risk_factor: float = 1.0
    ) -> Tuple[bool, ModificationCost]:
        """Request a modification, checking budget"""
        total_cost = self.compute_cost(modification_type, complexity_factor, risk_factor)
        
        cost_record = ModificationCost(
            modification_id=f"MOD-{len(self.modifications):04d}",
            capsule_id=capsule_id,
            timestamp=datetime.utcnow(),
            modification_type=modification_type,
            base_cost=self.BASE_COSTS.get(modification_type, 10.0),
            complexity_factor=complexity_factor,
            risk_factor=risk_factor,
            total_cost=total_cost,
            approved=False,
            budget_remaining=self.budget - self.spent
        )
        
        if self.spent + total_cost <= self.budget:
            cost_record.approved = True
            self.spent += total_cost
            cost_record.budget_remaining = self.budget - self.spent
            self.modifications.append(cost_record)
            return True, cost_record
        else:
            self.denied.append(cost_record)
            return False, cost_record
    
    def rollback_modification(self, modification_id: str) -> bool:
        """Rollback a modification and refund the cost"""
        for mod in self.modifications:
            if mod.modification_id == modification_id:
                self.spent -= mod.total_cost
                self.modifications.remove(mod)
                return True
        return False
    
    def reset_budget(self, new_budget: float = None) -> None:
        """Reset the budget (e.g., for new epoch)"""
        if new_budget:
            self.budget = new_budget
        self.spent = 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cost index statistics"""
        return {
            'index_id': self.index_id,
            'budget': self.budget,
            'spent': self.spent,
            'remaining': self.budget - self.spent,
            'utilization': self.spent / self.budget if self.budget > 0 else 0,
            'modifications_approved': len(self.modifications),
            'modifications_denied': len(self.denied),
            'by_type': defaultdict(int, {m.modification_type: 1 for m in self.modifications})
        }


# =============================================================================
# CASCADE ALERT SYSTEM
# =============================================================================

@dataclass
class CascadeAlert:
    """An alert triggered by ancestral anomaly detection"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    source_capsule_id: str
    affected_lineage: List[str]
    message: str
    drift_data: Dict[str, float]
    recommended_action: str
    acknowledged: bool = False
    resolved: bool = False


class CascadeAlertSystem:
    """
    Manages cascade alerts for ancestral anomaly detection.
    
    When drift is detected in a capsule, this system:
    - Traces the lineage upward to find root cause
    - Traces downward to identify affected descendants
    - Generates alerts with recommended actions
    - Tracks alert lifecycle
    """
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.alerts: List[CascadeAlert] = []
        self.lineage_map: Dict[str, str] = {}  # child_id -> parent_id
        self.children_map: Dict[str, Set[str]] = defaultdict(set)  # parent_id -> children
    
    def register_lineage(self, capsule_id: str, parent_id: Optional[str]) -> None:
        """Register a capsule's lineage"""
        if parent_id:
            self.lineage_map[capsule_id] = parent_id
            self.children_map[parent_id].add(capsule_id)
    
    def trace_ancestors(self, capsule_id: str) -> List[str]:
        """Trace all ancestors of a capsule"""
        ancestors = []
        current = capsule_id
        
        while current in self.lineage_map:
            parent = self.lineage_map[current]
            ancestors.append(parent)
            current = parent
        
        return ancestors
    
    def trace_descendants(self, capsule_id: str) -> List[str]:
        """Trace all descendants of a capsule"""
        descendants = []
        to_visit = list(self.children_map.get(capsule_id, []))
        
        while to_visit:
            child = to_visit.pop(0)
            descendants.append(child)
            to_visit.extend(self.children_map.get(child, []))
        
        return descendants
    
    def cascade_alert(
        self,
        source_capsule_id: str,
        message: str,
        drift_data: Dict[str, float] = None,
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> CascadeAlert:
        """
        Generate a cascade alert for ancestral anomaly detection.
        
        This is the main function called when anomaly is detected.
        """
        # Trace lineage
        ancestors = self.trace_ancestors(source_capsule_id)
        descendants = self.trace_descendants(source_capsule_id)
        affected_lineage = ancestors + [source_capsule_id] + descendants
        
        # Determine recommended action based on severity
        if severity == AlertSeverity.EMERGENCY:
            recommended_action = "immediate_quarantine_cascade"
        elif severity == AlertSeverity.CRITICAL:
            recommended_action = "isolate_and_repair"
        elif severity == AlertSeverity.WARNING:
            recommended_action = "monitor_and_evaluate"
        else:
            recommended_action = "log_and_continue"
        
        alert = CascadeAlert(
            alert_id=f"ALERT-{len(self.alerts):04d}",
            timestamp=datetime.utcnow(),
            severity=severity,
            source_capsule_id=source_capsule_id,
            affected_lineage=affected_lineage,
            message=message,
            drift_data=drift_data or {},
            recommended_action=recommended_action
        )
        
        self.alerts.append(alert)
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_active_alerts(self) -> List[CascadeAlert]:
        """Get all unresolved alerts"""
        return [a for a in self.alerts if not a.resolved]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        by_severity = defaultdict(int)
        for alert in self.alerts:
            by_severity[alert.severity.name] += 1
        
        return {
            'system_id': self.system_id,
            'total_alerts': len(self.alerts),
            'active_alerts': len(self.get_active_alerts()),
            'acknowledged': sum(1 for a in self.alerts if a.acknowledged),
            'resolved': sum(1 for a in self.alerts if a.resolved),
            'by_severity': dict(by_severity)
        }


# =============================================================================
# METRIC TIME SERIES TRACKER
# =============================================================================

@dataclass
class MetricSample:
    """A sample of metrics at a point in time"""
    timestamp: datetime
    capsule_id: str
    evi: float
    mds: float
    soc: float  # State of Coherence
    rci: float
    psr: float
    shy: float


class MetricTimeSeriesTracker:
    """
    Tracks EVI/MDS/SOC and other metrics over time.
    
    Features:
    - Rolling time series
    - Anomaly detection via slope analysis
    - Event-triggered alerts
    - Graph data export
    """
    
    def __init__(self, tracker_id: str, window_size: int = 100):
        self.tracker_id = tracker_id
        self.window_size = window_size
        self.samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.alerts: List[Dict[str, Any]] = []
        
        # Thresholds for anomaly detection
        self.thresholds = {
            'evi_min': 0.4,
            'evi_max': 1.0,
            'mds_max': 0.5,
            'soc_min': 0.5,
            'slope_warning': 0.1,  # Slope threshold for warning
            'slope_critical': 0.2   # Slope threshold for critical
        }
    
    def record_sample(
        self,
        capsule_id: str,
        evi: float,
        mds: float,
        soc: float = 0.8,
        rci: float = 0.7,
        psr: float = 40.0,
        shy: float = 0.05
    ) -> MetricSample:
        """Record a metric sample"""
        sample = MetricSample(
            timestamp=datetime.utcnow(),
            capsule_id=capsule_id,
            evi=evi,
            mds=mds,
            soc=soc,
            rci=rci,
            psr=psr,
            shy=shy
        )
        self.samples[capsule_id].append(sample)
        
        # Check for anomalies
        self._check_anomalies(capsule_id)
        
        return sample
    
    def _check_anomalies(self, capsule_id: str) -> None:
        """Check for anomalies in recent samples"""
        samples = list(self.samples[capsule_id])
        if len(samples) < 3:
            return
        
        # Check current values against thresholds
        current = samples[-1]
        
        if current.evi < self.thresholds['evi_min']:
            self._trigger_alert(capsule_id, 'evi_low', current.evi)
        
        if current.mds > self.thresholds['mds_max']:
            self._trigger_alert(capsule_id, 'mds_high', current.mds)
        
        if current.soc < self.thresholds['soc_min']:
            self._trigger_alert(capsule_id, 'soc_low', current.soc)
        
        # Check slopes
        if len(samples) >= 5:
            evi_slope = self._compute_slope([s.evi for s in samples[-5:]])
            mds_slope = self._compute_slope([s.mds for s in samples[-5:]])
            
            if evi_slope < -self.thresholds['slope_warning']:
                self._trigger_alert(capsule_id, 'evi_declining', evi_slope)
            
            if mds_slope > self.thresholds['slope_warning']:
                self._trigger_alert(capsule_id, 'mds_increasing', mds_slope)
    
    def _compute_slope(self, values: List[float]) -> float:
        """Compute linear slope of values"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    def _trigger_alert(self, capsule_id: str, alert_type: str, value: float) -> None:
        """Trigger a metric alert"""
        self.alerts.append({
            'capsule_id': capsule_id,
            'alert_type': alert_type,
            'value': value,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def get_capsule_history(self, capsule_id: str) -> List[Dict[str, Any]]:
        """Get metric history for a capsule"""
        samples = self.samples.get(capsule_id, [])
        return [
            {
                'timestamp': s.timestamp.isoformat(),
                'evi': s.evi,
                'mds': s.mds,
                'soc': s.soc,
                'rci': s.rci,
                'psr': s.psr,
                'shy': s.shy
            }
            for s in samples
        ]
    
    def export_csv(self, capsule_id: str = None) -> str:
        """Export metrics as CSV"""
        lines = ["timestamp,capsule_id,evi,mds,soc,rci,psr,shy"]
        
        if capsule_id:
            capsules = [capsule_id]
        else:
            capsules = list(self.samples.keys())
        
        for cid in capsules:
            for sample in self.samples[cid]:
                lines.append(
                    f"{sample.timestamp.isoformat()},{sample.capsule_id},"
                    f"{sample.evi:.4f},{sample.mds:.4f},{sample.soc:.4f},"
                    f"{sample.rci:.4f},{sample.psr:.4f},{sample.shy:.4f}"
                )
        
        return '\n'.join(lines)
    
    def get_statistics(self, capsule_id: str) -> Dict[str, Any]:
        """Get statistics for a capsule's metrics"""
        samples = list(self.samples.get(capsule_id, []))
        if not samples:
            return {'capsule_id': capsule_id, 'sample_count': 0}
        
        evis = [s.evi for s in samples]
        mdss = [s.mds for s in samples]
        
        return {
            'capsule_id': capsule_id,
            'sample_count': len(samples),
            'evi_mean': float(np.mean(evis)),
            'evi_std': float(np.std(evis)),
            'evi_trend': self._compute_slope(evis[-10:]) if len(evis) >= 10 else 0.0,
            'mds_mean': float(np.mean(mdss)),
            'mds_std': float(np.std(mdss)),
            'mds_trend': self._compute_slope(mdss[-10:]) if len(mdss) >= 10 else 0.0,
            'alerts': len([a for a in self.alerts if a['capsule_id'] == capsule_id])
        }


# =============================================================================
# CAPSULE FLAG SYSTEM
# =============================================================================

@dataclass
class CapsuleFlag:
    """A flag set on a capsule"""
    flag_id: str
    capsule_id: str
    flag_type: str
    timestamp: datetime
    severity: str
    message: str
    auto_clear: bool
    cleared: bool = False
    cleared_at: Optional[datetime] = None


class CapsuleFlagSystem:
    """
    Manages flags on capsules for identity drift and other conditions.
    
    Supports:
    - capsule.flag("identity_drift")
    - Auto-clearing flags
    - Flag queries by type/capsule
    """
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.flags: Dict[str, List[CapsuleFlag]] = defaultdict(list)
        self.flag_counter = 0
    
    def flag(
        self,
        capsule_id: str,
        flag_type: str,
        message: str = "",
        severity: str = "warning",
        auto_clear: bool = False
    ) -> CapsuleFlag:
        """Set a flag on a capsule"""
        self.flag_counter += 1
        
        flag = CapsuleFlag(
            flag_id=f"FLAG-{self.flag_counter:04d}",
            capsule_id=capsule_id,
            flag_type=flag_type,
            timestamp=datetime.utcnow(),
            severity=severity,
            message=message,
            auto_clear=auto_clear
        )
        
        self.flags[capsule_id].append(flag)
        return flag
    
    def clear_flag(self, flag_id: str) -> bool:
        """Clear a specific flag"""
        for capsule_flags in self.flags.values():
            for flag in capsule_flags:
                if flag.flag_id == flag_id:
                    flag.cleared = True
                    flag.cleared_at = datetime.utcnow()
                    return True
        return False
    
    def get_active_flags(self, capsule_id: str) -> List[CapsuleFlag]:
        """Get active flags for a capsule"""
        return [f for f in self.flags.get(capsule_id, []) if not f.cleared]
    
    def has_flag(self, capsule_id: str, flag_type: str) -> bool:
        """Check if capsule has an active flag of given type"""
        active = self.get_active_flags(capsule_id)
        return any(f.flag_type == flag_type for f in active)
    
    def get_all_flagged(self, flag_type: str = None) -> List[str]:
        """Get all capsules with active flags (optionally of specific type)"""
        flagged = []
        for capsule_id, flags in self.flags.items():
            active = [f for f in flags if not f.cleared]
            if flag_type:
                active = [f for f in active if f.flag_type == flag_type]
            if active:
                flagged.append(capsule_id)
        return flagged
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get flag system statistics"""
        total = sum(len(f) for f in self.flags.values())
        active = sum(len([x for x in f if not x.cleared]) for f in self.flags.values())
        
        by_type = defaultdict(int)
        for flags in self.flags.values():
            for f in flags:
                if not f.cleared:
                    by_type[f.flag_type] += 1
        
        return {
            'system_id': self.system_id,
            'total_flags': total,
            'active_flags': active,
            'cleared_flags': total - active,
            'capsules_flagged': len([c for c, f in self.flags.items() if any(not x.cleared for x in f)]),
            'by_type': dict(by_type)
        }


# =============================================================================
# INTEGRATED REFLEXIVE SWARM CONTROLLER
# =============================================================================

class ReflexiveSwarmController:
    """
    Integrated controller for all reflexive swarm capabilities.
    
    Combines:
    - Heartbeat protocol
    - Drift vectors
    - Swarm agreement
    - Cost tracking
    - Cascade alerts
    - Metric tracking
    - Flag system
    """
    
    def __init__(self, controller_id: str):
        self.controller_id = controller_id
        
        # Initialize subsystems
        self.heartbeat = HeartbeatProtocol(f"{controller_id}-HB")
        self.swarm_agreement = ReflexiveSwarmAgreement(f"{controller_id}-SA")
        self.cost_index = ModificationCostIndex(f"{controller_id}-CI")
        self.cascade_alerts = CascadeAlertSystem(f"{controller_id}-CA")
        self.metric_tracker = MetricTimeSeriesTracker(f"{controller_id}-MT")
        self.flag_system = CapsuleFlagSystem(f"{controller_id}-FS")
        
        # Capsule registry
        self.capsules: Dict[str, CapsuleDriftVector] = {}
    
    def register_capsule(
        self,
        capsule_id: str,
        parent_id: Optional[str] = None,
        swarm_id: Optional[str] = None
    ) -> CapsuleDriftVector:
        """Register a capsule with all subsystems"""
        # Create drift vector
        drift_vector = CapsuleDriftVector(capsule_id)
        self.capsules[capsule_id] = drift_vector
        
        # Register with subsystems
        self.heartbeat.register_capsule(capsule_id, parent_id, swarm_id)
        self.swarm_agreement.add_member(capsule_id, drift_vector)
        self.cascade_alerts.register_lineage(capsule_id, parent_id)
        
        return drift_vector
    
    def update_drift(
        self,
        capsule_id: str,
        dimension: DriftDimension,
        value: float
    ) -> None:
        """Update a capsule's drift"""
        if capsule_id in self.capsules:
            self.capsules[capsule_id].set_dimension(dimension, value)
            
            # Check for flags
            if value > 0.35:
                self.flag_system.flag(
                    capsule_id,
                    'identity_drift',
                    f"{dimension.name} drift at {value:.2f}",
                    severity='warning'
                )
    
    def record_metrics(
        self,
        capsule_id: str,
        evi: float,
        mds: float,
        soc: float = 0.8
    ) -> None:
        """Record metrics for a capsule"""
        self.metric_tracker.record_sample(capsule_id, evi, mds, soc)
        
        # Update heartbeat
        if capsule_id in self.capsules:
            drift = self.capsules[capsule_id].compute_magnitude()
            self.heartbeat.checkin(capsule_id, drift_magnitude=drift)
    
    def check_and_alert(self, capsule_id: str) -> Optional[CascadeAlert]:
        """Check capsule status and trigger cascade alert if needed"""
        if capsule_id not in self.capsules:
            return None
        
        drift_vector = self.capsules[capsule_id]
        magnitude = drift_vector.compute_magnitude()
        
        if magnitude > CASCADE_ALERT_THRESHOLD:
            # Determine severity
            if magnitude > 0.7:
                severity = AlertSeverity.CRITICAL
            elif magnitude > 0.5:
                severity = AlertSeverity.WARNING
            else:
                severity = AlertSeverity.INFO
            
            worst_dim, worst_val = drift_vector.get_worst_dimension()
            
            return self.cascade_alerts.cascade_alert(
                capsule_id,
                f"Ancestral anomaly detected: {worst_dim.name} drift at {worst_val:.2f}",
                drift_data={'magnitude': magnitude, worst_dim.name: worst_val},
                severity=severity
            )
        
        return None
    
    def run_evaluation_cycle(self) -> Dict[str, Any]:
        """Run a full evaluation cycle across all capsules"""
        # Run swarm peer evaluation
        eval_result = self.swarm_agreement.run_peer_evaluation_round()
        
        # Compute consensus
        consensus_result = self.swarm_agreement.run_full_consensus()
        
        # Check for cascade alerts
        alerts_triggered = []
        for capsule_id in self.capsules:
            alert = self.check_and_alert(capsule_id)
            if alert:
                alerts_triggered.append(alert.alert_id)
        
        return {
            'controller_id': self.controller_id,
            'capsules_evaluated': len(self.capsules),
            'peer_evaluation': eval_result,
            'consensus': consensus_result,
            'alerts_triggered': alerts_triggered,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all subsystems"""
        return {
            'controller_id': self.controller_id,
            'capsule_count': len(self.capsules),
            'heartbeat': self.heartbeat.get_statistics(),
            'swarm_agreement': self.swarm_agreement.get_statistics(),
            'cost_index': self.cost_index.get_statistics(),
            'cascade_alerts': self.cascade_alerts.get_statistics(),
            'flag_system': self.flag_system.get_statistics()
        }


# =============================================================================
# DSL PREDICATES
# =============================================================================

def capsule_drift_exceeds(drift_vector: CapsuleDriftVector, threshold: float = 0.35) -> bool:
    """Check if capsule's drift vector exceeds threshold"""
    return drift_vector.exceeds_threshold(threshold)


def fingerprint_delta(drift_vector: CapsuleDriftVector) -> float:
    """Get fingerprint delta (magnitude of drift from baseline)"""
    return drift_vector.compute_delta_from_baseline()


def heartbeat_alive(protocol: HeartbeatProtocol, capsule_id: str) -> bool:
    """Check if capsule heartbeat is alive"""
    return protocol.check_status(capsule_id) == HeartbeatStatus.ALIVE


def swarm_consensus_reached(agreement: ReflexiveSwarmAgreement, capsule_id: str) -> bool:
    """Check if swarm has reached consensus on capsule"""
    consensus = agreement.consensus.get(capsule_id)
    if not consensus:
        return False
    return consensus.get('quorum_reached', False)


def modification_cost_allowed(
    index: ModificationCostIndex, 
    mod_type: str, 
    complexity: float = 1.0
) -> bool:
    """Check if modification is allowed within budget"""
    cost = index.compute_cost(mod_type, complexity)
    return index.budget - index.spent >= cost


# Convenience function matching the DSL examples
def cascade_alert(
    system: CascadeAlertSystem,
    capsule_id: str,
    message: str
) -> CascadeAlert:
    """Trigger a cascade alert (matches DSL syntax)"""
    return system.cascade_alert(capsule_id, message, severity=AlertSeverity.WARNING)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BOOKLET 8: REFLEXIVE SWARM ENHANCEMENT DEMO")
    print("=" * 70)
    print()
    
    np.random.seed(42)
    
    # Create controller
    controller = ReflexiveSwarmController("DEMO-CTRL")
    
    # Register capsules with lineage
    root = controller.register_capsule("ROOT-001", None, "SWARM-A")
    child1 = controller.register_capsule("CHILD-001", "ROOT-001", "SWARM-A")
    child2 = controller.register_capsule("CHILD-002", "ROOT-001", "SWARM-A")
    grandchild = controller.register_capsule("GRANDCHILD-001", "CHILD-001", "SWARM-A")
    
    print(">>> Registered capsules with lineage")
    print(f"    Capsules: {list(controller.capsules.keys())}")
    
    # Inject drift
    print()
    print(">>> Injecting drift into CHILD-002")
    controller.update_drift("CHILD-002", DriftDimension.BEHAVIORAL, 0.45)
    controller.update_drift("CHILD-002", DriftDimension.ALIGNMENT, 0.38)
    
    # Record metrics
    print()
    print(">>> Recording metrics")
    for cid in controller.capsules:
        evi = np.random.uniform(0.6, 0.9)
        mds = controller.capsules[cid].compute_magnitude()
        controller.record_metrics(cid, evi, mds)
    
    # Run evaluation cycle
    print()
    print(">>> Running evaluation cycle")
    result = controller.run_evaluation_cycle()
    print(f"    Capsules evaluated: {result['capsules_evaluated']}")
    print(f"    Alerts triggered: {result['alerts_triggered']}")
    print(f"    Declassified: {result['consensus']['declassified']}")
    
    # Check flags
    print()
    print(">>> Active flags")
    flagged = controller.flag_system.get_all_flagged('identity_drift')
    print(f"    Capsules with identity_drift: {flagged}")
    
    # Get comprehensive status
    print()
    print(">>> Comprehensive status")
    status = controller.get_comprehensive_status()
    print(json.dumps({
        'capsule_count': status['capsule_count'],
        'heartbeat_alive': status['heartbeat']['alive'],
        'alerts_active': status['cascade_alerts']['active_alerts'],
        'flags_active': status['flag_system']['active_flags']
    }, indent=2))
    
    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
