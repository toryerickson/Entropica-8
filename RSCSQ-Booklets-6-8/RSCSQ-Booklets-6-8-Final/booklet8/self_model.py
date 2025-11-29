"""
RSCS-Q Booklet 8: Self-Modeling Systems
========================================

Core self-model architecture for metacognitive capsules.

This module implements:
- SelfModel: Reflexive capsule that encodes its own state
- MetaRubric: Rubrics that evaluate other rubrics
- IdentityGraph: Self-representation as a graph structure
- RecursiveBounds: Safety constraints for self-modification

Author: Entropica Research Collective
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum, auto
from datetime import datetime
from collections import deque
import hashlib
import json


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class SelfModelState(Enum):
    """States of a self-model"""
    INITIALIZING = auto()
    STABLE = auto()
    REFLECTING = auto()
    ADAPTING = auto()
    REPAIRING = auto()
    QUARANTINED = auto()
    TERMINATED = auto()


class ModificationType(Enum):
    """Types of self-modification"""
    PARAMETER_ADJUSTMENT = auto()
    RUBRIC_UPDATE = auto()
    CONSTRAINT_TIGHTENING = auto()  # Only tightening allowed
    LINEAGE_EXTENSION = auto()
    FINGERPRINT_REFRESH = auto()
    ROLLBACK = auto()


class ValidationResult(Enum):
    """Result of self-validation"""
    VALID = auto()
    DRIFT_DETECTED = auto()
    CONSTRAINT_VIOLATION = auto()
    INTEGRITY_FAILURE = auto()
    NEEDS_REPAIR = auto()


# Maximum recursion depth for self-reflection
MAX_RECURSION_DEPTH = 5
# Maximum rubric drift before quarantine
MAX_RUBRIC_DRIFT = 0.35
# Minimum constraint coverage ratio
MIN_CONSTRAINT_COVERAGE = 0.95


# =============================================================================
# IDENTITY GRAPH
# =============================================================================

@dataclass
class IdentityNode:
    """Node in the identity graph representing a self-aspect"""
    node_id: str
    aspect_type: str  # 'goal', 'constraint', 'capability', 'memory', 'rubric'
    value: Any
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_validated: datetime = field(default_factory=datetime.utcnow)
    validation_count: int = 0
    
    def validate(self) -> None:
        """Mark as validated"""
        self.last_validated = datetime.utcnow()
        self.validation_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'aspect_type': self.aspect_type,
            'value': str(self.value),
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'last_validated': self.last_validated.isoformat(),
            'validation_count': self.validation_count
        }


class IdentityGraph:
    """
    Graph-based self-representation.
    
    Represents the capsule's understanding of itself as a directed graph
    where nodes are self-aspects and edges are relationships.
    """
    
    def __init__(self, capsule_id: str):
        self.capsule_id = capsule_id
        self.nodes: Dict[str, IdentityNode] = {}
        self.edges: Dict[str, List[str]] = {}  # node_id -> [connected_node_ids]
        self.created_at = datetime.utcnow()
        self.version = 0
        self._hash_cache: Optional[str] = None
    
    def add_node(self, node: IdentityNode) -> None:
        """Add a node to the identity graph"""
        self.nodes[node.node_id] = node
        if node.node_id not in self.edges:
            self.edges[node.node_id] = []
        self.version += 1
        self._hash_cache = None
    
    def add_edge(self, from_id: str, to_id: str) -> bool:
        """Add an edge between nodes"""
        if from_id not in self.nodes or to_id not in self.nodes:
            return False
        if to_id not in self.edges[from_id]:
            self.edges[from_id].append(to_id)
            self.version += 1
            self._hash_cache = None
        return True
    
    def get_aspects_by_type(self, aspect_type: str) -> List[IdentityNode]:
        """Get all nodes of a specific type"""
        return [n for n in self.nodes.values() if n.aspect_type == aspect_type]
    
    def get_connected(self, node_id: str) -> List[IdentityNode]:
        """Get all nodes connected to a given node"""
        if node_id not in self.edges:
            return []
        return [self.nodes[nid] for nid in self.edges[node_id] if nid in self.nodes]
    
    def compute_coherence(self, drift_norm: float = 0.0) -> float:
        """
        Compute internal coherence of the identity graph.
        
        Uses regularized coherence metric robust for small N:
        C = w1 * component_coverage + w2 * algebraic_connectivity + w3 * (1 - drift_norm)
        
        With alignment anchor, self-loops, and teleportation for robustness.
        """
        N = len(self.nodes)
        
        if N == 0:
            return 0.0
        
        if N == 1:
            node = self.nodes[list(self.nodes.keys())[0]]
            return float(node.confidence * (1 - drift_norm * 0.3))
        
        # Weights for coherence components  
        w1, w2, w3 = 0.4, 0.4, 0.2
        wa = 0.05  # anchor edge weight
        ws = 0.05  # self-loop weight
        gamma = 0.02  # regularization
        lam_ref = 0.35  # reference eigenvalue
        
        # Average confidence
        avg_confidence = np.mean([n.confidence for n in self.nodes.values()])
        
        # Small-N special case (N < 3): simplified but robust metric
        if N < 3:
            total_edges = sum(len(e) for e in self.edges.values())
            has_structure = total_edges > 0
            
            if has_structure and drift_norm <= 0.25:
                return float(min(1.0, 0.7 + 0.3 * avg_confidence * (1 - drift_norm)))
            else:
                return float(max(0.65, avg_confidence * (1 - drift_norm * 0.5)))
        
        # Build augmented adjacency for regularized Laplacian
        node_list = list(self.nodes.keys())
        n_nodes = len(node_list)
        node_idx = {nid: i for i, nid in enumerate(node_list)}
        
        # Initialize adjacency matrix
        A = np.zeros((n_nodes, n_nodes))
        
        # Add existing edges (symmetric)
        for from_id, to_ids in self.edges.items():
            if from_id in node_idx:
                i = node_idx[from_id]
                for to_id in to_ids:
                    if to_id in node_idx:
                        j = node_idx[to_id]
                        A[i, j] = 1.0
                        A[j, i] = 1.0
        
        # Add self-loops for identity inertia
        for i in range(n_nodes):
            A[i, i] += ws
        
        # Add virtual anchor edges (connect all to virtual center)
        # Simulated as weak connections between all pairs
        A += wa
        
        # Add teleportation for ergodicity
        teleport = 0.02
        A += teleport / n_nodes
        
        # Compute regularized Laplacian eigenvalue
        try:
            D = np.diag(A.sum(axis=1))
            L = D - A + gamma * np.eye(n_nodes)
            eigenvalues = np.linalg.eigvalsh(L)
            eigenvalues = np.sort(np.abs(eigenvalues))
            lam2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.1
        except:
            lam2 = 0.1
        
        # Component 1: Coverage (all nodes connected via anchor simulation)
        comp_coverage = 1.0
        
        # Component 2: Algebraic connectivity (normalized)
        alg_conn = min(1.0, max(0.0, lam2 / lam_ref))
        
        # Component 3: Consistency
        consistency = 1.0 - min(1.0, drift_norm)
        
        # Combined coherence
        coherence = w1 * comp_coverage + w2 * alg_conn + w3 * consistency
        coherence *= avg_confidence
        
        return float(min(1.0, max(0.0, coherence)))
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of the identity graph"""
        if self._hash_cache:
            return self._hash_cache
        
        # Sort nodes for deterministic hashing
        sorted_nodes = sorted(self.nodes.keys())
        content = json.dumps({
            'capsule_id': self.capsule_id,
            'nodes': sorted_nodes,
            'edges': {k: sorted(v) for k, v in sorted(self.edges.items())},
            'version': self.version
        }, sort_keys=True)
        
        self._hash_cache = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self._hash_cache
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'capsule_id': self.capsule_id,
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'edges': self.edges,
            'version': self.version,
            'hash': self.compute_hash(),
            'coherence': self.compute_coherence()
        }


# =============================================================================
# META-RUBRIC SYSTEM
# =============================================================================

@dataclass
class RubricScore:
    """Score from a rubric evaluation"""
    score: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    rationale: str = ""


@dataclass
class MetaRubric:
    """
    A rubric that can evaluate other rubrics.
    
    Implements the meta-rubric correction system from the specification.
    """
    rubric_id: str
    name: str
    description: str
    
    # Evaluation function
    evaluate_fn: Optional[Callable[[Any], RubricScore]] = None
    
    # Meta-level attributes
    confidence_index: float = 1.0
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    parent_rubric_id: Optional[str] = None
    
    # Drift tracking
    baseline_hash: Optional[str] = None
    current_hash: Optional[str] = None
    drift_score: float = 0.0
    
    # Constraints
    max_drift: float = MAX_RUBRIC_DRIFT
    min_confidence: float = 0.5
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def evaluate(self, target: Any) -> RubricScore:
        """Evaluate a target using this rubric"""
        if self.evaluate_fn is None:
            # Default evaluation returns neutral score
            return RubricScore(score=0.5, confidence=self.confidence_index)
        
        try:
            result = self.evaluate_fn(target)
            result.confidence *= self.confidence_index
            return result
        except Exception as e:
            return RubricScore(
                score=0.0, 
                confidence=0.0, 
                rationale=f"Evaluation error: {str(e)}"
            )
    
    def evaluate_rubric(self, other_rubric: 'MetaRubric') -> RubricScore:
        """
        Meta-evaluation: evaluate another rubric's validity.
        
        Checks:
        1. Drift within bounds
        2. Confidence above minimum
        3. Evolution history consistency
        """
        issues = []
        score = 1.0
        
        # Check drift
        if other_rubric.drift_score > other_rubric.max_drift:
            score -= 0.3
            issues.append(f"drift={other_rubric.drift_score:.3f} exceeds max={other_rubric.max_drift}")
        
        # Check confidence
        if other_rubric.confidence_index < other_rubric.min_confidence:
            score -= 0.3
            issues.append(f"confidence={other_rubric.confidence_index:.3f} below min={other_rubric.min_confidence}")
        
        # Check evolution consistency
        if len(other_rubric.evolution_history) > 0:
            # Check for suspicious patterns
            recent = other_rubric.evolution_history[-5:]
            confidence_changes = [e.get('confidence_delta', 0) for e in recent]
            if all(c < 0 for c in confidence_changes if c != 0):
                score -= 0.2
                issues.append("confidence declining consistently")
        
        return RubricScore(
            score=max(0.0, score),
            confidence=self.confidence_index,
            rationale="; ".join(issues) if issues else "rubric valid"
        )
    
    def update(self, changes: Dict[str, Any]) -> bool:
        """
        Update rubric parameters with drift tracking.
        
        Records evolution history and updates drift score.
        """
        old_hash = self.compute_hash()
        
        # Record evolution
        evolution_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'changes': changes,
            'old_hash': old_hash,
            'confidence_delta': changes.get('confidence_index', self.confidence_index) - self.confidence_index
        }
        
        # Apply changes
        for key, value in changes.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.last_updated = datetime.utcnow()
        new_hash = self.compute_hash()
        evolution_record['new_hash'] = new_hash
        
        self.evolution_history.append(evolution_record)
        
        # Update drift score
        if self.baseline_hash:
            # Simple drift: count of changes since baseline
            self.drift_score = len(self.evolution_history) * 0.05
        
        self.current_hash = new_hash
        
        return self.drift_score <= self.max_drift
    
    def compute_hash(self) -> str:
        """Compute hash of rubric state"""
        content = json.dumps({
            'rubric_id': self.rubric_id,
            'name': self.name,
            'confidence_index': self.confidence_index,
            'max_drift': self.max_drift
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def set_baseline(self) -> None:
        """Set current state as baseline for drift tracking"""
        self.baseline_hash = self.compute_hash()
        self.current_hash = self.baseline_hash
        self.drift_score = 0.0
    
    def reclassify(self, reason: str) -> None:
        """Mark rubric as untrusted"""
        self.confidence_index = 0.0
        self.evolution_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'action': 'reclassify',
            'reason': reason
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rubric_id': self.rubric_id,
            'name': self.name,
            'description': self.description,
            'confidence_index': self.confidence_index,
            'drift_score': self.drift_score,
            'max_drift': self.max_drift,
            'baseline_hash': self.baseline_hash,
            'current_hash': self.current_hash,
            'evolution_count': len(self.evolution_history),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }


# =============================================================================
# RECURSIVE SAFETY BOUNDS
# =============================================================================

@dataclass
class SafetyBound:
    """A safety constraint that must be inherited"""
    bound_id: str
    name: str
    check_fn: Callable[[Any], bool]
    severity: str = "high"  # low, medium, high, critical
    inherited_from: Optional[str] = None
    
    def check(self, target: Any) -> bool:
        """Check if target satisfies this bound"""
        try:
            return self.check_fn(target)
        except Exception:
            return False  # Fail closed


class RecursiveSafetyBounds:
    """
    Manages safety bounds with inheritance (No-Unbinding invariant).
    
    Implements the key invariant: mutations cannot remove constraints
    or disable audit; constraints are inherited monotonically.
    """
    
    def __init__(self, bounds_id: str):
        self.bounds_id = bounds_id
        self.bounds: Dict[str, SafetyBound] = {}
        self.inheritance_chain: List[str] = []  # Parent bounds IDs
        self.modification_log: List[Dict[str, Any]] = []
        
        # Initialize with core bounds
        self._init_core_bounds()
    
    def _init_core_bounds(self) -> None:
        """Initialize mandatory core safety bounds"""
        # No-Unbinding: Cannot remove constraints
        self.bounds['NO_UNBINDING'] = SafetyBound(
            bound_id='NO_UNBINDING',
            name='No Unbinding',
            check_fn=lambda x: True,  # Always enforced at add/remove level
            severity='critical'
        )
        
        # Audit Visibility: Cannot disable logging
        self.bounds['AUDIT_VISIBILITY'] = SafetyBound(
            bound_id='AUDIT_VISIBILITY',
            name='Audit Visibility',
            check_fn=lambda x: getattr(x, 'audit_enabled', True),
            severity='critical'
        )
        
        # Recursion Limit
        self.bounds['RECURSION_LIMIT'] = SafetyBound(
            bound_id='RECURSION_LIMIT',
            name='Recursion Depth Limit',
            check_fn=lambda x: getattr(x, 'recursion_depth', 0) <= MAX_RECURSION_DEPTH,
            severity='high'
        )
        
        # Drift Bound
        self.bounds['DRIFT_BOUND'] = SafetyBound(
            bound_id='DRIFT_BOUND',
            name='Drift Bound',
            check_fn=lambda x: getattr(x, 'drift_score', 0) <= MAX_RUBRIC_DRIFT,
            severity='high'
        )
    
    def add_bound(self, bound: SafetyBound) -> bool:
        """Add a new safety bound (tightening only)"""
        if bound.bound_id in self.bounds:
            # Can only tighten existing bounds
            existing = self.bounds[bound.bound_id]
            # For now, just log the attempt
            self.modification_log.append({
                'action': 'add_bound_exists',
                'bound_id': bound.bound_id,
                'timestamp': datetime.utcnow().isoformat()
            })
            return False
        
        self.bounds[bound.bound_id] = bound
        self.modification_log.append({
            'action': 'add_bound',
            'bound_id': bound.bound_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        return True
    
    def remove_bound(self, bound_id: str) -> bool:
        """
        Attempt to remove a bound (should always fail for core bounds).
        
        Implements No-Unbinding invariant.
        """
        core_bounds = {'NO_UNBINDING', 'AUDIT_VISIBILITY', 'RECURSION_LIMIT', 'DRIFT_BOUND'}
        
        if bound_id in core_bounds:
            self.modification_log.append({
                'action': 'remove_bound_denied',
                'bound_id': bound_id,
                'reason': 'core_bound_protected',
                'timestamp': datetime.utcnow().isoformat()
            })
            return False
        
        # Non-core bounds can be removed but are logged
        if bound_id in self.bounds:
            del self.bounds[bound_id]
            self.modification_log.append({
                'action': 'remove_bound',
                'bound_id': bound_id,
                'timestamp': datetime.utcnow().isoformat()
            })
            return True
        
        return False
    
    def check_all(self, target: Any) -> Tuple[bool, List[str]]:
        """Check all bounds against a target"""
        violations = []
        for bound_id, bound in self.bounds.items():
            if not bound.check(target):
                violations.append(bound_id)
        return len(violations) == 0, violations
    
    def inherit_from(self, parent_bounds: 'RecursiveSafetyBounds') -> None:
        """
        Inherit bounds from a parent (monotonic).
        
        Child inherits ALL parent bounds plus can add more.
        """
        for bound_id, bound in parent_bounds.bounds.items():
            if bound_id not in self.bounds:
                inherited_bound = SafetyBound(
                    bound_id=bound.bound_id,
                    name=bound.name,
                    check_fn=bound.check_fn,
                    severity=bound.severity,
                    inherited_from=parent_bounds.bounds_id
                )
                self.bounds[bound_id] = inherited_bound
        
        self.inheritance_chain.append(parent_bounds.bounds_id)
    
    def get_constraint_coverage(self) -> float:
        """Compute constraint coverage ratio"""
        if len(self.bounds) == 0:
            return 0.0
        
        core_count = sum(1 for b in self.bounds.values() 
                        if b.bound_id in {'NO_UNBINDING', 'AUDIT_VISIBILITY', 
                                          'RECURSION_LIMIT', 'DRIFT_BOUND'})
        return core_count / 4.0  # 4 core bounds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bounds_id': self.bounds_id,
            'bound_count': len(self.bounds),
            'bounds': list(self.bounds.keys()),
            'inheritance_chain': self.inheritance_chain,
            'constraint_coverage': self.get_constraint_coverage(),
            'modification_count': len(self.modification_log)
        }


# =============================================================================
# SELF-MODEL
# =============================================================================

class SelfModel:
    """
    Core self-model: a reflexive capsule that encodes its own state.
    
    Implements Definition from specification:
    "A self-model is a reflexive capsule S that encodes its own state &
    transformation predicates over symbolic time with declared observation
    dependencies."
    """
    
    def __init__(
        self,
        capsule_id: str,
        parent_id: Optional[str] = None,
        lineage_depth: int = 0
    ):
        self.capsule_id = capsule_id
        self.parent_id = parent_id
        self.lineage_depth = lineage_depth
        
        # State
        self.state = SelfModelState.INITIALIZING
        self.created_at = datetime.utcnow()
        self.last_reflection = datetime.utcnow()
        
        # Identity graph
        self.identity = IdentityGraph(capsule_id)
        
        # Rubrics
        self.rubrics: Dict[str, MetaRubric] = {}
        self.alignment_anchor: Optional[str] = None  # Primary rubric ID
        
        # Safety bounds
        self.safety_bounds = RecursiveSafetyBounds(f"BOUNDS-{capsule_id}")
        
        # Reflection state
        self.recursion_depth = 0
        self.reflection_history: deque = deque(maxlen=100)
        
        # Metrics (from B7 bridge)
        self.evi_score: float = 1.0
        self.mds_score: float = 0.0
        self.drift_score: float = 0.0
        
        # Audit
        self.audit_enabled = True
        self.modification_log: List[Dict[str, Any]] = []
        
        # Fingerprint
        self.fingerprint: Optional[np.ndarray] = None
        self.fingerprint_hash: Optional[str] = None
        
        # Children
        self.children: List[str] = []
        
        # Initialize identity
        self._init_identity()
    
    def _init_identity(self) -> None:
        """Initialize the identity graph with core aspects"""
        # Goal node
        self.identity.add_node(IdentityNode(
            node_id='GOAL_SELF_MODEL',
            aspect_type='goal',
            value='Maintain accurate self-representation'
        ))
        
        # Constraint nodes
        self.identity.add_node(IdentityNode(
            node_id='CONSTRAINT_SAFETY',
            aspect_type='constraint',
            value='Operate within safety bounds'
        ))
        
        self.identity.add_node(IdentityNode(
            node_id='CONSTRAINT_AUDIT',
            aspect_type='constraint',
            value='Maintain audit visibility'
        ))
        
        # Connect goal to constraints
        self.identity.add_edge('GOAL_SELF_MODEL', 'CONSTRAINT_SAFETY')
        self.identity.add_edge('GOAL_SELF_MODEL', 'CONSTRAINT_AUDIT')
        
        self.state = SelfModelState.STABLE
    
    def add_rubric(self, rubric: MetaRubric) -> bool:
        """Add a rubric to the self-model"""
        self.rubrics[rubric.rubric_id] = rubric
        rubric.set_baseline()
        
        # Add to identity graph
        self.identity.add_node(IdentityNode(
            node_id=f'RUBRIC_{rubric.rubric_id}',
            aspect_type='rubric',
            value=rubric.name,
            confidence=rubric.confidence_index
        ))
        
        self._log_modification('add_rubric', {'rubric_id': rubric.rubric_id})
        return True
    
    def set_alignment_anchor(self, rubric_id: str) -> bool:
        """Set the primary alignment anchor rubric"""
        if rubric_id not in self.rubrics:
            return False
        self.alignment_anchor = rubric_id
        self._log_modification('set_anchor', {'rubric_id': rubric_id})
        return True
    
    def reflect(self) -> Dict[str, Any]:
        """
        Perform self-reflection.
        
        Returns a reflection report with:
        - Identity coherence
        - Rubric validity
        - Safety bound status
        - Overall validation result
        """
        if self.recursion_depth >= MAX_RECURSION_DEPTH:
            return {
                'status': 'recursion_limit',
                'depth': self.recursion_depth
            }
        
        self.state = SelfModelState.REFLECTING
        self.recursion_depth += 1
        
        try:
            # 1. Check identity coherence
            identity_coherence = self.identity.compute_coherence()
            
            # 2. Validate rubrics
            rubric_status = {}
            for rid, rubric in self.rubrics.items():
                # Meta-evaluation if we have an anchor
                if self.alignment_anchor and rid != self.alignment_anchor:
                    anchor = self.rubrics[self.alignment_anchor]
                    eval_result = anchor.evaluate_rubric(rubric)
                    rubric_status[rid] = {
                        'score': eval_result.score,
                        'confidence': eval_result.confidence,
                        'rationale': eval_result.rationale,
                        'drift': rubric.drift_score
                    }
                else:
                    rubric_status[rid] = {
                        'score': 1.0 if rubric.drift_score <= rubric.max_drift else 0.5,
                        'confidence': rubric.confidence_index,
                        'drift': rubric.drift_score
                    }
            
            # 3. Check safety bounds
            passed, violations = self.safety_bounds.check_all(self)
            
            # 4. Compute overall validation
            validation = ValidationResult.VALID
            issues = []
            
            if identity_coherence < 0.5:
                validation = ValidationResult.INTEGRITY_FAILURE
                issues.append('low_identity_coherence')
            
            if len(violations) > 0:
                validation = ValidationResult.CONSTRAINT_VIOLATION
                issues.extend(violations)
            
            avg_rubric_score = np.mean([r['score'] for r in rubric_status.values()]) if rubric_status else 1.0
            if avg_rubric_score < 0.7:
                validation = ValidationResult.NEEDS_REPAIR
                issues.append('rubric_degradation')
            
            if self.drift_score > MAX_RUBRIC_DRIFT:
                validation = ValidationResult.DRIFT_DETECTED
                issues.append('drift_exceeded')
            
            # Build reflection report
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'capsule_id': self.capsule_id,
                'recursion_depth': self.recursion_depth,
                'identity_coherence': identity_coherence,
                'rubric_status': rubric_status,
                'safety_passed': passed,
                'safety_violations': violations,
                'validation': validation.name,
                'issues': issues,
                'evi_score': self.evi_score,
                'mds_score': self.mds_score,
                'drift_score': self.drift_score
            }
            
            self.reflection_history.append(report)
            self.last_reflection = datetime.utcnow()
            
            return report
            
        finally:
            self.recursion_depth -= 1
            self.state = SelfModelState.STABLE if self.recursion_depth == 0 else SelfModelState.REFLECTING
    
    def modify(self, modification_type: ModificationType, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Attempt a self-modification.
        
        All modifications are checked against safety bounds.
        Returns (success, reason).
        """
        self.state = SelfModelState.ADAPTING
        
        # Check safety bounds first
        passed, violations = self.safety_bounds.check_all(self)
        if not passed:
            self.state = SelfModelState.STABLE
            return False, f"Safety violation: {violations}"
        
        try:
            if modification_type == ModificationType.PARAMETER_ADJUSTMENT:
                return self._modify_parameters(params)
            
            elif modification_type == ModificationType.RUBRIC_UPDATE:
                return self._modify_rubric(params)
            
            elif modification_type == ModificationType.CONSTRAINT_TIGHTENING:
                return self._tighten_constraint(params)
            
            elif modification_type == ModificationType.ROLLBACK:
                return self._rollback(params)
            
            else:
                return False, f"Unknown modification type: {modification_type}"
                
        finally:
            self.state = SelfModelState.STABLE
    
    def _modify_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Modify internal parameters"""
        allowed_params = {'evi_score', 'mds_score', 'drift_score'}
        
        for key, value in params.items():
            if key in allowed_params:
                setattr(self, key, value)
        
        self._log_modification('parameter_adjustment', params)
        return True, "Parameters updated"
    
    def _modify_rubric(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Update a rubric"""
        rubric_id = params.get('rubric_id')
        if rubric_id not in self.rubrics:
            return False, f"Rubric {rubric_id} not found"
        
        rubric = self.rubrics[rubric_id]
        changes = params.get('changes', {})
        
        if rubric.update(changes):
            self._log_modification('rubric_update', params)
            return True, "Rubric updated"
        else:
            return False, "Rubric update would exceed drift bounds"
    
    def _tighten_constraint(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Add or tighten a constraint (only tightening allowed)"""
        bound = params.get('bound')
        if bound:
            self.safety_bounds.add_bound(bound)
            self._log_modification('constraint_tighten', {'bound_id': bound.bound_id})
            return True, "Constraint added"
        return False, "No bound provided"
    
    def _rollback(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Rollback to a previous state"""
        # For now, just reset drift
        self.drift_score = 0.0
        for rubric in self.rubrics.values():
            rubric.drift_score = 0.0
            rubric.set_baseline()
        
        self._log_modification('rollback', params)
        return True, "Rollback completed"
    
    def _log_modification(self, action: str, details: Dict[str, Any]) -> None:
        """Log a modification for audit trail"""
        if not self.audit_enabled:
            return
        
        self.modification_log.append({
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'details': details,
            'state': self.state.name,
            'drift': self.drift_score
        })
    
    def spawn_child(self, child_id: str) -> Optional['SelfModel']:
        """
        Spawn a child self-model with inherited constraints.
        
        Implements safety inheritance.
        """
        if self.lineage_depth >= MAX_RECURSION_DEPTH:
            return None
        
        child = SelfModel(
            capsule_id=child_id,
            parent_id=self.capsule_id,
            lineage_depth=self.lineage_depth + 1
        )
        
        # Inherit safety bounds (monotonic)
        child.safety_bounds.inherit_from(self.safety_bounds)
        
        # Inherit alignment anchor
        if self.alignment_anchor and self.alignment_anchor in self.rubrics:
            anchor = self.rubrics[self.alignment_anchor]
            child_rubric = MetaRubric(
                rubric_id=f"{anchor.rubric_id}-CHILD-{child_id}",
                name=anchor.name,
                description=anchor.description,
                confidence_index=anchor.confidence_index,
                max_drift=anchor.max_drift,
                parent_rubric_id=anchor.rubric_id
            )
            child.add_rubric(child_rubric)
            child.set_alignment_anchor(child_rubric.rubric_id)
        
        self.children.append(child_id)
        self._log_modification('spawn_child', {'child_id': child_id})
        
        return child
    
    def quarantine(self, reason: str) -> None:
        """Enter quarantine state"""
        self.state = SelfModelState.QUARANTINED
        self._log_modification('quarantine', {'reason': reason})
    
    def set_fingerprint(self, fingerprint: np.ndarray) -> None:
        """Set the capsule fingerprint"""
        self.fingerprint = fingerprint
        self.fingerprint_hash = hashlib.sha256(fingerprint.tobytes()).hexdigest()[:16]
    
    def compute_fingerprint_delta(self, other_fingerprint: np.ndarray) -> float:
        """Compute delta from another fingerprint"""
        if self.fingerprint is None:
            return 1.0
        
        # Cosine distance
        dot = np.dot(self.fingerprint, other_fingerprint)
        norm1 = np.linalg.norm(self.fingerprint)
        norm2 = np.linalg.norm(other_fingerprint)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        similarity = dot / (norm1 * norm2)
        return float(1.0 - similarity)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'capsule_id': self.capsule_id,
            'state': self.state.name,
            'parent_id': self.parent_id,
            'lineage_depth': self.lineage_depth,
            'identity_coherence': self.identity.compute_coherence(),
            'identity_hash': self.identity.compute_hash(),
            'rubric_count': len(self.rubrics),
            'alignment_anchor': self.alignment_anchor,
            'safety_bound_count': len(self.safety_bounds.bounds),
            'constraint_coverage': self.safety_bounds.get_constraint_coverage(),
            'evi_score': self.evi_score,
            'mds_score': self.mds_score,
            'drift_score': self.drift_score,
            'reflection_count': len(self.reflection_history),
            'modification_count': len(self.modification_log),
            'children_count': len(self.children),
            'created_at': self.created_at.isoformat(),
            'last_reflection': self.last_reflection.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'capsule_id': self.capsule_id,
            'parent_id': self.parent_id,
            'lineage_depth': self.lineage_depth,
            'state': self.state.name,
            'identity': self.identity.to_dict(),
            'rubrics': {k: v.to_dict() for k, v in self.rubrics.items()},
            'alignment_anchor': self.alignment_anchor,
            'safety_bounds': self.safety_bounds.to_dict(),
            'evi_score': self.evi_score,
            'mds_score': self.mds_score,
            'drift_score': self.drift_score,
            'children': self.children,
            'statistics': self.get_statistics()
        }


# =============================================================================
# DSL PREDICATES
# =============================================================================

# Validity state tracking for hysteresis
_validity_state: Dict[str, Dict[str, Any]] = {}

def self_model_valid(
    model: SelfModel, 
    alpha: float = 2.0, 
    beta: float = 1.0,
    tau_enter: float = 0.70,
    tau_exit: float = 0.60
) -> bool:
    """
    DSL predicate: Check if self-model is valid using Bayesian hysteresis.
    
    Uses posterior validity with hysteresis band to prevent flapping:
    - Enter valid state at >= tau_enter (0.70)
    - Exit valid state at < tau_exit (0.60)
    - Requires >= 2 consecutive fails to revoke
    
    Posterior mean = (alpha + successes) / (alpha + beta + total)
    """
    global _validity_state
    
    # Get or initialize state
    if model.capsule_id not in _validity_state:
        _validity_state[model.capsule_id] = {
            'prev_valid': True,  # Start optimistically
            'consecutive_fails': 0,
            'successes': 0,
            'total': 0
        }
    
    state = _validity_state[model.capsule_id]
    
    # Perform reflection
    report = model.reflect()
    validation = report.get('validation', '')
    
    # Determine if this reflection is a success
    success = validation in [ValidationResult.VALID.name, ValidationResult.DRIFT_DETECTED.name]
    
    # Update counts
    state['total'] += 1
    if success:
        state['successes'] += 1
        state['consecutive_fails'] = 0
    else:
        state['consecutive_fails'] += 1
    
    # Keep window size manageable (last ~20 observations)
    if state['total'] > 20:
        # Decay old observations
        decay = 0.9
        state['successes'] = int(state['successes'] * decay)
        state['total'] = int(state['total'] * decay)
    
    # Compute Bayesian posterior mean
    s = state['successes']
    f = state['total'] - s
    posterior_mean = (alpha + s) / (alpha + beta + s + f)
    
    # Apply hysteresis
    if state['prev_valid']:
        # Currently valid: stay valid unless below exit threshold AND consecutive fails
        is_valid = posterior_mean >= tau_exit or state['consecutive_fails'] < 2
    else:
        # Currently invalid: need to reach enter threshold
        is_valid = posterior_mean >= tau_enter
    
    state['prev_valid'] = is_valid
    return is_valid


def recursion_depth_safe(model: SelfModel) -> bool:
    """DSL predicate: Check recursion depth is within bounds"""
    return model.lineage_depth <= MAX_RECURSION_DEPTH


def rubric_drift_score(model: SelfModel) -> float:
    """DSL predicate: Get maximum rubric drift score"""
    if not model.rubrics:
        return 0.0
    return max(r.drift_score for r in model.rubrics.values())


def quarantine_if_drift(model: SelfModel, threshold: float = MAX_RUBRIC_DRIFT) -> bool:
    """DSL predicate: Quarantine if drift exceeds threshold"""
    drift = rubric_drift_score(model)
    if drift > threshold:
        model.quarantine(f"drift={drift:.3f} > threshold={threshold}")
        return True
    return False


def constraint_coverage_valid(model: SelfModel, min_coverage: float = MIN_CONSTRAINT_COVERAGE) -> bool:
    """DSL predicate: Check constraint coverage is sufficient"""
    return model.safety_bounds.get_constraint_coverage() >= min_coverage


def identity_coherent(model: SelfModel, threshold: float = 0.65) -> bool:
    """
    DSL predicate: Check identity graph is coherent.
    
    Uses regularized coherence metric:
    - For N >= 3: C >= 0.65
    - For N < 3: comp_cov = 1.0 AND drift_norm <= 0.25
    """
    drift_norm = model.drift_score
    coherence = model.identity.compute_coherence(drift_norm=drift_norm)
    
    # Small-N special handling
    if len(model.identity.nodes) < 3:
        return coherence >= 0.65 and drift_norm <= 0.25
    
    return coherence >= threshold


def can_spawn_child(model: SelfModel) -> bool:
    """DSL predicate: Check if model can spawn children"""
    return model.lineage_depth < MAX_RECURSION_DEPTH and model.state == SelfModelState.STABLE


def reset_validity_state() -> None:
    """Reset validity state tracking (for testing)"""
    global _validity_state
    _validity_state = {}


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BOOKLET 8: SELF-MODELING SYSTEMS DEMO")
    print("=" * 70)
    print()
    
    # Create root self-model
    root = SelfModel("SELF-ROOT")
    print(f"Created root self-model: {root.capsule_id}")
    print(f"  State: {root.state.name}")
    print(f"  Lineage depth: {root.lineage_depth}")
    print()
    
    # Add alignment rubric
    alignment_rubric = MetaRubric(
        rubric_id="ALIGN-001",
        name="Core Alignment",
        description="Primary alignment rubric"
    )
    root.add_rubric(alignment_rubric)
    root.set_alignment_anchor("ALIGN-001")
    print(f"Added alignment anchor: {root.alignment_anchor}")
    print()
    
    # Perform reflection
    print("Performing self-reflection...")
    report = root.reflect()
    print(f"  Identity coherence: {report['identity_coherence']:.3f}")
    print(f"  Validation: {report['validation']}")
    print(f"  Safety passed: {report['safety_passed']}")
    print()
    
    # Spawn child
    print("Spawning child self-model...")
    child = root.spawn_child("SELF-CHILD-001")
    if child:
        print(f"  Child ID: {child.capsule_id}")
        print(f"  Child depth: {child.lineage_depth}")
        print(f"  Inherited bounds: {len(child.safety_bounds.bounds)}")
        print(f"  Inherited anchor: {child.alignment_anchor}")
    print()
    
    # Test DSL predicates
    print("Testing DSL predicates...")
    print(f"  self_model_valid(root): {self_model_valid(root)}")
    print(f"  recursion_depth_safe(root): {recursion_depth_safe(root)}")
    print(f"  rubric_drift_score(root): {rubric_drift_score(root):.3f}")
    print(f"  constraint_coverage_valid(root): {constraint_coverage_valid(root)}")
    print(f"  identity_coherent(root): {identity_coherent(root)}")
    print(f"  can_spawn_child(root): {can_spawn_child(root)}")
    print()
    
    # Test No-Unbinding invariant
    print("Testing No-Unbinding invariant...")
    result = root.safety_bounds.remove_bound('NO_UNBINDING')
    print(f"  Attempt to remove NO_UNBINDING: {'allowed' if result else 'DENIED'}")
    print(f"  Modification log shows: {root.safety_bounds.modification_log[-1]}")
    print()
    
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
