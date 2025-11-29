"""
RSCS-Q Booklet 8: Self-Modeling Systems
========================================

This package provides the metacognitive architecture for RSCS-Q,
enabling capsules to model, reflect on, and safely modify themselves.

Key Components:
- SelfModel: Reflexive capsule with identity graph
- MetaRubric: Rubrics that evaluate other rubrics
- RubricRepairEngine: Self-healing rubric system
- ReflexLog: Append-only audit trail
- ObserverMesh: Quorum-based observation system
- B8Visualizer: Lineage and drift visualization
- DriftCascadeSimulation: Case study framework

Author: Entropica Research Collective
Version: 1.0
"""

# Self-Model System
from .self_model import (
    # Core classes
    SelfModel,
    IdentityGraph,
    IdentityNode,
    MetaRubric,
    RubricScore,
    RecursiveSafetyBounds,
    SafetyBound,
    # Enums
    SelfModelState,
    ModificationType,
    ValidationResult,
    # Constants
    MAX_RECURSION_DEPTH,
    MAX_RUBRIC_DRIFT,
    MIN_CONSTRAINT_COVERAGE,
    # DSL Predicates
    self_model_valid,
    recursion_depth_safe,
    rubric_drift_score,
    quarantine_if_drift,
    constraint_coverage_valid,
    identity_coherent,
    can_spawn_child,
    reset_validity_state,
)

# Rubric Repair Engine
from .rubric_repair import (
    # Core classes
    RubricRepairEngine,
    RubricDiagnostic,
    DiagnosticReport,
    RepairResult,
    RepairStrategy,
    ResetBaselineStrategy,
    RestoreConfidenceStrategy,
    PruneEvolutionStrategy,
    QuarantineStrategy,
    # Enums
    DiagnosticResult,
    RepairAction,
    RepairOutcome,
    # DSL Predicates
    rubric_needs_repair,
    auto_repair_rubric,
    rubric_trajectory,
    model_health_ratio,
)

# ReflexLog and Observer System
from .reflex_log import (
    # Core classes
    ReflexLog,
    ReflexLogEvent,
    ObserverMesh,
    Observer,
    RuntimeMetrics,
    # Enums
    EventType,
    ObserverPhase,
    CollapseValidity,
    # DSL Predicates
    reflex_log_valid,
    quorum_reached,
    metrics_gates_passed,
    collapse_valid,
)

# Visual Toolkit
from .visual_toolkit import (
    LineageVisualizer,
    LineageNode,
    DriftHeatmap,
    RecoveryTimeline,
    RecoveryEvent,
    B8Visualizer,
)

# Case Study
from .case_study import (
    DriftCascadeSimulation,
    CaseStudyConfig,
    CaseStudyResults,
)

# Reflexive Swarm Enhancements
from .reflexive_swarm import (
    # Core classes
    CapsuleDriftVector,
    HeartbeatProtocol,
    HeartbeatRecord,
    ReflexiveSwarmAgreement,
    DriftVote,
    ModificationCostIndex,
    ModificationCost,
    CascadeAlertSystem,
    CascadeAlert,
    MetricTimeSeriesTracker,
    MetricSample,
    CapsuleFlagSystem,
    CapsuleFlag,
    ReflexiveSwarmController,
    # Enums
    HeartbeatStatus,
    AlertSeverity,
    DriftDimension,
    # DSL Predicates
    capsule_drift_exceeds,
    fingerprint_delta,
    heartbeat_alive,
    swarm_consensus_reached,
    modification_cost_allowed,
    cascade_alert,
)

# Simulation Harness
from .simulation_harness import (
    B8SimulationHarness,
    SimulationConfig,
    SyntheticModelGenerator,
    B8TestCases,
)

__version__ = "1.0.0"
__author__ = "Entropica Research Collective"

__all__ = [
    # Self-Model
    "SelfModel",
    "IdentityGraph",
    "IdentityNode",
    "MetaRubric",
    "RubricScore",
    "RecursiveSafetyBounds",
    "SafetyBound",
    "SelfModelState",
    "ModificationType",
    "ValidationResult",
    "MAX_RECURSION_DEPTH",
    "MAX_RUBRIC_DRIFT",
    "MIN_CONSTRAINT_COVERAGE",
    "self_model_valid",
    "recursion_depth_safe",
    "rubric_drift_score",
    "quarantine_if_drift",
    "constraint_coverage_valid",
    "identity_coherent",
    "can_spawn_child",
    "reset_validity_state",
    # Rubric Repair
    "RubricRepairEngine",
    "RubricDiagnostic",
    "DiagnosticReport",
    "RepairResult",
    "RepairStrategy",
    "ResetBaselineStrategy",
    "RestoreConfidenceStrategy",
    "PruneEvolutionStrategy",
    "QuarantineStrategy",
    "DiagnosticResult",
    "RepairAction",
    "RepairOutcome",
    "rubric_needs_repair",
    "auto_repair_rubric",
    "rubric_trajectory",
    "model_health_ratio",
    # ReflexLog
    "ReflexLog",
    "ReflexLogEvent",
    "ObserverMesh",
    "Observer",
    "RuntimeMetrics",
    "EventType",
    "ObserverPhase",
    "CollapseValidity",
    "reflex_log_valid",
    "quorum_reached",
    "metrics_gates_passed",
    "collapse_valid",
    # Visual Toolkit
    "LineageVisualizer",
    "LineageNode",
    "DriftHeatmap",
    "RecoveryTimeline",
    "RecoveryEvent",
    "B8Visualizer",
    # Case Study
    "DriftCascadeSimulation",
    "CaseStudyConfig",
    "CaseStudyResults",
    # Reflexive Swarm
    "CapsuleDriftVector",
    "HeartbeatProtocol",
    "HeartbeatRecord",
    "ReflexiveSwarmAgreement",
    "DriftVote",
    "ModificationCostIndex",
    "ModificationCost",
    "CascadeAlertSystem",
    "CascadeAlert",
    "MetricTimeSeriesTracker",
    "MetricSample",
    "CapsuleFlagSystem",
    "CapsuleFlag",
    "ReflexiveSwarmController",
    "HeartbeatStatus",
    "AlertSeverity",
    "DriftDimension",
    "capsule_drift_exceeds",
    "fingerprint_delta",
    "heartbeat_alive",
    "swarm_consensus_reached",
    "modification_cost_allowed",
    "cascade_alert",
    # Simulation
    "B8SimulationHarness",
    "SimulationConfig",
    "SyntheticModelGenerator",
    "B8TestCases",
]
