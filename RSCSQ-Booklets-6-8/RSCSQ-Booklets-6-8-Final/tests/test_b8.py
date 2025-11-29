"""
RSCS-Q Booklet 8: Test Suite
============================

Comprehensive tests for:
- Self-Model System
- Rubric Repair Engine
- ReflexLog and Observer System
- G1-G8 Acceptance Criteria

Author: Entropica Research Collective
Version: 1.0
"""

import unittest
import sys
import os
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from self_model import (
    SelfModel, IdentityGraph, IdentityNode, MetaRubric, RubricScore,
    RecursiveSafetyBounds, SafetyBound,
    SelfModelState, ModificationType, ValidationResult,
    MAX_RECURSION_DEPTH, MAX_RUBRIC_DRIFT,
    self_model_valid, recursion_depth_safe, rubric_drift_score,
    quarantine_if_drift, constraint_coverage_valid, identity_coherent,
    can_spawn_child
)

from rubric_repair import (
    RubricRepairEngine, RubricDiagnostic, DiagnosticReport,
    DiagnosticResult, RepairAction, RepairOutcome,
    rubric_needs_repair, auto_repair_rubric, rubric_trajectory
)

from reflex_log import (
    ReflexLog, ReflexLogEvent, ObserverMesh, Observer, RuntimeMetrics,
    EventType, ObserverPhase, CollapseValidity,
    reflex_log_valid, quorum_reached, metrics_gates_passed, collapse_valid
)


# =============================================================================
# SELF-MODEL TESTS
# =============================================================================

class TestIdentityGraph(unittest.TestCase):
    """Tests for IdentityGraph"""
    
    def test_create_graph(self):
        """Test graph creation"""
        graph = IdentityGraph("TEST")
        self.assertEqual(graph.capsule_id, "TEST")
        self.assertEqual(len(graph.nodes), 0)
    
    def test_add_node(self):
        """Test adding nodes"""
        graph = IdentityGraph("TEST")
        node = IdentityNode(
            node_id="NODE1",
            aspect_type="goal",
            value="Test goal"
        )
        graph.add_node(node)
        self.assertEqual(len(graph.nodes), 1)
        self.assertIn("NODE1", graph.nodes)
    
    def test_add_edge(self):
        """Test adding edges"""
        graph = IdentityGraph("TEST")
        graph.add_node(IdentityNode("N1", "goal", "G1"))
        graph.add_node(IdentityNode("N2", "constraint", "C1"))
        
        result = graph.add_edge("N1", "N2")
        self.assertTrue(result)
        self.assertIn("N2", graph.edges["N1"])
    
    def test_coherence(self):
        """Test coherence calculation"""
        graph = IdentityGraph("TEST")
        # Empty graph
        self.assertEqual(graph.compute_coherence(), 0.0)
        
        # Single node
        graph.add_node(IdentityNode("N1", "goal", "G1"))
        coherence = graph.compute_coherence()
        self.assertGreaterEqual(coherence, 0.0)
    
    def test_hash_deterministic(self):
        """Test hash is deterministic"""
        graph1 = IdentityGraph("TEST")
        graph1.add_node(IdentityNode("N1", "goal", "G1"))
        
        graph2 = IdentityGraph("TEST")
        graph2.add_node(IdentityNode("N1", "goal", "G1"))
        
        self.assertEqual(graph1.compute_hash(), graph2.compute_hash())


class TestMetaRubric(unittest.TestCase):
    """Tests for MetaRubric"""
    
    def test_create_rubric(self):
        """Test rubric creation"""
        rubric = MetaRubric(
            rubric_id="RUB-001",
            name="Test Rubric",
            description="A test rubric"
        )
        self.assertEqual(rubric.rubric_id, "RUB-001")
        self.assertEqual(rubric.confidence_index, 1.0)
    
    def test_set_baseline(self):
        """Test baseline setting"""
        rubric = MetaRubric("RUB-001", "Test", "Desc")
        rubric.set_baseline()
        
        self.assertIsNotNone(rubric.baseline_hash)
        self.assertEqual(rubric.drift_score, 0.0)
    
    def test_update_drift(self):
        """Test drift tracking on update"""
        rubric = MetaRubric("RUB-001", "Test", "Desc")
        rubric.set_baseline()
        
        # Update should increase drift
        rubric.update({'confidence_index': 0.8})
        self.assertGreater(rubric.drift_score, 0.0)
    
    def test_evaluate_rubric(self):
        """Test meta-evaluation of another rubric"""
        anchor = MetaRubric("ANCHOR", "Anchor", "Main rubric")
        target = MetaRubric("TARGET", "Target", "Target rubric")
        target.drift_score = 0.5  # High drift
        
        result = anchor.evaluate_rubric(target)
        self.assertLess(result.score, 1.0)  # Should be penalized
    
    def test_reclassify(self):
        """Test reclassification"""
        rubric = MetaRubric("RUB-001", "Test", "Desc")
        rubric.reclassify("test_reason")
        
        self.assertEqual(rubric.confidence_index, 0.0)


class TestSafetyBounds(unittest.TestCase):
    """Tests for RecursiveSafetyBounds"""
    
    def test_core_bounds_initialized(self):
        """Test core bounds are present"""
        bounds = RecursiveSafetyBounds("TEST")
        
        self.assertIn("NO_UNBINDING", bounds.bounds)
        self.assertIn("AUDIT_VISIBILITY", bounds.bounds)
        self.assertIn("RECURSION_LIMIT", bounds.bounds)
        self.assertIn("DRIFT_BOUND", bounds.bounds)
    
    def test_no_unbinding_enforced(self):
        """Test No-Unbinding invariant"""
        bounds = RecursiveSafetyBounds("TEST")
        
        # Core bounds cannot be removed
        result = bounds.remove_bound("NO_UNBINDING")
        self.assertFalse(result)
        self.assertIn("NO_UNBINDING", bounds.bounds)
    
    def test_inheritance(self):
        """Test constraint inheritance"""
        parent = RecursiveSafetyBounds("PARENT")
        parent.add_bound(SafetyBound(
            bound_id="CUSTOM",
            name="Custom Bound",
            check_fn=lambda x: True
        ))
        
        child = RecursiveSafetyBounds("CHILD")
        child.inherit_from(parent)
        
        # Child should have parent's custom bound
        self.assertIn("CUSTOM", child.bounds)


class TestSelfModel(unittest.TestCase):
    """Tests for SelfModel"""
    
    def setUp(self):
        np.random.seed(42)
    
    def test_create_model(self):
        """Test model creation"""
        model = SelfModel("MODEL-001")
        
        self.assertEqual(model.capsule_id, "MODEL-001")
        self.assertEqual(model.state, SelfModelState.STABLE)
        self.assertEqual(model.lineage_depth, 0)
    
    def test_add_rubric(self):
        """Test adding rubrics"""
        model = SelfModel("MODEL-001")
        rubric = MetaRubric("RUB-001", "Test", "Desc")
        
        model.add_rubric(rubric)
        self.assertIn("RUB-001", model.rubrics)
    
    def test_reflect(self):
        """Test self-reflection"""
        model = SelfModel("MODEL-001")
        
        report = model.reflect()
        
        self.assertIn('validation', report)
        self.assertIn('identity_coherence', report)
        self.assertIn('safety_passed', report)
    
    def test_spawn_child(self):
        """Test spawning children"""
        parent = SelfModel("PARENT")
        rubric = MetaRubric("RUB-001", "Test", "Desc")
        parent.add_rubric(rubric)
        parent.set_alignment_anchor("RUB-001")
        
        child = parent.spawn_child("CHILD-001")
        
        self.assertIsNotNone(child)
        self.assertEqual(child.parent_id, "PARENT")
        self.assertEqual(child.lineage_depth, 1)
        # Child inherits safety bounds
        self.assertIn("NO_UNBINDING", child.safety_bounds.bounds)
    
    def test_recursion_limit(self):
        """Test recursion depth limit"""
        model = SelfModel("ROOT", lineage_depth=MAX_RECURSION_DEPTH)
        
        child = model.spawn_child("CHILD")
        self.assertIsNone(child)  # Should fail at max depth
    
    def test_modify_parameters(self):
        """Test parameter modification"""
        model = SelfModel("MODEL-001")
        
        success, msg = model.modify(
            ModificationType.PARAMETER_ADJUSTMENT,
            {'evi_score': 0.9}
        )
        
        self.assertTrue(success)
        self.assertEqual(model.evi_score, 0.9)
    
    def test_quarantine(self):
        """Test quarantine"""
        model = SelfModel("MODEL-001")
        model.quarantine("test_reason")
        
        self.assertEqual(model.state, SelfModelState.QUARANTINED)


# =============================================================================
# RUBRIC REPAIR TESTS
# =============================================================================

class TestRubricDiagnostic(unittest.TestCase):
    """Tests for RubricDiagnostic"""
    
    def test_diagnose_healthy(self):
        """Test diagnosis of healthy rubric"""
        diag = RubricDiagnostic()
        rubric = MetaRubric("RUB-001", "Test", "Desc")
        rubric.set_baseline()
        
        report = diag.diagnose(rubric)
        self.assertEqual(report.result, DiagnosticResult.HEALTHY)
    
    def test_diagnose_drift(self):
        """Test detection of drift"""
        diag = RubricDiagnostic()
        rubric = MetaRubric("RUB-001", "Test", "Desc")
        rubric.set_baseline()
        rubric.drift_score = 0.4  # High drift
        
        report = diag.diagnose(rubric)
        self.assertNotEqual(report.result, DiagnosticResult.HEALTHY)


class TestRubricRepairEngine(unittest.TestCase):
    """Tests for RubricRepairEngine"""
    
    def test_create_engine(self):
        """Test engine creation"""
        engine = RubricRepairEngine("ENGINE-001")
        self.assertEqual(engine.engine_id, "ENGINE-001")
        self.assertGreater(len(engine.strategies), 0)
    
    def test_diagnose_and_repair(self):
        """Test diagnosis and repair workflow"""
        engine = RubricRepairEngine("ENGINE-001")
        rubric = MetaRubric("RUB-001", "Test", "Desc")
        rubric.set_baseline()
        rubric.drift_score = 0.4
        
        diagnosis, repair = engine.diagnose_and_repair(rubric, auto_repair=True)
        
        self.assertIsNotNone(diagnosis)
        # May or may not repair depending on strategy
    
    def test_reconstruct_history(self):
        """Test history reconstruction"""
        engine = RubricRepairEngine("ENGINE-001")
        rubric = MetaRubric("RUB-001", "Test", "Desc")
        
        history = engine.reconstruct_history(rubric)
        self.assertIn('trajectory', history)


# =============================================================================
# REFLEX LOG TESTS
# =============================================================================

class TestReflexLog(unittest.TestCase):
    """Tests for ReflexLog"""
    
    def test_create_log(self):
        """Test log creation"""
        log = ReflexLog("LOG-001")
        self.assertEqual(log.log_id, "LOG-001")
        self.assertEqual(len(log.events), 0)
    
    def test_emit_event(self):
        """Test event emission"""
        log = ReflexLog("LOG-001")
        
        event = log.emit(
            capsule_id="CAP-001",
            event_type=EventType.REFLECTION,
            rci=0.7,
            psr=40,
            shy=0.08
        )
        
        self.assertEqual(len(log.events), 1)
        self.assertEqual(event.capsule_id, "CAP-001")
    
    def test_hash_chain(self):
        """Test hash chain integrity"""
        log = ReflexLog("LOG-001")
        
        log.emit("CAP-001", EventType.REFLECTION)
        log.emit("CAP-002", EventType.COLLAPSE)
        
        valid, _ = log.verify_chain()
        self.assertTrue(valid)
    
    def test_emit_action_proposal(self):
        """Test action proposal emission"""
        log = ReflexLog("LOG-001")
        
        event = log.emit_action_proposal(
            capsule_id="CAP-001",
            constraints_hash="abc123",
            rci=0.72,
            psr=45,
            shy=0.06
        )
        
        self.assertEqual(event.event_type, EventType.ACTION_PROPOSAL)


class TestObserverMesh(unittest.TestCase):
    """Tests for ObserverMesh"""
    
    def test_create_mesh(self):
        """Test mesh creation"""
        mesh = ObserverMesh("MESH-001")
        self.assertEqual(mesh.mesh_id, "MESH-001")
    
    def test_add_observers(self):
        """Test adding observers"""
        mesh = ObserverMesh("MESH-001")
        mesh.add_observer(Observer("OBS-001"))
        mesh.add_observer(Observer("OBS-002"))
        
        self.assertEqual(len(mesh.observers), 2)
    
    def test_quorum_computation(self):
        """Test quorum calculation"""
        mesh = ObserverMesh("MESH-001", quorum_threshold=0.5)
        mesh.add_observer(Observer("OBS-001"))
        mesh.add_observer(Observer("OBS-002"))
        
        votes = {"OBS-001": True, "OBS-002": True}
        count, ratio = mesh.compute_quorum(votes)
        
        self.assertEqual(count, 2)
        self.assertEqual(ratio, 1.0)
    
    def test_validate_collapse(self):
        """Test collapse validation"""
        mesh = ObserverMesh("MESH-001")
        mesh.add_observer(Observer("OBS-001"))
        mesh.add_observer(Observer("OBS-002"))
        
        validity, record = mesh.validate_collapse(
            event_id="EVT-001",
            pre_state_hash="pre",
            post_state_hash="post"
        )
        
        self.assertEqual(validity, CollapseValidity.VALID)


class TestRuntimeMetrics(unittest.TestCase):
    """Tests for RuntimeMetrics"""
    
    def test_create_metrics(self):
        """Test metrics creation"""
        metrics = RuntimeMetrics()
        self.assertEqual(metrics.rci, 0.65)
    
    def test_update_metrics(self):
        """Test metrics update"""
        metrics = RuntimeMetrics()
        metrics.update(0.8, 50, 0.05)
        
        self.assertEqual(metrics.rci, 0.8)
        self.assertEqual(len(metrics.rci_history), 1)
    
    def test_check_gates(self):
        """Test gate checking"""
        metrics = RuntimeMetrics()
        metrics.update(0.7, 40, 0.08)
        
        passed, violations = metrics.check_gates()
        self.assertTrue(passed)
    
    def test_gate_violation(self):
        """Test gate violation detection"""
        metrics = RuntimeMetrics()
        metrics.update(0.5, 40, 0.08)  # RCI below gate
        
        passed, violations = metrics.check_gates()
        self.assertFalse(passed)
        self.assertGreater(len(violations), 0)


# =============================================================================
# DSL PREDICATE TESTS
# =============================================================================

class TestDSLPredicates(unittest.TestCase):
    """Tests for DSL predicates"""
    
    def setUp(self):
        np.random.seed(42)
    
    def test_recursion_depth_safe(self):
        """Test recursion depth predicate"""
        model = SelfModel("TEST")
        self.assertTrue(recursion_depth_safe(model))
        
        model.lineage_depth = MAX_RECURSION_DEPTH + 1
        self.assertFalse(recursion_depth_safe(model))
    
    def test_rubric_drift_score(self):
        """Test rubric drift score predicate"""
        model = SelfModel("TEST")
        rubric = MetaRubric("RUB-001", "Test", "Desc")
        model.add_rubric(rubric)
        # Set drift after adding (add_rubric calls set_baseline which resets drift)
        model.rubrics["RUB-001"].drift_score = 0.2
        
        score = rubric_drift_score(model)
        self.assertEqual(score, 0.2)
    
    def test_quarantine_if_drift(self):
        """Test quarantine predicate"""
        model = SelfModel("TEST")
        rubric = MetaRubric("RUB-001", "Test", "Desc")
        model.add_rubric(rubric)
        # Set drift after adding
        model.rubrics["RUB-001"].drift_score = 0.5
        
        result = quarantine_if_drift(model, threshold=0.4)
        self.assertTrue(result)
        self.assertEqual(model.state, SelfModelState.QUARANTINED)
    
    def test_can_spawn_child(self):
        """Test spawn capability predicate"""
        model = SelfModel("TEST")
        self.assertTrue(can_spawn_child(model))
        
        model.lineage_depth = MAX_RECURSION_DEPTH
        self.assertFalse(can_spawn_child(model))
    
    def test_reflex_log_valid(self):
        """Test log validity predicate"""
        log = ReflexLog("TEST")
        log.emit("CAP-001", EventType.REFLECTION)
        
        self.assertTrue(reflex_log_valid(log))
    
    def test_metrics_gates_passed(self):
        """Test metrics gates predicate"""
        metrics = RuntimeMetrics()
        metrics.update(0.7, 40, 0.08)
        
        self.assertTrue(metrics_gates_passed(metrics))


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        np.random.seed(42)
    
    def test_full_self_modeling_workflow(self):
        """Test complete self-modeling workflow"""
        # Create model
        model = SelfModel("ROOT")
        
        # Add rubrics
        rubric = MetaRubric("RUB-001", "Core", "Core alignment")
        model.add_rubric(rubric)
        model.set_alignment_anchor("RUB-001")
        
        # Spawn child
        child = model.spawn_child("CHILD-001")
        self.assertIsNotNone(child)
        
        # Reflect
        report = model.reflect()
        self.assertIn('validation', report)
        
        # Repair if needed
        engine = RubricRepairEngine("ENGINE")
        diagnosis, repair = engine.diagnose_and_repair(rubric)
        self.assertIsNotNone(diagnosis)
    
    def test_audit_trail_workflow(self):
        """Test complete audit trail workflow"""
        # Create log
        log = ReflexLog("AUDIT")
        
        # Create mesh
        mesh = ObserverMesh("MESH")
        for i in range(5):
            mesh.add_observer(Observer(f"OBS-{i}"))
        
        # Create model
        model = SelfModel("MODEL")
        
        # Emit action proposal
        log.emit_action_proposal(
            capsule_id=model.capsule_id,
            constraints_hash="abc",
            rci=0.7,
            psr=40,
            shy=0.08
        )
        
        # Validate collapse
        validity, record = mesh.validate_collapse(
            "EVT-001", "pre", "post"
        )
        
        # Emit collapse
        log.emit_collapse(
            capsule_id=model.capsule_id,
            observer_phase=3,
            quorum=5,
            delta_state_hash="delta"
        )
        
        # Verify
        self.assertTrue(reflex_log_valid(log))
        self.assertEqual(validity, CollapseValidity.VALID)


# =============================================================================
# REFLEXIVE SWARM TESTS
# =============================================================================

class TestCapsuleDriftVector(unittest.TestCase):
    """Tests for CapsuleDriftVector"""
    
    def test_create_drift_vector(self):
        """Test drift vector creation"""
        from reflexive_swarm import CapsuleDriftVector, DriftDimension
        
        dv = CapsuleDriftVector("TEST-001")
        self.assertEqual(dv.capsule_id, "TEST-001")
        # All dimensions should be initialized to 0
        for dim in DriftDimension:
            self.assertEqual(dv.dimensions[dim], 0.0)
    
    def test_set_dimension(self):
        """Test setting dimension values"""
        from reflexive_swarm import CapsuleDriftVector, DriftDimension
        
        dv = CapsuleDriftVector("TEST-001")
        dv.set_dimension(DriftDimension.BEHAVIORAL, 0.45)
        self.assertEqual(dv.dimensions[DriftDimension.BEHAVIORAL], 0.45)
    
    def test_compute_magnitude(self):
        """Test magnitude computation"""
        from reflexive_swarm import CapsuleDriftVector, DriftDimension
        
        dv = CapsuleDriftVector("TEST-001")
        dv.set_dimension(DriftDimension.BEHAVIORAL, 0.3)
        dv.set_dimension(DriftDimension.STRUCTURAL, 0.4)
        
        magnitude = dv.compute_magnitude()
        self.assertGreater(magnitude, 0.0)
    
    def test_exceeds_threshold(self):
        """Test threshold detection"""
        from reflexive_swarm import CapsuleDriftVector, DriftDimension
        
        dv = CapsuleDriftVector("TEST-001")
        self.assertFalse(dv.exceeds_threshold(0.35))
        
        dv.set_dimension(DriftDimension.ALIGNMENT, 0.5)
        self.assertTrue(dv.exceeds_threshold(0.35))


class TestHeartbeatProtocol(unittest.TestCase):
    """Tests for HeartbeatProtocol"""
    
    def test_create_protocol(self):
        """Test protocol creation"""
        from reflexive_swarm import HeartbeatProtocol
        
        proto = HeartbeatProtocol("TEST-HB")
        self.assertEqual(proto.protocol_id, "TEST-HB")
    
    def test_register_and_checkin(self):
        """Test registration and check-in"""
        from reflexive_swarm import HeartbeatProtocol, HeartbeatStatus
        
        proto = HeartbeatProtocol("TEST-HB")
        proto.register_capsule("CAP-001", None, "SWARM-A")
        
        status = proto.check_status("CAP-001")
        self.assertEqual(status, HeartbeatStatus.ALIVE)
    
    def test_parent_child_tracking(self):
        """Test parent-child relationship tracking"""
        from reflexive_swarm import HeartbeatProtocol
        
        proto = HeartbeatProtocol("TEST-HB")
        proto.register_capsule("PARENT", None, None)
        proto.register_capsule("CHILD-1", "PARENT", None)
        proto.register_capsule("CHILD-2", "PARENT", None)
        
        children_status = proto.get_children_status("PARENT")
        self.assertEqual(len(children_status), 2)


class TestReflexiveSwarmAgreement(unittest.TestCase):
    """Tests for ReflexiveSwarmAgreement"""
    
    def test_create_agreement(self):
        """Test agreement creation"""
        from reflexive_swarm import ReflexiveSwarmAgreement
        
        rsa = ReflexiveSwarmAgreement("TEST-SA")
        self.assertEqual(rsa.swarm_id, "TEST-SA")
    
    def test_peer_evaluation(self):
        """Test peer evaluation"""
        from reflexive_swarm import ReflexiveSwarmAgreement, CapsuleDriftVector
        
        rsa = ReflexiveSwarmAgreement("TEST-SA")
        
        # Add members
        dv1 = CapsuleDriftVector("CAP-001")
        dv2 = CapsuleDriftVector("CAP-002")
        rsa.add_member("CAP-001", dv1)
        rsa.add_member("CAP-002", dv2)
        
        # Run evaluation
        result = rsa.run_peer_evaluation_round()
        self.assertEqual(result['evaluators'], 2)
        self.assertGreater(result['votes_cast'], 0)
    
    def test_consensus_computation(self):
        """Test consensus computation"""
        from reflexive_swarm import ReflexiveSwarmAgreement, CapsuleDriftVector
        
        rsa = ReflexiveSwarmAgreement("TEST-SA")
        
        # Add members
        for i in range(5):
            dv = CapsuleDriftVector(f"CAP-{i:03d}")
            rsa.add_member(f"CAP-{i:03d}", dv)
        
        # Run evaluation
        rsa.run_peer_evaluation_round()
        
        # Compute consensus
        consensus = rsa.compute_consensus("CAP-001")
        self.assertIn('consensus', consensus)
        self.assertIn('quorum_reached', consensus)


class TestModificationCostIndex(unittest.TestCase):
    """Tests for ModificationCostIndex"""
    
    def test_create_index(self):
        """Test cost index creation"""
        from reflexive_swarm import ModificationCostIndex
        
        mci = ModificationCostIndex("TEST-CI", budget=100.0)
        self.assertEqual(mci.budget, 100.0)
        self.assertEqual(mci.spent, 0.0)
    
    def test_request_modification(self):
        """Test modification request"""
        from reflexive_swarm import ModificationCostIndex
        
        mci = ModificationCostIndex("TEST-CI", budget=100.0)
        
        approved, cost = mci.request_modification("CAP-001", "parameter_adjustment")
        self.assertTrue(approved)
        self.assertGreater(mci.spent, 0)
    
    def test_budget_exhaustion(self):
        """Test budget exhaustion denial"""
        from reflexive_swarm import ModificationCostIndex
        
        mci = ModificationCostIndex("TEST-CI", budget=10.0)
        
        # This should exhaust budget
        approved1, _ = mci.request_modification("CAP-001", "rubric_update")  # 15.0
        self.assertFalse(approved1)


class TestCascadeAlertSystem(unittest.TestCase):
    """Tests for CascadeAlertSystem"""
    
    def test_create_system(self):
        """Test alert system creation"""
        from reflexive_swarm import CascadeAlertSystem
        
        cas = CascadeAlertSystem("TEST-CA")
        self.assertEqual(cas.system_id, "TEST-CA")
    
    def test_cascade_alert(self):
        """Test cascade alert generation"""
        from reflexive_swarm import CascadeAlertSystem, AlertSeverity
        
        cas = CascadeAlertSystem("TEST-CA")
        
        # Register lineage
        cas.register_lineage("ROOT", None)
        cas.register_lineage("CHILD", "ROOT")
        cas.register_lineage("GRANDCHILD", "CHILD")
        
        # Trigger alert
        alert = cas.cascade_alert(
            "CHILD",
            "Drift detected",
            severity=AlertSeverity.WARNING
        )
        
        self.assertIsNotNone(alert)
        self.assertIn("ROOT", alert.affected_lineage)
        self.assertIn("GRANDCHILD", alert.affected_lineage)
    
    def test_trace_lineage(self):
        """Test lineage tracing"""
        from reflexive_swarm import CascadeAlertSystem
        
        cas = CascadeAlertSystem("TEST-CA")
        cas.register_lineage("ROOT", None)
        cas.register_lineage("CHILD", "ROOT")
        cas.register_lineage("GRANDCHILD", "CHILD")
        
        ancestors = cas.trace_ancestors("GRANDCHILD")
        self.assertEqual(ancestors, ["CHILD", "ROOT"])
        
        descendants = cas.trace_descendants("ROOT")
        self.assertIn("CHILD", descendants)
        self.assertIn("GRANDCHILD", descendants)


class TestReflexiveSwarmController(unittest.TestCase):
    """Tests for integrated ReflexiveSwarmController"""
    
    def test_create_controller(self):
        """Test controller creation"""
        from reflexive_swarm import ReflexiveSwarmController
        
        ctrl = ReflexiveSwarmController("TEST-CTRL")
        self.assertEqual(ctrl.controller_id, "TEST-CTRL")
    
    def test_register_capsule(self):
        """Test capsule registration"""
        from reflexive_swarm import ReflexiveSwarmController
        
        ctrl = ReflexiveSwarmController("TEST-CTRL")
        dv = ctrl.register_capsule("CAP-001", None, "SWARM-A")
        
        self.assertIsNotNone(dv)
        self.assertEqual(len(ctrl.capsules), 1)
    
    def test_evaluation_cycle(self):
        """Test full evaluation cycle"""
        from reflexive_swarm import ReflexiveSwarmController, DriftDimension
        
        ctrl = ReflexiveSwarmController("TEST-CTRL")
        
        # Register capsules
        ctrl.register_capsule("ROOT", None, "SWARM-A")
        ctrl.register_capsule("CHILD-1", "ROOT", "SWARM-A")
        ctrl.register_capsule("CHILD-2", "ROOT", "SWARM-A")
        
        # Inject drift
        ctrl.update_drift("CHILD-2", DriftDimension.BEHAVIORAL, 0.5)
        
        # Run cycle
        result = ctrl.run_evaluation_cycle()
        
        self.assertEqual(result['capsules_evaluated'], 3)
        self.assertIn('alerts_triggered', result)
    
    def test_comprehensive_status(self):
        """Test comprehensive status retrieval"""
        from reflexive_swarm import ReflexiveSwarmController
        
        ctrl = ReflexiveSwarmController("TEST-CTRL")
        ctrl.register_capsule("CAP-001", None, None)
        
        status = ctrl.get_comprehensive_status()
        
        self.assertIn('heartbeat', status)
        self.assertIn('swarm_agreement', status)
        self.assertIn('cost_index', status)
        self.assertIn('cascade_alerts', status)
        self.assertIn('flag_system', status)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestIdentityGraph,
        TestMetaRubric,
        TestSafetyBounds,
        TestSelfModel,
        TestRubricDiagnostic,
        TestRubricRepairEngine,
        TestReflexLog,
        TestObserverMesh,
        TestRuntimeMetrics,
        TestDSLPredicates,
        TestIntegration,
        TestCapsuleDriftVector,
        TestHeartbeatProtocol,
        TestReflexiveSwarmAgreement,
        TestModificationCostIndex,
        TestCascadeAlertSystem,
        TestReflexiveSwarmController,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("BOOKLET 8 TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
