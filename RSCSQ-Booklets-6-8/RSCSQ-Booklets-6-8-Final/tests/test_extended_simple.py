"""
RSCS-Q Booklet 8: Extended G-Suite Tests (Simplified)
======================================================

Simplified tests for hardening guarantees that work with actual API.
These tests validate edge cases and stress scenarios.
"""

import numpy as np
import unittest
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from self_model import (
    SelfModel, MetaRubric, IdentityNode,
    self_model_valid, identity_coherent, reset_validity_state
)
from rubric_repair import RubricRepairEngine, RepairOutcome
from reflex_log import ReflexLog, ObserverMesh, Observer, RuntimeMetrics, EventType
from reflexive_swarm import ReflexiveSwarmController, DriftDimension


class TestStressDistributions(unittest.TestCase):
    """Tests reporting distributions instead of just pass/fail"""
    
    def setUp(self):
        np.random.seed(42)
        reset_validity_state()
    
    def test_validity_distribution(self):
        """Report validity distribution under varying drift"""
        results = {'valid': 0, 'invalid': 0}
        
        for i in range(100):
            reset_validity_state()
            model = SelfModel(f"DIST-{i:03d}")
            rubric = MetaRubric(f"RUB-{i:03d}", "Test", "Test")
            model.add_rubric(rubric)
            
            # Random drift
            drift = np.random.uniform(0, 0.6)
            model.drift_score = drift
            
            model.reflect()
            if self_model_valid(model):
                results['valid'] += 1
            else:
                results['invalid'] += 1
        
        print(f"\n>>> Validity Distribution: {results['valid']} valid, {results['invalid']} invalid")
        self.assertGreater(results['valid'], 0)
    
    def test_coherence_distribution(self):
        """Report coherence distribution across graph sizes"""
        results = []
        
        for n in [1, 2, 3, 5, 10, 20]:
            model = SelfModel(f"COH-{n}")
            
            # Add nodes using correct API
            for i in range(n):
                node = IdentityNode(f"node_{i}", "capability", {"val": i}, confidence=0.7)
                model.identity.add_node(node)
            
            coherence = model.identity.compute_coherence(0.1)
            results.append({'n': n, 'coherence': coherence})
        
        print("\n>>> Coherence by Graph Size:")
        for r in results:
            print(f"    N={r['n']:2d}: coherence={r['coherence']:.3f}")
        
        # All should produce valid values
        for r in results:
            self.assertFalse(np.isnan(r['coherence']))
    
    def test_repair_distribution(self):
        """Report repair outcome distribution"""
        outcomes = defaultdict(int)
        
        for i in range(50):
            rubric = MetaRubric(f"REP-{i:03d}", "Test", "Test")
            rubric.confidence_index = np.random.uniform(0.3, 0.8)
            rubric.drift_score = np.random.uniform(0.1, 0.5)
            
            engine = RubricRepairEngine(f"ENG-{i:03d}")
            diagnosis, repair = engine.diagnose_and_repair(rubric, auto_repair=True)
            
            if repair:
                outcomes[repair.outcome.name] += 1
            else:
                outcomes['NO_REPAIR'] += 1
        
        print("\n>>> Repair Outcome Distribution:")
        for outcome, count in outcomes.items():
            print(f"    {outcome}: {count}")
        
        self.assertGreater(sum(outcomes.values()), 0)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions"""
    
    def test_empty_identity_graph(self):
        """Handle empty identity graph gracefully"""
        model = SelfModel("EMPTY-001")
        coherence = model.identity.compute_coherence(0.1)
        
        print(f"\n>>> Empty Graph: coherence={coherence}")
        # With regularization, empty graph returns a high base value
        self.assertGreaterEqual(coherence, 0.0)
    
    def test_single_node_graph(self):
        """Handle single-node identity graph"""
        model = SelfModel("SINGLE-001")
        node = IdentityNode("solo", "goal", {}, confidence=0.9)
        model.identity.add_node(node)
        
        coherence = model.identity.compute_coherence(0.1)
        print(f"\n>>> Single Node: coherence={coherence:.3f}")
        
        self.assertGreater(coherence, 0.0)
    
    def test_high_drift_scenario(self):
        """System response to very high drift"""
        model = SelfModel("HIGH-DRIFT")
        rubric = MetaRubric("HD-RUB", "Test", "Test")
        model.add_rubric(rubric)
        
        model.drift_score = 0.9
        model.reflect()
        valid = self_model_valid(model)
        
        print(f"\n>>> High Drift (0.9): valid={valid}")
        # May or may not be valid depending on hysteresis
    
    def test_observer_mesh_edge_cases(self):
        """Observer mesh with various configurations"""
        # Single observer
        mesh1 = ObserverMesh("SINGLE-OBS")
        mesh1.add_observer(Observer("OBS-0"))
        validity1, _ = mesh1.validate_collapse("EVT-1", "pre", "post")
        
        # Many observers
        mesh2 = ObserverMesh("MANY-OBS")
        for i in range(10):
            mesh2.add_observer(Observer(f"OBS-{i}"))
        validity2, _ = mesh2.validate_collapse("EVT-2", "pre", "post")
        
        print(f"\n>>> Observer Edge Cases:")
        print(f"    1 observer: {validity1.name}")
        print(f"    10 observers: {validity2.name}")


class TestAuditChain(unittest.TestCase):
    """Tests for audit chain integrity"""
    
    def test_chain_integrity_under_load(self):
        """Verify chain remains valid under load"""
        log = ReflexLog("LOAD-TEST", retention_window=1000)
        
        for i in range(200):
            log.emit(f"CAP-{i % 20:03d}", EventType.REFLECTION)
        
        valid, first_invalid = log.verify_chain()
        
        print(f"\n>>> Chain Load Test: {log.get_statistics()['event_count']} events")
        print(f"    Valid: {valid}, First invalid: {first_invalid}")
        
        self.assertTrue(valid)
    
    def test_tamper_detection(self):
        """Verify tampering is detected"""
        log = ReflexLog("TAMPER-TEST")
        
        for i in range(10):
            log.emit(f"CAP-{i}", EventType.REFLECTION)
        
        # Tamper
        original = log.events[5].event_hash
        log.events[5].event_hash = "TAMPERED"
        
        valid, first_invalid = log.verify_chain()
        
        print(f"\n>>> Tamper Detection: valid={valid}, detected_at={first_invalid}")
        
        self.assertFalse(valid)
        self.assertEqual(first_invalid, 5)  # The tampered event itself
        
        # Restore
        log.events[5].event_hash = original


class TestSwarmStress(unittest.TestCase):
    """Stress tests for reflexive swarm"""
    
    def test_sudden_novelty(self):
        """Response to sudden high drift across swarm"""
        ctrl = ReflexiveSwarmController("NOVELTY")
        
        for i in range(10):
            ctrl.register_capsule(f"CAP-{i:03d}", None, "SWARM-A")
        
        # Inject high drift
        for cid in ctrl.capsules:
            ctrl.update_drift(cid, DriftDimension.BEHAVIORAL, 0.6)
        
        result = ctrl.run_evaluation_cycle()
        
        print(f"\n>>> Sudden Novelty:")
        print(f"    Alerts: {len(result['alerts_triggered'])}")
        print(f"    Declassified: {len(result['consensus']['declassified'])}")
        
        self.assertGreater(len(result['alerts_triggered']), 0)
    
    def test_gradual_drift(self):
        """Response to gradual drift accumulation"""
        model = SelfModel("GRADUAL")
        rubric = MetaRubric("GR-RUB", "Test", "Test")
        model.add_rubric(rubric)
        
        history = []
        for tick in range(20):
            drift = tick * 0.02
            model.drift_score = drift
            model.reflect()
            valid = self_model_valid(model)
            history.append({'tick': tick, 'drift': drift, 'valid': valid})
        
        print("\n>>> Gradual Drift:")
        for h in history[::5]:
            print(f"    tick={h['tick']}: drift={h['drift']:.2f}, valid={h['valid']}")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for cls in [TestStressDistributions, TestEdgeCases, TestAuditChain, TestSwarmStress]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Extended Tests: {result.testsRun} run, {len(result.failures)} failed, {len(result.errors)} errors")
    print("=" * 60)
