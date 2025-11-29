"""
RSCS-Q Capstone: Integration Harness
====================================

Cross-booklet validation harness for B6→B7→B8 integration.

Author: Entropica Research Collective
Version: 3.0.1
"""

import sys
import os
import json
import unittest
from datetime import datetime
from typing import Dict, Any, List

# Add module paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Booklet6'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Booklet7'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Booklet8', 'src'))


class IntegrationHarness:
    """
    Cross-booklet integration test harness.
    
    Validates the B6→B7→B8 pipeline with end-to-end scenarios.
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration"""
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {
            'scenarios': ['baseline', 'drift', 'recovery'],
            'ticks': 100,
            'seed': 42
        }
    
    def run_scenario(self, scenario: str) -> Dict[str, Any]:
        """Run a single integration scenario"""
        result = {
            'scenario': scenario,
            'start': datetime.utcnow().isoformat(),
            'b6_pass': False,
            'b7_pass': False,
            'b8_pass': False,
            'integration_pass': False,
            'errors': []
        }
        
        try:
            # Test B6 components
            from drift_surface import B6Interface, EVIScore, MDSScore
            b6 = B6Interface(f"INT-{scenario}")
            import numpy as np
            
            pred = np.random.rand(10)
            actual = pred + np.random.rand(10) * 0.1
            evi = b6.compute_evi(pred, actual)
            mds = b6.compute_mds(actual)
            
            result['b6_pass'] = evi.is_valid()
            result['b6_metrics'] = {'evi': evi.value, 'mds': mds.value}
            
        except Exception as e:
            result['errors'].append(f"B6: {str(e)}")
        
        try:
            # Test B7 components
            from meta_kernel_bridge import MetaKernelBridge, SwarmCoherence, SwarmMember
            
            bridge = MetaKernelBridge(f"INT-{scenario}")
            bridge.receive_b6_metrics(
                evi=result.get('b6_metrics', {}).get('evi', 0.5),
                mds=result.get('b6_metrics', {}).get('mds', 0.3)
            )
            
            # Add swarm members
            for i in range(5):
                m = SwarmMember(f"M-{i}")
                m.update_hash({'scenario': scenario, 'tick': i})
                bridge.swarm.add_member(m)
            
            coherence = bridge.swarm.compute_coherence()
            result['b7_pass'] = coherence >= 0.67
            result['b7_metrics'] = {
                'coherence': coherence,
                'activation': bridge.current_profile.level.name
            }
            
        except Exception as e:
            result['errors'].append(f"B7: {str(e)}")
        
        try:
            # Test B8 components
            from self_model import SelfModel, MetaRubric, self_model_valid
            from rubric_repair import RubricRepairEngine
            from reflex_log import ReflexLog, EventType
            
            model = SelfModel(f"INT-{scenario}")
            rubric = MetaRubric("ALIGN-001", "Integration", "Test")
            model.add_rubric(rubric)
            model.set_alignment_anchor("ALIGN-001")
            model.reflect()
            
            log = ReflexLog(f"INT-{scenario}")
            for i in range(10):
                log.emit(f"CAP-{i}", EventType.REFLECTION)
            
            valid, _ = log.verify_chain()
            
            result['b8_pass'] = self_model_valid(model) and valid
            result['b8_metrics'] = {
                'validity': model.validity_score,
                'log_valid': valid
            }
            
        except Exception as e:
            result['errors'].append(f"B8: {str(e)}")
        
        # Overall integration
        result['integration_pass'] = (
            result['b6_pass'] and 
            result['b7_pass'] and 
            result['b8_pass']
        )
        result['end'] = datetime.utcnow().isoformat()
        
        self.results.append(result)
        return result
    
    def run_all(self) -> Dict[str, Any]:
        """Run all configured scenarios"""
        self.start_time = datetime.utcnow()
        
        for scenario in self.config['scenarios']:
            self.run_scenario(scenario)
        
        summary = {
            'total': len(self.results),
            'passed': sum(1 for r in self.results if r['integration_pass']),
            'failed': sum(1 for r in self.results if not r['integration_pass']),
            'duration': (datetime.utcnow() - self.start_time).total_seconds(),
            'results': self.results
        }
        
        return summary
    
    def generate_report(self) -> str:
        """Generate human-readable report"""
        lines = [
            "=" * 60,
            "RSCS-Q CAPSTONE INTEGRATION REPORT",
            "=" * 60,
            f"Timestamp: {datetime.utcnow().isoformat()}",
            ""
        ]
        
        for r in self.results:
            status = "✅ PASS" if r['integration_pass'] else "❌ FAIL"
            lines.append(f"Scenario: {r['scenario']} - {status}")
            lines.append(f"  B6: {'✓' if r['b6_pass'] else '✗'}")
            lines.append(f"  B7: {'✓' if r['b7_pass'] else '✗'}")
            lines.append(f"  B8: {'✓' if r['b8_pass'] else '✗'}")
            if r['errors']:
                lines.append(f"  Errors: {r['errors']}")
            lines.append("")
        
        passed = sum(1 for r in self.results if r['integration_pass'])
        total = len(self.results)
        lines.append("=" * 60)
        lines.append(f"SUMMARY: {passed}/{total} scenarios passed")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def run_integration_tests():
    """Run integration test suite"""
    harness = IntegrationHarness()
    summary = harness.run_all()
    
    print(harness.generate_report())
    
    return summary['failed'] == 0


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
