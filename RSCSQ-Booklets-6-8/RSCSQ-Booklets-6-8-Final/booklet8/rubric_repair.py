"""
RSCS-Q Booklet 8: Rubric Repair Engine
======================================

Self-healing system for rubric maintenance and repair.

This module implements:
- RubricDiagnostic: Analyzes rubric health
- RepairStrategy: Defines repair approaches
- RubricRepairEngine: Orchestrates repair operations
- RepairLog: Audit trail for repairs

Author: Entropica Research Collective
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum, auto
from datetime import datetime
from collections import deque
import hashlib
import json

from self_model import (
    MetaRubric, SelfModel, RubricScore, ValidationResult,
    MAX_RUBRIC_DRIFT
)


# =============================================================================
# ENUMS
# =============================================================================

class DiagnosticResult(Enum):
    """Result of rubric diagnostic"""
    HEALTHY = auto()
    DRIFT_WARNING = auto()
    CONFIDENCE_LOW = auto()
    CONSISTENCY_FAILURE = auto()
    EVOLUTION_ANOMALY = auto()
    CRITICAL_FAILURE = auto()


class RepairAction(Enum):
    """Types of repair actions"""
    RESET_BASELINE = auto()
    RESTORE_CONFIDENCE = auto()
    PRUNE_EVOLUTION = auto()
    MERGE_PARENT = auto()
    QUARANTINE = auto()
    REBUILD = auto()
    NO_ACTION = auto()


class RepairOutcome(Enum):
    """Outcome of repair attempt"""
    SUCCESS = auto()
    PARTIAL = auto()
    FAILED = auto()
    DEFERRED = auto()


# =============================================================================
# DIAGNOSTIC SYSTEM
# =============================================================================

@dataclass
class DiagnosticReport:
    """Report from rubric diagnostic"""
    rubric_id: str
    result: DiagnosticResult
    severity: float  # 0.0 (healthy) to 1.0 (critical)
    issues: List[str]
    recommendations: List[RepairAction]
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rubric_id': self.rubric_id,
            'result': self.result.name,
            'severity': self.severity,
            'issues': self.issues,
            'recommendations': [r.name for r in self.recommendations],
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat()
        }


class RubricDiagnostic:
    """
    Diagnostic system for analyzing rubric health.
    
    Checks for:
    - Drift accumulation
    - Confidence degradation
    - Evolution consistency
    - Inter-rubric conflicts
    """
    
    def __init__(self):
        self.diagnostic_history: deque = deque(maxlen=1000)
        self.thresholds = {
            'drift_warning': 0.2,
            'drift_critical': 0.35,
            'confidence_warning': 0.6,
            'confidence_critical': 0.4,
            'evolution_anomaly_rate': 0.3
        }
    
    def diagnose(self, rubric: MetaRubric) -> DiagnosticReport:
        """Perform comprehensive diagnosis of a rubric"""
        issues = []
        recommendations = []
        severity = 0.0
        
        metrics = {
            'drift_score': rubric.drift_score,
            'confidence_index': rubric.confidence_index,
            'evolution_count': len(rubric.evolution_history),
            'max_drift': rubric.max_drift
        }
        
        # Check drift
        drift_status = self._check_drift(rubric)
        if drift_status['issue']:
            issues.append(drift_status['message'])
            recommendations.extend(drift_status['recommendations'])
            severity = max(severity, drift_status['severity'])
        
        # Check confidence
        conf_status = self._check_confidence(rubric)
        if conf_status['issue']:
            issues.append(conf_status['message'])
            recommendations.extend(conf_status['recommendations'])
            severity = max(severity, conf_status['severity'])
        
        # Check evolution consistency
        evol_status = self._check_evolution(rubric)
        if evol_status['issue']:
            issues.append(evol_status['message'])
            recommendations.extend(evol_status['recommendations'])
            severity = max(severity, evol_status['severity'])
        
        # Determine overall result
        if severity >= 0.8:
            result = DiagnosticResult.CRITICAL_FAILURE
        elif severity >= 0.6:
            result = DiagnosticResult.CONSISTENCY_FAILURE
        elif severity >= 0.4:
            result = DiagnosticResult.CONFIDENCE_LOW
        elif severity >= 0.2:
            result = DiagnosticResult.DRIFT_WARNING
        else:
            result = DiagnosticResult.HEALTHY
            recommendations = [RepairAction.NO_ACTION]
        
        report = DiagnosticReport(
            rubric_id=rubric.rubric_id,
            result=result,
            severity=severity,
            issues=issues,
            recommendations=recommendations,
            metrics=metrics
        )
        
        self.diagnostic_history.append(report)
        return report
    
    def _check_drift(self, rubric: MetaRubric) -> Dict[str, Any]:
        """Check drift status"""
        drift = rubric.drift_score
        
        if drift >= self.thresholds['drift_critical']:
            return {
                'issue': True,
                'message': f"Critical drift: {drift:.3f}",
                'severity': 0.9,
                'recommendations': [RepairAction.RESET_BASELINE, RepairAction.QUARANTINE]
            }
        elif drift >= self.thresholds['drift_warning']:
            return {
                'issue': True,
                'message': f"Drift warning: {drift:.3f}",
                'severity': 0.5,
                'recommendations': [RepairAction.RESET_BASELINE]
            }
        
        return {'issue': False, 'severity': 0.0, 'recommendations': []}
    
    def _check_confidence(self, rubric: MetaRubric) -> Dict[str, Any]:
        """Check confidence level"""
        conf = rubric.confidence_index
        
        if conf <= self.thresholds['confidence_critical']:
            return {
                'issue': True,
                'message': f"Critical confidence: {conf:.3f}",
                'severity': 0.8,
                'recommendations': [RepairAction.RESTORE_CONFIDENCE, RepairAction.MERGE_PARENT]
            }
        elif conf <= self.thresholds['confidence_warning']:
            return {
                'issue': True,
                'message': f"Low confidence: {conf:.3f}",
                'severity': 0.4,
                'recommendations': [RepairAction.RESTORE_CONFIDENCE]
            }
        
        return {'issue': False, 'severity': 0.0, 'recommendations': []}
    
    def _check_evolution(self, rubric: MetaRubric) -> Dict[str, Any]:
        """Check evolution history for anomalies"""
        history = rubric.evolution_history
        
        if len(history) < 3:
            return {'issue': False, 'severity': 0.0, 'recommendations': []}
        
        # Check for consistently declining confidence
        recent = history[-5:]
        confidence_deltas = [e.get('confidence_delta', 0) for e in recent]
        
        declining = sum(1 for d in confidence_deltas if d < 0)
        if declining / len(confidence_deltas) > self.thresholds['evolution_anomaly_rate']:
            return {
                'issue': True,
                'message': f"Evolution anomaly: {declining}/{len(confidence_deltas)} declining",
                'severity': 0.6,
                'recommendations': [RepairAction.PRUNE_EVOLUTION]
            }
        
        return {'issue': False, 'severity': 0.0, 'recommendations': []}
    
    def diagnose_model(self, model: SelfModel) -> List[DiagnosticReport]:
        """Diagnose all rubrics in a self-model"""
        reports = []
        for rubric in model.rubrics.values():
            reports.append(self.diagnose(rubric))
        return reports


# =============================================================================
# REPAIR STRATEGIES
# =============================================================================

@dataclass
class RepairResult:
    """Result of a repair operation"""
    rubric_id: str
    action: RepairAction
    outcome: RepairOutcome
    details: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rubric_id': self.rubric_id,
            'action': self.action.name,
            'outcome': self.outcome.name,
            'details': self.details,
            'before_metrics': self.before_metrics,
            'after_metrics': self.after_metrics,
            'timestamp': self.timestamp.isoformat()
        }


class RepairStrategy:
    """Base class for repair strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def can_repair(self, rubric: MetaRubric, diagnosis: DiagnosticReport) -> bool:
        """Check if this strategy can handle the diagnosis"""
        raise NotImplementedError
    
    def repair(self, rubric: MetaRubric, diagnosis: DiagnosticReport) -> RepairResult:
        """Execute the repair"""
        raise NotImplementedError


class ResetBaselineStrategy(RepairStrategy):
    """Strategy: Reset baseline to current state"""
    
    def __init__(self):
        super().__init__("ResetBaseline")
    
    def can_repair(self, rubric: MetaRubric, diagnosis: DiagnosticReport) -> bool:
        return RepairAction.RESET_BASELINE in diagnosis.recommendations
    
    def repair(self, rubric: MetaRubric, diagnosis: DiagnosticReport) -> RepairResult:
        before = {
            'drift_score': rubric.drift_score,
            'baseline_hash': rubric.baseline_hash
        }
        
        # Reset baseline
        rubric.set_baseline()
        
        after = {
            'drift_score': rubric.drift_score,
            'baseline_hash': rubric.baseline_hash
        }
        
        return RepairResult(
            rubric_id=rubric.rubric_id,
            action=RepairAction.RESET_BASELINE,
            outcome=RepairOutcome.SUCCESS,
            details="Baseline reset to current state",
            before_metrics=before,
            after_metrics=after
        )


class RestoreConfidenceStrategy(RepairStrategy):
    """Strategy: Restore confidence through validation"""
    
    def __init__(self, validation_boost: float = 0.2):
        super().__init__("RestoreConfidence")
        self.validation_boost = validation_boost
    
    def can_repair(self, rubric: MetaRubric, diagnosis: DiagnosticReport) -> bool:
        return RepairAction.RESTORE_CONFIDENCE in diagnosis.recommendations
    
    def repair(self, rubric: MetaRubric, diagnosis: DiagnosticReport) -> RepairResult:
        before = {'confidence_index': rubric.confidence_index}
        
        # Restore confidence with validation boost
        new_confidence = min(1.0, rubric.confidence_index + self.validation_boost)
        rubric.confidence_index = new_confidence
        
        # Log the repair
        rubric.evolution_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'action': 'confidence_repair',
            'old_confidence': before['confidence_index'],
            'new_confidence': new_confidence
        })
        
        after = {'confidence_index': rubric.confidence_index}
        
        return RepairResult(
            rubric_id=rubric.rubric_id,
            action=RepairAction.RESTORE_CONFIDENCE,
            outcome=RepairOutcome.SUCCESS if new_confidence >= 0.6 else RepairOutcome.PARTIAL,
            details=f"Confidence restored: {before['confidence_index']:.3f} -> {new_confidence:.3f}",
            before_metrics=before,
            after_metrics=after
        )


class PruneEvolutionStrategy(RepairStrategy):
    """Strategy: Prune problematic evolution history"""
    
    def __init__(self, prune_count: int = 3):
        super().__init__("PruneEvolution")
        self.prune_count = prune_count
    
    def can_repair(self, rubric: MetaRubric, diagnosis: DiagnosticReport) -> bool:
        return RepairAction.PRUNE_EVOLUTION in diagnosis.recommendations
    
    def repair(self, rubric: MetaRubric, diagnosis: DiagnosticReport) -> RepairResult:
        before = {'evolution_count': len(rubric.evolution_history)}
        
        # Prune recent problematic entries
        if len(rubric.evolution_history) > self.prune_count:
            # Remove last N entries
            rubric.evolution_history = rubric.evolution_history[:-self.prune_count]
            
            # Recalculate drift
            rubric.drift_score = len(rubric.evolution_history) * 0.05
        
        after = {
            'evolution_count': len(rubric.evolution_history),
            'drift_score': rubric.drift_score
        }
        
        return RepairResult(
            rubric_id=rubric.rubric_id,
            action=RepairAction.PRUNE_EVOLUTION,
            outcome=RepairOutcome.SUCCESS,
            details=f"Pruned {self.prune_count} evolution entries",
            before_metrics=before,
            after_metrics=after
        )


class QuarantineStrategy(RepairStrategy):
    """Strategy: Quarantine rubric (mark as untrusted)"""
    
    def __init__(self):
        super().__init__("Quarantine")
    
    def can_repair(self, rubric: MetaRubric, diagnosis: DiagnosticReport) -> bool:
        return RepairAction.QUARANTINE in diagnosis.recommendations
    
    def repair(self, rubric: MetaRubric, diagnosis: DiagnosticReport) -> RepairResult:
        before = {'confidence_index': rubric.confidence_index}
        
        # Quarantine by setting confidence to 0
        rubric.reclassify("quarantine_repair")
        
        after = {'confidence_index': rubric.confidence_index}
        
        return RepairResult(
            rubric_id=rubric.rubric_id,
            action=RepairAction.QUARANTINE,
            outcome=RepairOutcome.SUCCESS,
            details="Rubric quarantined",
            before_metrics=before,
            after_metrics=after
        )


# =============================================================================
# RUBRIC REPAIR ENGINE
# =============================================================================

class RubricRepairEngine:
    """
    Orchestrates rubric diagnosis and repair.
    
    Implements the rubric repair system from the specification:
    - Analyzes misfiring rubrics
    - Suggests corrections
    - Reconstructs historical rubric behavior
    - Automatically re-integrates verified rubrics
    """
    
    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        self.diagnostic = RubricDiagnostic()
        
        # Initialize repair strategies
        self.strategies: List[RepairStrategy] = [
            ResetBaselineStrategy(),
            RestoreConfidenceStrategy(),
            PruneEvolutionStrategy(),
            QuarantineStrategy()
        ]
        
        # Repair history
        self.repair_history: deque = deque(maxlen=1000)
        self.repair_count = 0
        self.success_count = 0
    
    def diagnose_and_repair(
        self, 
        rubric: MetaRubric,
        auto_repair: bool = True
    ) -> Tuple[DiagnosticReport, Optional[RepairResult]]:
        """
        Diagnose a rubric and optionally repair it.
        
        Returns (diagnosis, repair_result or None).
        """
        # Diagnose
        diagnosis = self.diagnostic.diagnose(rubric)
        
        # If healthy or no auto-repair, return diagnosis only
        if diagnosis.result == DiagnosticResult.HEALTHY or not auto_repair:
            return diagnosis, None
        
        # Find applicable repair strategy
        repair_result = None
        for strategy in self.strategies:
            if strategy.can_repair(rubric, diagnosis):
                repair_result = strategy.repair(rubric, diagnosis)
                self.repair_history.append(repair_result)
                self.repair_count += 1
                if repair_result.outcome == RepairOutcome.SUCCESS:
                    self.success_count += 1
                break
        
        return diagnosis, repair_result
    
    def repair_model(
        self, 
        model: SelfModel,
        auto_repair: bool = True
    ) -> Dict[str, Any]:
        """
        Diagnose and repair all rubrics in a self-model.
        
        Returns comprehensive report.
        """
        reports = []
        repairs = []
        
        for rubric in model.rubrics.values():
            diagnosis, repair = self.diagnose_and_repair(rubric, auto_repair)
            reports.append(diagnosis)
            if repair:
                repairs.append(repair)
        
        # Summary statistics
        healthy_count = sum(1 for r in reports if r.result == DiagnosticResult.HEALTHY)
        critical_count = sum(1 for r in reports if r.result == DiagnosticResult.CRITICAL_FAILURE)
        repair_success = sum(1 for r in repairs if r.outcome == RepairOutcome.SUCCESS)
        
        return {
            'model_id': model.capsule_id,
            'timestamp': datetime.utcnow().isoformat(),
            'rubric_count': len(model.rubrics),
            'healthy_count': healthy_count,
            'critical_count': critical_count,
            'repairs_attempted': len(repairs),
            'repairs_successful': repair_success,
            'health_ratio': healthy_count / len(model.rubrics) if model.rubrics else 1.0,
            'reports': [r.to_dict() for r in reports],
            'repairs': [r.to_dict() for r in repairs]
        }
    
    def reconstruct_history(self, rubric: MetaRubric) -> Dict[str, Any]:
        """
        Reconstruct historical rubric behavior.
        
        Analyzes evolution history to understand rubric trajectory.
        """
        history = rubric.evolution_history
        
        if not history:
            return {
                'rubric_id': rubric.rubric_id,
                'trajectory': 'stable',
                'confidence_trend': 'none',
                'events': []
            }
        
        # Analyze confidence trend
        confidence_deltas = [e.get('confidence_delta', 0) for e in history]
        avg_delta = np.mean(confidence_deltas) if confidence_deltas else 0
        
        if avg_delta > 0.05:
            confidence_trend = 'improving'
        elif avg_delta < -0.05:
            confidence_trend = 'declining'
        else:
            confidence_trend = 'stable'
        
        # Identify significant events
        events = []
        for i, entry in enumerate(history):
            if entry.get('action') in ['reclassify', 'confidence_repair']:
                events.append({
                    'index': i,
                    'action': entry.get('action'),
                    'timestamp': entry.get('timestamp')
                })
        
        # Determine overall trajectory
        if rubric.confidence_index < 0.3:
            trajectory = 'degraded'
        elif rubric.drift_score > MAX_RUBRIC_DRIFT:
            trajectory = 'drifted'
        elif confidence_trend == 'declining':
            trajectory = 'at_risk'
        else:
            trajectory = 'healthy'
        
        return {
            'rubric_id': rubric.rubric_id,
            'trajectory': trajectory,
            'confidence_trend': confidence_trend,
            'avg_confidence_delta': float(avg_delta),
            'evolution_count': len(history),
            'significant_events': events,
            'current_drift': rubric.drift_score,
            'current_confidence': rubric.confidence_index
        }
    
    def suggest_corrections(self, rubric: MetaRubric) -> List[Dict[str, Any]]:
        """Suggest corrections for a rubric"""
        diagnosis = self.diagnostic.diagnose(rubric)
        history = self.reconstruct_history(rubric)
        
        suggestions = []
        
        for action in diagnosis.recommendations:
            if action == RepairAction.NO_ACTION:
                continue
            
            suggestion = {
                'action': action.name,
                'priority': 'high' if diagnosis.severity >= 0.6 else 'medium',
                'rationale': '',
                'impact': ''
            }
            
            if action == RepairAction.RESET_BASELINE:
                suggestion['rationale'] = f"Drift score ({rubric.drift_score:.3f}) exceeds threshold"
                suggestion['impact'] = "Will reset drift tracking; loses historical comparison"
            
            elif action == RepairAction.RESTORE_CONFIDENCE:
                suggestion['rationale'] = f"Confidence ({rubric.confidence_index:.3f}) is low"
                suggestion['impact'] = "Will boost confidence; requires validation"
            
            elif action == RepairAction.PRUNE_EVOLUTION:
                suggestion['rationale'] = f"Evolution shows {history['confidence_trend']} trend"
                suggestion['impact'] = "Will remove recent problematic entries"
            
            elif action == RepairAction.QUARANTINE:
                suggestion['rationale'] = f"Critical issues detected: {diagnosis.issues}"
                suggestion['impact'] = "Rubric will be disabled until manual review"
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'engine_id': self.engine_id,
            'total_repairs': self.repair_count,
            'successful_repairs': self.success_count,
            'success_rate': self.success_count / self.repair_count if self.repair_count > 0 else 1.0,
            'strategies_available': len(self.strategies),
            'diagnostic_history_size': len(self.diagnostic.diagnostic_history),
            'repair_history_size': len(self.repair_history)
        }


# =============================================================================
# DSL PREDICATES
# =============================================================================

def rubric_needs_repair(engine: RubricRepairEngine, rubric: MetaRubric) -> bool:
    """DSL predicate: Check if rubric needs repair"""
    diagnosis = engine.diagnostic.diagnose(rubric)
    return diagnosis.result != DiagnosticResult.HEALTHY


def auto_repair_rubric(engine: RubricRepairEngine, rubric: MetaRubric) -> bool:
    """DSL predicate: Automatically repair rubric if needed"""
    _, repair = engine.diagnose_and_repair(rubric, auto_repair=True)
    return repair is not None and repair.outcome == RepairOutcome.SUCCESS


def rubric_trajectory(engine: RubricRepairEngine, rubric: MetaRubric) -> str:
    """DSL predicate: Get rubric trajectory"""
    history = engine.reconstruct_history(rubric)
    return history['trajectory']


def model_health_ratio(engine: RubricRepairEngine, model: SelfModel) -> float:
    """DSL predicate: Get model health ratio"""
    report = engine.repair_model(model, auto_repair=False)
    return report['health_ratio']


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BOOKLET 8: RUBRIC REPAIR ENGINE DEMO")
    print("=" * 70)
    print()
    
    # Create engine
    engine = RubricRepairEngine("REPAIR-001")
    print(f"Created repair engine: {engine.engine_id}")
    print()
    
    # Create a rubric with issues
    rubric = MetaRubric(
        rubric_id="DRIFT-RUBRIC",
        name="Drifting Rubric",
        description="A rubric that has drifted"
    )
    rubric.set_baseline()
    
    # Simulate drift
    for i in range(10):
        rubric.update({'confidence_index': rubric.confidence_index - 0.05})
    
    print(f"Created rubric with drift: {rubric.drift_score:.3f}")
    print(f"Confidence: {rubric.confidence_index:.3f}")
    print()
    
    # Diagnose
    print("Diagnosing rubric...")
    diagnosis = engine.diagnostic.diagnose(rubric)
    print(f"  Result: {diagnosis.result.name}")
    print(f"  Severity: {diagnosis.severity:.3f}")
    print(f"  Issues: {diagnosis.issues}")
    print(f"  Recommendations: {[r.name for r in diagnosis.recommendations]}")
    print()
    
    # Get suggestions
    print("Getting repair suggestions...")
    suggestions = engine.suggest_corrections(rubric)
    for s in suggestions:
        print(f"  {s['action']} ({s['priority']}): {s['rationale']}")
    print()
    
    # Auto-repair
    print("Performing auto-repair...")
    diagnosis, repair = engine.diagnose_and_repair(rubric, auto_repair=True)
    if repair:
        print(f"  Action: {repair.action.name}")
        print(f"  Outcome: {repair.outcome.name}")
        print(f"  Details: {repair.details}")
    print()
    
    # Reconstruct history
    print("Reconstructing history...")
    history = engine.reconstruct_history(rubric)
    print(f"  Trajectory: {history['trajectory']}")
    print(f"  Confidence trend: {history['confidence_trend']}")
    print()
    
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
