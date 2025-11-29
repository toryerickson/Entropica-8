"""
RSCS-Q Booklet 8: Drift-Debt Governance
=======================================

Implements drift-debt tracking for repair governance to prevent
"papering over" problems with superficial repairs.

Features:
- Cumulative debt tracking per capsule
- Repair cost assignment by strategy
- Cool-off periods after repairs
- Counterfactual validation
- Budget enforcement with quarantine

Author: Entropica Research Collective
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import deque
from enum import Enum, auto


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_BUDGET = 10.0
MAX_BUDGET = 15.0
COOLING_PERIOD_TICKS = 5
DECAY_RATE_PER_TICK = 0.05
MIN_UPLIFT_REQUIRED = 0.1
REPLAY_PERCENTAGE = 10


# =============================================================================
# REPAIR COST TABLE
# =============================================================================

REPAIR_COSTS = {
    'reset_baseline': 1.0,
    'restore_confidence': 1.5,
    'prune_evolution': 2.0,
    'quarantine': 0.5,  # Low cost - it's containment, not repair
    'rollback': 2.5,
    'full_rebuild': 5.0
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DebtRecord:
    """Record of a single debt-incurring event"""
    timestamp: datetime
    capsule_id: str
    repair_type: str
    cost: float
    pre_drift: float
    post_drift: float
    counterfactual_validated: bool = False
    uplift: float = 0.0


@dataclass
class CapsuleDebtState:
    """Debt state for a single capsule"""
    capsule_id: str
    current_debt: float = 0.0
    total_repairs: int = 0
    last_repair_tick: int = -999  # Long ago
    in_cooling: bool = False
    quarantine_triggered: bool = False
    history: List[DebtRecord] = field(default_factory=list)


class DebtBudgetStatus(Enum):
    """Status of debt budget"""
    HEALTHY = auto()      # Well below budget
    WARNING = auto()      # Approaching budget
    CRITICAL = auto()     # At or near budget
    EXCEEDED = auto()     # Budget exceeded, quarantine


# =============================================================================
# DRIFT-DEBT LEDGER
# =============================================================================

class DriftDebtLedger:
    """
    Tracks cumulative drift-debt across capsules.
    
    Drift-debt rises on each repair operation and naturally decays over time.
    When debt exceeds budget, the capsule is quarantined.
    """
    
    def __init__(
        self,
        ledger_id: str,
        initial_budget: float = DEFAULT_BUDGET,
        max_budget: float = MAX_BUDGET,
        decay_rate: float = DECAY_RATE_PER_TICK,
        cooling_period: int = COOLING_PERIOD_TICKS
    ):
        self.ledger_id = ledger_id
        self.initial_budget = initial_budget
        self.max_budget = max_budget
        self.decay_rate = decay_rate
        self.cooling_period = cooling_period
        
        self.capsule_states: Dict[str, CapsuleDebtState] = {}
        self.current_tick = 0
        self.global_statistics = {
            'total_repairs': 0,
            'total_debt_incurred': 0.0,
            'total_quarantines': 0,
            'counterfactual_validations': 0,
            'failed_validations': 0
        }
    
    def get_or_create_state(self, capsule_id: str) -> CapsuleDebtState:
        """Get or create debt state for a capsule"""
        if capsule_id not in self.capsule_states:
            self.capsule_states[capsule_id] = CapsuleDebtState(capsule_id=capsule_id)
        return self.capsule_states[capsule_id]
    
    def tick(self) -> None:
        """Advance time by one tick, applying debt decay"""
        self.current_tick += 1
        
        for state in self.capsule_states.values():
            # Apply decay
            if state.current_debt > 0:
                state.current_debt = max(0, state.current_debt - self.decay_rate)
            
            # Check cooling period
            if state.in_cooling:
                ticks_since_repair = self.current_tick - state.last_repair_tick
                if ticks_since_repair >= self.cooling_period:
                    state.in_cooling = False
    
    def can_repair(self, capsule_id: str, repair_type: str, emergency: bool = False) -> Tuple[bool, str]:
        """
        Check if a repair is allowed for a capsule.
        
        Returns (allowed, reason)
        """
        state = self.get_or_create_state(capsule_id)
        cost = REPAIR_COSTS.get(repair_type, 1.0)
        
        # Check quarantine
        if state.quarantine_triggered:
            return False, "capsule_quarantined_for_debt"
        
        # Check cooling period (unless emergency)
        if state.in_cooling and not emergency:
            return False, "cooling_period_active"
        
        # Check budget
        projected_debt = state.current_debt + cost
        if projected_debt > self.max_budget:
            return False, "would_exceed_budget"
        
        return True, "allowed"
    
    def record_repair(
        self,
        capsule_id: str,
        repair_type: str,
        pre_drift: float,
        post_drift: float,
        force: bool = False
    ) -> Tuple[bool, DebtRecord]:
        """
        Record a repair operation and incur debt.
        
        Returns (success, debt_record)
        """
        state = self.get_or_create_state(capsule_id)
        cost = REPAIR_COSTS.get(repair_type, 1.0)
        
        # Check if allowed
        if not force:
            allowed, reason = self.can_repair(capsule_id, repair_type)
            if not allowed:
                return False, None
        
        # Create record
        record = DebtRecord(
            timestamp=datetime.utcnow(),
            capsule_id=capsule_id,
            repair_type=repair_type,
            cost=cost,
            pre_drift=pre_drift,
            post_drift=post_drift,
            uplift=pre_drift - post_drift
        )
        
        # Update state
        state.current_debt += cost
        state.total_repairs += 1
        state.last_repair_tick = self.current_tick
        state.in_cooling = True
        state.history.append(record)
        
        # Update global stats
        self.global_statistics['total_repairs'] += 1
        self.global_statistics['total_debt_incurred'] += cost
        
        # Check if budget exceeded
        if state.current_debt >= self.max_budget:
            state.quarantine_triggered = True
            self.global_statistics['total_quarantines'] += 1
        
        return True, record
    
    def validate_counterfactual(
        self,
        capsule_id: str,
        record_index: int,
        counterfactual_drift: float
    ) -> Tuple[bool, float]:
        """
        Validate that a repair actually helped vs. doing nothing.
        
        Returns (passed, actual_uplift)
        """
        state = self.get_or_create_state(capsule_id)
        
        if record_index >= len(state.history):
            return False, 0.0
        
        record = state.history[record_index]
        
        # Compare actual post-drift to counterfactual (what would have happened)
        actual_uplift = record.pre_drift - record.post_drift
        counterfactual_uplift = record.pre_drift - counterfactual_drift
        
        # Repair is valid if it did better than doing nothing
        relative_improvement = actual_uplift - counterfactual_uplift
        passed = relative_improvement >= MIN_UPLIFT_REQUIRED
        
        record.counterfactual_validated = True
        record.uplift = actual_uplift
        
        self.global_statistics['counterfactual_validations'] += 1
        if not passed:
            self.global_statistics['failed_validations'] += 1
        
        return passed, actual_uplift
    
    def get_budget_status(self, capsule_id: str) -> DebtBudgetStatus:
        """Get budget status for a capsule"""
        state = self.get_or_create_state(capsule_id)
        
        ratio = state.current_debt / self.max_budget
        
        if state.quarantine_triggered or ratio >= 1.0:
            return DebtBudgetStatus.EXCEEDED
        elif ratio >= 0.8:
            return DebtBudgetStatus.CRITICAL
        elif ratio >= 0.5:
            return DebtBudgetStatus.WARNING
        else:
            return DebtBudgetStatus.HEALTHY
    
    def get_capsule_report(self, capsule_id: str) -> Dict[str, Any]:
        """Get detailed report for a capsule"""
        state = self.get_or_create_state(capsule_id)
        
        return {
            'capsule_id': capsule_id,
            'current_debt': state.current_debt,
            'budget_ratio': state.current_debt / self.max_budget,
            'budget_status': self.get_budget_status(capsule_id).name,
            'total_repairs': state.total_repairs,
            'in_cooling': state.in_cooling,
            'quarantine_triggered': state.quarantine_triggered,
            'history_length': len(state.history),
            'avg_uplift': np.mean([r.uplift for r in state.history]) if state.history else 0.0
        }
    
    def get_global_report(self) -> Dict[str, Any]:
        """Get global ledger report"""
        capsule_count = len(self.capsule_states)
        
        if capsule_count == 0:
            return {
                'ledger_id': self.ledger_id,
                'capsule_count': 0,
                'statistics': self.global_statistics
            }
        
        total_debt = sum(s.current_debt for s in self.capsule_states.values())
        quarantined = sum(1 for s in self.capsule_states.values() if s.quarantine_triggered)
        in_cooling = sum(1 for s in self.capsule_states.values() if s.in_cooling)
        
        by_status = {status.name: 0 for status in DebtBudgetStatus}
        for capsule_id in self.capsule_states:
            status = self.get_budget_status(capsule_id)
            by_status[status.name] += 1
        
        return {
            'ledger_id': self.ledger_id,
            'current_tick': self.current_tick,
            'capsule_count': capsule_count,
            'total_debt': total_debt,
            'avg_debt': total_debt / capsule_count,
            'quarantined_count': quarantined,
            'in_cooling_count': in_cooling,
            'by_status': by_status,
            'statistics': self.global_statistics
        }
    
    def reset_capsule(self, capsule_id: str) -> None:
        """Reset a capsule's debt state (e.g., after full recovery)"""
        if capsule_id in self.capsule_states:
            state = self.capsule_states[capsule_id]
            state.current_debt = 0.0
            state.in_cooling = False
            state.quarantine_triggered = False
            # Keep history for audit


# =============================================================================
# GOVERNED REPAIR ENGINE WRAPPER
# =============================================================================

class GovernedRepairEngine:
    """
    Repair engine wrapper with drift-debt governance.
    
    Wraps the base RubricRepairEngine with debt tracking.
    """
    
    def __init__(
        self,
        engine_id: str,
        base_engine: 'RubricRepairEngine' = None
    ):
        self.engine_id = engine_id
        self.ledger = DriftDebtLedger(f"{engine_id}-LEDGER")
        
        # Import here to avoid circular dependency
        if base_engine is None:
            from rubric_repair import RubricRepairEngine
            self.base_engine = RubricRepairEngine(engine_id)
        else:
            self.base_engine = base_engine
    
    def diagnose_and_repair(
        self,
        rubric: 'MetaRubric',
        capsule_id: str,
        auto_repair: bool = True,
        emergency: bool = False
    ) -> Tuple[Any, Any, bool]:
        """
        Diagnose and optionally repair with debt governance.
        
        Returns (diagnosis, repair_result, debt_recorded)
        """
        # Get pre-repair drift
        pre_drift = rubric.drift_score
        
        # Check if repair allowed
        if auto_repair:
            allowed, reason = self.ledger.can_repair(
                capsule_id, "restore_confidence", emergency
            )
            if not allowed:
                # Return diagnosis only, no repair
                diagnosis, _ = self.base_engine.diagnose_and_repair(
                    rubric, auto_repair=False
                )
                return diagnosis, None, False
        
        # Perform diagnosis and repair
        diagnosis, repair_result = self.base_engine.diagnose_and_repair(
            rubric, auto_repair=auto_repair
        )
        
        # Record debt if repair occurred
        debt_recorded = False
        if repair_result and repair_result.success:
            post_drift = rubric.drift_score
            repair_type = repair_result.action.name.lower() if repair_result.action else "unknown"
            
            success, record = self.ledger.record_repair(
                capsule_id,
                repair_type,
                pre_drift,
                post_drift
            )
            debt_recorded = success
        
        return diagnosis, repair_result, debt_recorded
    
    def tick(self) -> None:
        """Advance time"""
        self.ledger.tick()
    
    def get_status(self, capsule_id: str) -> Dict[str, Any]:
        """Get status for a capsule"""
        return {
            'capsule_report': self.ledger.get_capsule_report(capsule_id),
            'can_repair': self.ledger.can_repair(capsule_id, "restore_confidence")
        }
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global status"""
        return {
            'engine_id': self.engine_id,
            'ledger': self.ledger.get_global_report(),
            'base_engine_stats': self.base_engine.statistics
        }


# =============================================================================
# DSL PREDICATES
# =============================================================================

def debt_allows_repair(ledger: DriftDebtLedger, capsule_id: str, repair_type: str) -> bool:
    """Check if debt budget allows a repair"""
    allowed, _ = ledger.can_repair(capsule_id, repair_type)
    return allowed


def debt_status_healthy(ledger: DriftDebtLedger, capsule_id: str) -> bool:
    """Check if capsule's debt status is healthy"""
    return ledger.get_budget_status(capsule_id) == DebtBudgetStatus.HEALTHY


def debt_quarantine_triggered(ledger: DriftDebtLedger, capsule_id: str) -> bool:
    """Check if capsule has been quarantined due to debt"""
    state = ledger.get_or_create_state(capsule_id)
    return state.quarantine_triggered


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DRIFT-DEBT GOVERNANCE DEMO")
    print("=" * 70)
    print()
    
    np.random.seed(42)
    
    # Create ledger
    ledger = DriftDebtLedger("DEMO-LEDGER", initial_budget=10.0, max_budget=15.0)
    
    # Simulate repairs
    capsule_id = "CAP-001"
    
    print(">>> Simulating repair sequence")
    for i in range(12):
        pre_drift = np.random.uniform(0.3, 0.5)
        post_drift = np.random.uniform(0.1, 0.3)
        
        allowed, reason = ledger.can_repair(capsule_id, "restore_confidence")
        
        if allowed:
            success, record = ledger.record_repair(
                capsule_id, "restore_confidence", pre_drift, post_drift
            )
            print(f"  Repair {i+1}: success={success}, debt={ledger.capsule_states[capsule_id].current_debt:.2f}")
        else:
            print(f"  Repair {i+1}: BLOCKED - {reason}")
        
        # Tick for decay
        ledger.tick()
    
    print()
    print(">>> Capsule Report")
    report = ledger.get_capsule_report(capsule_id)
    for k, v in report.items():
        print(f"  {k}: {v}")
    
    print()
    print(">>> Global Report")
    global_report = ledger.get_global_report()
    print(f"  Total repairs: {global_report['statistics']['total_repairs']}")
    print(f"  Total debt incurred: {global_report['statistics']['total_debt_incurred']:.2f}")
    print(f"  Quarantines: {global_report['statistics']['total_quarantines']}")
    
    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
