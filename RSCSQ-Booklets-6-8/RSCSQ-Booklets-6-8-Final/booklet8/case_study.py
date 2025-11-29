"""
RSCS-Q Booklet 8: Simulation Case Study
=======================================

Demonstrates a complete self-modeling scenario:
- Branching mission tree with capsule families
- Drift cascade from one family
- Reflective swarm isolation and repair
- Measured outcome metrics

Author: Entropica Research Collective
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from self_model import (
    SelfModel, MetaRubric, SelfModelState, ModificationType,
    MAX_RUBRIC_DRIFT, self_model_valid, identity_coherent,
    reset_validity_state
)
from rubric_repair import (
    RubricRepairEngine, DiagnosticResult, RepairOutcome,
    rubric_needs_repair, auto_repair_rubric
)
from reflex_log import (
    ReflexLog, ObserverMesh, Observer, RuntimeMetrics,
    EventType, CollapseValidity
)
from visual_toolkit import B8Visualizer, RecoveryTimeline


# =============================================================================
# CASE STUDY: DRIFT CASCADE AND RECOVERY
# =============================================================================

@dataclass
class CaseStudyConfig:
    """Configuration for case study"""
    # Tree structure
    num_families: int = 3
    depth_per_family: int = 3
    children_per_node: int = 2
    
    # Drift injection
    infected_family: int = 1  # Which family to infect
    drift_severity: float = 0.45  # Initial drift level
    cascade_factor: float = 0.8  # Drift propagation factor
    
    # Recovery
    detection_threshold: float = 0.3
    repair_success_rate: float = 0.7
    
    # Timing
    simulation_ticks: int = 20
    
    # Random seed
    random_seed: int = 42


@dataclass
class CaseStudyResults:
    """Results from case study"""
    total_capsules: int = 0
    infected_capsules: int = 0
    recovered_capsules: int = 0
    quarantined_capsules: int = 0
    
    detection_time: int = 0  # Ticks until first detection
    containment_time: int = 0  # Ticks until cascade stopped
    recovery_time: int = 0  # Ticks until full recovery
    
    drift_suppression_score: float = 0.0  # How much drift was reduced
    rubric_integrity_ratio: float = 0.0  # Healthy rubrics / total
    
    cascade_depth: int = 0  # How far drift propagated
    repairs_attempted: int = 0
    repairs_successful: int = 0
    
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_capsules': self.total_capsules,
            'infected_capsules': self.infected_capsules,
            'recovered_capsules': self.recovered_capsules,
            'quarantined_capsules': self.quarantined_capsules,
            'detection_time': self.detection_time,
            'containment_time': self.containment_time,
            'recovery_time': self.recovery_time,
            'drift_suppression_score': self.drift_suppression_score,
            'rubric_integrity_ratio': self.rubric_integrity_ratio,
            'cascade_depth': self.cascade_depth,
            'repairs_attempted': self.repairs_attempted,
            'repairs_successful': self.repairs_successful,
            'timeline_length': len(self.timeline)
        }


class DriftCascadeSimulation:
    """
    Simulates a drift cascade through a capsule family tree.
    
    Scenario:
    1. Create a tree of capsule families
    2. Inject drift into one family's root
    3. Watch drift cascade to children
    4. Observe detection, isolation, and repair
    5. Measure recovery metrics
    """
    
    def __init__(self, config: CaseStudyConfig = None):
        self.config = config or CaseStudyConfig()
        np.random.seed(self.config.random_seed)
        
        # Components
        self.models: Dict[str, SelfModel] = {}
        self.families: Dict[int, List[str]] = {}  # family_id -> capsule_ids
        self.repair_engine = RubricRepairEngine("CASE-REPAIR")
        self.log = ReflexLog("CASE-LOG")
        self.mesh = ObserverMesh("CASE-MESH")
        self.visualizer = B8Visualizer()
        
        # State tracking
        self.current_tick = 0
        self.infected: set = set()
        self.detected: set = set()
        self.quarantined: set = set()
        self.recovered: set = set()
        
        # Results
        self.results = CaseStudyResults()
    
    def setup(self) -> None:
        """Build the capsule tree"""
        reset_validity_state()
        
        # Add observers
        for i in range(5):
            self.mesh.add_observer(Observer(f"OBS-{i}"))
        
        # Create families
        for family_id in range(self.config.num_families):
            root_id = f"FAM{family_id}-ROOT"
            root = self._create_capsule(root_id, None, 0, family_id)
            self.families[family_id] = [root_id]
            
            # Create tree
            self._create_subtree(root, 1, family_id)
        
        self.results.total_capsules = len(self.models)
        
        # Take initial snapshot
        self.visualizer.ingest_models(list(self.models.values()))
    
    def _create_capsule(
        self, 
        capsule_id: str, 
        parent_id: Optional[str], 
        depth: int,
        family_id: int
    ) -> SelfModel:
        """Create a single capsule"""
        if parent_id and parent_id in self.models:
            parent = self.models[parent_id]
            model = parent.spawn_child(capsule_id)
            if model is None:
                model = SelfModel(capsule_id, parent_id, depth)
        else:
            model = SelfModel(capsule_id, parent_id, depth)
        
        # Add rubric
        rubric = MetaRubric(
            rubric_id=f"{capsule_id}-RUB",
            name=f"Alignment for {capsule_id}",
            description="Family alignment rubric"
        )
        model.add_rubric(rubric)
        model.set_alignment_anchor(rubric.rubric_id)
        
        # Set initial healthy metrics
        model.evi_score = np.random.uniform(0.7, 0.95)
        model.drift_score = np.random.uniform(0.0, 0.1)
        
        self.models[capsule_id] = model
        return model
    
    def _create_subtree(self, parent: SelfModel, depth: int, family_id: int) -> None:
        """Recursively create subtree"""
        if depth >= self.config.depth_per_family:
            return
        
        for i in range(self.config.children_per_node):
            child_id = f"{parent.capsule_id}-C{i}"
            child = self._create_capsule(child_id, parent.capsule_id, depth, family_id)
            self.families[family_id].append(child_id)
            self._create_subtree(child, depth + 1, family_id)
    
    def inject_drift(self) -> None:
        """Inject drift into the target family"""
        target_family = self.config.infected_family
        if target_family not in self.families:
            return
        
        # Infect root of target family
        root_id = self.families[target_family][0]
        root = self.models[root_id]
        
        root.drift_score = self.config.drift_severity
        for rubric in root.rubrics.values():
            rubric.drift_score = self.config.drift_severity
            rubric.confidence_index = 0.4
        
        self.infected.add(root_id)
        self._log_event('drift_injected', root_id, {'severity': self.config.drift_severity})
        
        self.visualizer.timeline.record_drift_detected(root_id, self.config.drift_severity)
    
    def propagate_drift(self) -> int:
        """Propagate drift to children (cascade)"""
        newly_infected = []
        
        for capsule_id in list(self.infected):
            model = self.models[capsule_id]
            
            # Spread to children
            for child_id in model.children:
                if child_id in self.models and child_id not in self.infected:
                    child = self.models[child_id]
                    
                    # Cascade with attenuation
                    inherited_drift = model.drift_score * self.config.cascade_factor
                    if inherited_drift > 0.1:  # Threshold for propagation
                        child.drift_score = inherited_drift
                        for rubric in child.rubrics.values():
                            rubric.drift_score = inherited_drift
                            rubric.confidence_index = max(0.3, rubric.confidence_index - 0.2)
                        
                        newly_infected.append(child_id)
        
        for cid in newly_infected:
            self.infected.add(cid)
            self._log_event('cascade_spread', cid, {'source': 'parent'})
        
        return len(newly_infected)
    
    def detect_anomalies(self) -> List[str]:
        """Detect capsules with anomalous drift"""
        detected_this_tick = []
        
        for capsule_id, model in self.models.items():
            if capsule_id in self.detected or capsule_id in self.quarantined:
                continue
            
            if model.drift_score > self.config.detection_threshold:
                self.detected.add(capsule_id)
                detected_this_tick.append(capsule_id)
                self._log_event('anomaly_detected', capsule_id, {'drift': model.drift_score})
                
                if self.results.detection_time == 0:
                    self.results.detection_time = self.current_tick
        
        return detected_this_tick
    
    def attempt_repairs(self) -> Tuple[int, int]:
        """Attempt to repair detected capsules"""
        attempted = 0
        successful = 0
        
        for capsule_id in list(self.detected):
            if capsule_id in self.quarantined or capsule_id in self.recovered:
                continue
            
            model = self.models[capsule_id]
            
            # Try to repair each rubric
            for rubric in model.rubrics.values():
                if rubric_needs_repair(self.repair_engine, rubric):
                    attempted += 1
                    self.results.repairs_attempted += 1
                    
                    # Simulate repair with configured success rate
                    if np.random.random() < self.config.repair_success_rate:
                        success = auto_repair_rubric(self.repair_engine, rubric)
                        if success:
                            successful += 1
                            self.results.repairs_successful += 1
                            model.drift_score = max(0, model.drift_score - 0.2)
                            
                            self._log_event('repair_success', capsule_id, {
                                'rubric_id': rubric.rubric_id,
                                'new_drift': model.drift_score
                            })
                            
                            self.visualizer.record_repair(
                                capsule_id, 'ResetBaseline', True, model.drift_score
                            )
            
            # Check if recovered
            if model.drift_score < self.config.detection_threshold:
                self.recovered.add(capsule_id)
                self.results.recovered_capsules += 1
                self._log_event('capsule_recovered', capsule_id, {})
                self.visualizer.timeline.record_recovery(capsule_id)
        
        return attempted, successful
    
    def quarantine_critical(self) -> int:
        """Quarantine capsules that couldn't be repaired"""
        quarantined_count = 0
        
        for capsule_id in list(self.detected):
            if capsule_id in self.quarantined or capsule_id in self.recovered:
                continue
            
            model = self.models[capsule_id]
            
            # Quarantine if drift is still high after multiple ticks
            if model.drift_score > MAX_RUBRIC_DRIFT:
                model.quarantine("persistent high drift")
                self.quarantined.add(capsule_id)
                quarantined_count += 1
                self.results.quarantined_capsules += 1
                
                self._log_event('capsule_quarantined', capsule_id, {'drift': model.drift_score})
                self.visualizer.timeline.record_quarantine(capsule_id, "persistent high drift")
        
        return quarantined_count
    
    def run_tick(self) -> Dict[str, Any]:
        """Run a single simulation tick"""
        self.current_tick += 1
        tick_events = {
            'tick': self.current_tick,
            'propagated': 0,
            'detected': [],
            'repairs': (0, 0),
            'quarantined': 0
        }
        
        # Phase 1: Propagate drift
        tick_events['propagated'] = self.propagate_drift()
        
        # Phase 2: Detect anomalies
        tick_events['detected'] = self.detect_anomalies()
        
        # Phase 3: Attempt repairs
        tick_events['repairs'] = self.attempt_repairs()
        
        # Phase 4: Quarantine critical cases
        tick_events['quarantined'] = self.quarantine_critical()
        
        # Check containment
        if tick_events['propagated'] == 0 and self.results.containment_time == 0 and self.current_tick > 1:
            self.results.containment_time = self.current_tick
        
        # Check full recovery
        remaining = self.infected - self.recovered - self.quarantined
        if len(remaining) == 0 and self.results.recovery_time == 0 and self.current_tick > 1:
            self.results.recovery_time = self.current_tick
        
        # Record snapshot
        self.visualizer.heatmap.record_snapshot(list(self.models.values()))
        
        self.results.timeline.append(tick_events)
        return tick_events
    
    def run_simulation(self, verbose: bool = True) -> CaseStudyResults:
        """Run the full simulation"""
        if verbose:
            print("=" * 70)
            print("CASE STUDY: DRIFT CASCADE AND RECOVERY")
            print("=" * 70)
            print()
        
        # Setup
        self.setup()
        if verbose:
            print(f"Setup complete: {self.results.total_capsules} capsules in {len(self.families)} families")
        
        # Inject drift
        self.inject_drift()
        if verbose:
            print(f"Drift injected into family {self.config.infected_family}")
            print()
        
        # Run simulation
        for tick in range(self.config.simulation_ticks):
            events = self.run_tick()
            
            if verbose:
                print(f"Tick {events['tick']:2d}: prop={events['propagated']}, det={len(events['detected'])}, "
                      f"repairs={events['repairs']}, quar={events['quarantined']}")
            
            # Early termination if everything resolved
            if self.results.recovery_time > 0:
                if verbose:
                    print("  >> Full recovery achieved!")
                break
        
        # Compute final metrics
        self._compute_final_metrics()
        
        if verbose:
            print()
            self._print_results()
        
        return self.results
    
    def _compute_final_metrics(self) -> None:
        """Compute final metrics"""
        self.results.infected_capsules = len(self.infected)
        
        # Drift suppression: how much drift was reduced from max
        initial_drift = self.config.drift_severity
        final_drifts = [self.models[cid].drift_score for cid in self.infected 
                       if cid not in self.quarantined]
        avg_final = np.mean(final_drifts) if final_drifts else 0
        self.results.drift_suppression_score = 1 - (avg_final / initial_drift) if initial_drift > 0 else 1.0
        
        # Rubric integrity
        total_rubrics = 0
        healthy_rubrics = 0
        for model in self.models.values():
            for rubric in model.rubrics.values():
                total_rubrics += 1
                if rubric.drift_score <= MAX_RUBRIC_DRIFT:
                    healthy_rubrics += 1
        self.results.rubric_integrity_ratio = healthy_rubrics / total_rubrics if total_rubrics > 0 else 1.0
        
        # Cascade depth: max depth infected
        max_depth = 0
        for cid in self.infected:
            model = self.models[cid]
            max_depth = max(max_depth, model.lineage_depth)
        self.results.cascade_depth = max_depth
    
    def _log_event(self, event_type: str, capsule_id: str, details: Dict[str, Any]) -> None:
        """Log an event to ReflexLog"""
        self.log.emit(
            capsule_id=capsule_id,
            event_type=EventType.ALERT,
            context={'event': event_type, **details}
        )
    
    def _print_results(self) -> None:
        """Print formatted results"""
        print("=" * 70)
        print("CASE STUDY RESULTS")
        print("=" * 70)
        print()
        
        print(">>> INFECTION METRICS")
        print(f"  Total capsules: {self.results.total_capsules}")
        print(f"  Infected: {self.results.infected_capsules}")
        print(f"  Cascade depth: {self.results.cascade_depth}")
        print()
        
        print(">>> RECOVERY METRICS")
        print(f"  Detection time: {self.results.detection_time} ticks")
        print(f"  Containment time: {self.results.containment_time} ticks")
        print(f"  Recovery time: {self.results.recovery_time} ticks")
        print()
        
        print(">>> REPAIR METRICS")
        print(f"  Repairs attempted: {self.results.repairs_attempted}")
        print(f"  Repairs successful: {self.results.repairs_successful}")
        print(f"  Recovered capsules: {self.results.recovered_capsules}")
        print(f"  Quarantined capsules: {self.results.quarantined_capsules}")
        print()
        
        print(">>> QUALITY METRICS")
        print(f"  Drift suppression score: {self.results.drift_suppression_score:.2%}")
        print(f"  Rubric integrity ratio: {self.results.rubric_integrity_ratio:.2%}")
        print()
        
        # Assessment
        success = (
            self.results.containment_time > 0 and
            self.results.drift_suppression_score > 0.5 and
            self.results.rubric_integrity_ratio > 0.7
        )
        
        if success:
            print("✅ CASE STUDY: SUCCESS")
            print("   System demonstrated effective drift detection, containment, and recovery.")
        else:
            print("⚠️  CASE STUDY: PARTIAL SUCCESS")
            print("   Some metrics below target; system shows areas for improvement.")
        
        print("=" * 70)
    
    def export_report(self) -> str:
        """Export comprehensive report"""
        lines = [
            "BOOKLET 8 CASE STUDY: DRIFT CASCADE AND RECOVERY",
            "=" * 70,
            "",
            "CONFIGURATION",
            "-" * 40,
            f"Families: {self.config.num_families}",
            f"Depth per family: {self.config.depth_per_family}",
            f"Target family: {self.config.infected_family}",
            f"Drift severity: {self.config.drift_severity}",
            f"Cascade factor: {self.config.cascade_factor}",
            "",
            "RESULTS",
            "-" * 40,
            json.dumps(self.results.to_dict(), indent=2),
            "",
            "VISUALIZATION REPORT",
            "-" * 40,
            self.visualizer.generate_report()
        ]
        return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run case study with default config
    sim = DriftCascadeSimulation()
    results = sim.run_simulation(verbose=True)
    
    print()
    print(">>> EXPORTING REPORT...")
    report = sim.export_report()
    print(f"Report generated ({len(report)} characters)")

