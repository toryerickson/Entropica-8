"""
RSCS-Q Booklet 8: Visual Lineage and Drift Toolkit
===================================================

Visualization tools for capsule evolution, drift vectors, and recovery paths.

This module implements:
- LineageVisualizer: Generate genealogical diagrams
- DriftHeatmap: Visualize drift across capsule populations  
- RecoveryTimeline: Track repair and recovery events
- DOT/GraphViz export for external rendering

Author: Entropica Research Collective
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict
import json

from self_model import SelfModel, SelfModelState, MAX_RECURSION_DEPTH


# =============================================================================
# LINEAGE TREE VISUALIZATION
# =============================================================================

@dataclass
class LineageNode:
    """Node in lineage visualization"""
    capsule_id: str
    parent_id: Optional[str]
    depth: int
    state: str
    drift_score: float
    coherence: float
    children: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'capsule_id': self.capsule_id,
            'parent_id': self.parent_id,
            'depth': self.depth,
            'state': self.state,
            'drift_score': self.drift_score,
            'coherence': self.coherence,
            'children': self.children
        }


class LineageVisualizer:
    """
    Generates visual representations of capsule lineages.
    
    Supports:
    - DOT/GraphViz format for external rendering
    - ASCII tree for terminal display
    - JSON export for web visualization
    """
    
    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self.roots: List[str] = []
    
    def add_model(self, model: SelfModel) -> None:
        """Add a self-model to the visualization"""
        node = LineageNode(
            capsule_id=model.capsule_id,
            parent_id=model.parent_id,
            depth=model.lineage_depth,
            state=model.state.name,
            drift_score=model.drift_score,
            coherence=model.identity.compute_coherence(model.drift_score),
            children=model.children.copy()
        )
        self.nodes[model.capsule_id] = node
        
        if model.parent_id is None:
            if model.capsule_id not in self.roots:
                self.roots.append(model.capsule_id)
    
    def add_models(self, models: List[SelfModel]) -> None:
        """Add multiple models"""
        for model in models:
            self.add_model(model)
    
    def generate_dot(self, title: str = "Capsule Lineage") -> str:
        """
        Generate DOT format for GraphViz rendering.
        
        Color coding:
        - Green: STABLE state
        - Yellow: REFLECTING/ADAPTING
        - Red: QUARANTINED
        - Gray: TERMINATED
        """
        lines = [
            f'digraph "{title}" {{',
            '  rankdir=TB;',
            '  node [shape=box, style=filled];',
            ''
        ]
        
        # Define node colors based on state
        state_colors = {
            'STABLE': '#90EE90',      # Light green
            'INITIALIZING': '#ADD8E6', # Light blue
            'REFLECTING': '#FFFFE0',   # Light yellow
            'ADAPTING': '#FFD700',     # Gold
            'REPAIRING': '#FFA500',    # Orange
            'QUARANTINED': '#FF6B6B',  # Light red
            'TERMINATED': '#D3D3D3'    # Light gray
        }
        
        # Add nodes
        for node_id, node in self.nodes.items():
            color = state_colors.get(node.state, '#FFFFFF')
            
            # Adjust color intensity based on drift
            if node.drift_score > 0.3:
                color = '#FF6B6B'  # Red for high drift
            
            label = f"{node.capsule_id}\\nD:{node.depth} drift:{node.drift_score:.2f}\\ncoh:{node.coherence:.2f}"
            lines.append(f'  "{node_id}" [label="{label}", fillcolor="{color}"];')
        
        lines.append('')
        
        # Add edges
        for node_id, node in self.nodes.items():
            if node.parent_id and node.parent_id in self.nodes:
                lines.append(f'  "{node.parent_id}" -> "{node_id}";')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def generate_ascii_tree(self) -> str:
        """Generate ASCII representation of lineage tree"""
        lines = []
        
        def render_node(node_id: str, prefix: str = "", is_last: bool = True):
            if node_id not in self.nodes:
                return
            
            node = self.nodes[node_id]
            connector = "└── " if is_last else "├── "
            
            state_marker = {
                'STABLE': '✓',
                'QUARANTINED': '✗',
                'REFLECTING': '○',
                'ADAPTING': '◐',
                'REPAIRING': '⚙',
            }.get(node.state, '?')
            
            drift_bar = "█" * int(node.drift_score * 10) + "░" * (10 - int(node.drift_score * 10))
            
            lines.append(f"{prefix}{connector}{node_id} [{state_marker}] drift:[{drift_bar}] coh:{node.coherence:.2f}")
            
            # Render children
            child_prefix = prefix + ("    " if is_last else "│   ")
            children = [c for c in node.children if c in self.nodes]
            for i, child_id in enumerate(children):
                render_node(child_id, child_prefix, i == len(children) - 1)
        
        for i, root_id in enumerate(self.roots):
            if i > 0:
                lines.append("")
            lines.append(f"LINEAGE TREE: {root_id}")
            lines.append("=" * 60)
            render_node(root_id, "", True)
        
        return '\n'.join(lines)
    
    def generate_json(self) -> str:
        """Generate JSON for web visualization"""
        return json.dumps({
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'roots': self.roots,
            'metadata': {
                'total_nodes': len(self.nodes),
                'max_depth': max((n.depth for n in self.nodes.values()), default=0),
                'generated_at': datetime.utcnow().isoformat()
            }
        }, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lineage statistics"""
        if not self.nodes:
            return {'total_nodes': 0}
        
        states = defaultdict(int)
        for node in self.nodes.values():
            states[node.state] += 1
        
        drifts = [n.drift_score for n in self.nodes.values()]
        coherences = [n.coherence for n in self.nodes.values()]
        
        return {
            'total_nodes': len(self.nodes),
            'root_count': len(self.roots),
            'max_depth': max(n.depth for n in self.nodes.values()),
            'state_distribution': dict(states),
            'drift_stats': {
                'mean': float(np.mean(drifts)),
                'std': float(np.std(drifts)),
                'max': float(max(drifts)),
                'min': float(min(drifts))
            },
            'coherence_stats': {
                'mean': float(np.mean(coherences)),
                'std': float(np.std(coherences)),
                'max': float(max(coherences)),
                'min': float(min(coherences))
            }
        }


# =============================================================================
# DRIFT HEATMAP
# =============================================================================

class DriftHeatmap:
    """
    Visualize drift across capsule populations over time.
    
    Generates CSV data suitable for heatmap rendering.
    """
    
    def __init__(self, time_steps: int = 20):
        self.time_steps = time_steps
        self.data: Dict[str, List[float]] = {}  # capsule_id -> drift history
        self.timestamps: List[str] = []
    
    def record_snapshot(self, models: List[SelfModel], timestamp: str = None) -> None:
        """Record drift values at current time"""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        
        self.timestamps.append(timestamp)
        
        for model in models:
            if model.capsule_id not in self.data:
                self.data[model.capsule_id] = []
            self.data[model.capsule_id].append(model.drift_score)
    
    def generate_csv(self) -> str:
        """Generate CSV format for heatmap visualization"""
        lines = ["capsule_id,time_step,drift_score"]
        
        for capsule_id, drifts in self.data.items():
            for t, drift in enumerate(drifts):
                lines.append(f"{capsule_id},{t},{drift:.4f}")
        
        return '\n'.join(lines)
    
    def generate_matrix_csv(self) -> str:
        """Generate matrix CSV (x=time, y=capsule, z=drift)"""
        lines = ["x,y,z"]
        
        capsule_list = list(self.data.keys())
        for y, capsule_id in enumerate(capsule_list):
            drifts = self.data[capsule_id]
            for x, drift in enumerate(drifts):
                lines.append(f"{x},{y},{drift:.4f}")
        
        return '\n'.join(lines)
    
    def get_drift_trends(self) -> Dict[str, str]:
        """Analyze drift trends per capsule"""
        trends = {}
        
        for capsule_id, drifts in self.data.items():
            if len(drifts) < 3:
                trends[capsule_id] = 'insufficient_data'
                continue
            
            recent = drifts[-3:]
            earlier = drifts[:3] if len(drifts) >= 6 else drifts[:len(drifts)//2]
            
            recent_avg = np.mean(recent)
            earlier_avg = np.mean(earlier)
            
            if recent_avg > earlier_avg * 1.1:
                trends[capsule_id] = 'increasing'
            elif recent_avg < earlier_avg * 0.9:
                trends[capsule_id] = 'decreasing'
            else:
                trends[capsule_id] = 'stable'
        
        return trends


# =============================================================================
# RECOVERY TIMELINE
# =============================================================================

@dataclass
class RecoveryEvent:
    """A recovery or repair event"""
    timestamp: datetime
    capsule_id: str
    event_type: str  # 'drift_detected', 'repair_started', 'repair_completed', 'quarantine', 'recovery'
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'capsule_id': self.capsule_id,
            'event_type': self.event_type,
            'details': self.details
        }


class RecoveryTimeline:
    """
    Track and visualize repair and recovery events.
    
    Useful for analyzing system resilience and response times.
    """
    
    def __init__(self):
        self.events: List[RecoveryEvent] = []
        self.active_issues: Dict[str, RecoveryEvent] = {}  # capsule_id -> drift event
    
    def record_drift_detected(self, capsule_id: str, drift_score: float) -> None:
        """Record drift detection"""
        event = RecoveryEvent(
            timestamp=datetime.utcnow(),
            capsule_id=capsule_id,
            event_type='drift_detected',
            details={'drift_score': drift_score}
        )
        self.events.append(event)
        self.active_issues[capsule_id] = event
    
    def record_repair_started(self, capsule_id: str, strategy: str) -> None:
        """Record start of repair"""
        event = RecoveryEvent(
            timestamp=datetime.utcnow(),
            capsule_id=capsule_id,
            event_type='repair_started',
            details={'strategy': strategy}
        )
        self.events.append(event)
    
    def record_repair_completed(self, capsule_id: str, success: bool, new_drift: float) -> None:
        """Record repair completion"""
        event = RecoveryEvent(
            timestamp=datetime.utcnow(),
            capsule_id=capsule_id,
            event_type='repair_completed',
            details={'success': success, 'new_drift': new_drift}
        )
        self.events.append(event)
        
        if success and capsule_id in self.active_issues:
            # Calculate recovery time
            start_event = self.active_issues[capsule_id]
            recovery_time = (event.timestamp - start_event.timestamp).total_seconds()
            event.details['recovery_time_seconds'] = recovery_time
            del self.active_issues[capsule_id]
    
    def record_quarantine(self, capsule_id: str, reason: str) -> None:
        """Record quarantine event"""
        event = RecoveryEvent(
            timestamp=datetime.utcnow(),
            capsule_id=capsule_id,
            event_type='quarantine',
            details={'reason': reason}
        )
        self.events.append(event)
    
    def record_recovery(self, capsule_id: str) -> None:
        """Record full recovery"""
        event = RecoveryEvent(
            timestamp=datetime.utcnow(),
            capsule_id=capsule_id,
            event_type='recovery',
            details={}
        )
        self.events.append(event)
        
        if capsule_id in self.active_issues:
            del self.active_issues[capsule_id]
    
    def generate_timeline_ascii(self) -> str:
        """Generate ASCII timeline visualization"""
        if not self.events:
            return "No events recorded"
        
        lines = [
            "RECOVERY TIMELINE",
            "=" * 80
        ]
        
        event_symbols = {
            'drift_detected': '⚠️ ',
            'repair_started': '🔧',
            'repair_completed': '✅' if True else '❌',
            'quarantine': '🔒',
            'recovery': '🌟'
        }
        
        for event in sorted(self.events, key=lambda e: e.timestamp):
            symbol = event_symbols.get(event.event_type, '•')
            time_str = event.timestamp.strftime('%H:%M:%S')
            
            details_str = ""
            if 'drift_score' in event.details:
                details_str = f" (drift={event.details['drift_score']:.3f})"
            elif 'strategy' in event.details:
                details_str = f" ({event.details['strategy']})"
            elif 'success' in event.details:
                status = "success" if event.details['success'] else "failed"
                details_str = f" ({status})"
            
            lines.append(f"  {time_str} {symbol} [{event.capsule_id}] {event.event_type}{details_str}")
        
        lines.append("=" * 80)
        
        # Summary
        event_counts = defaultdict(int)
        for e in self.events:
            event_counts[e.event_type] += 1
        
        lines.append(f"Total events: {len(self.events)}")
        lines.append(f"Active issues: {len(self.active_issues)}")
        for etype, count in event_counts.items():
            lines.append(f"  {etype}: {count}")
        
        return '\n'.join(lines)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics"""
        repair_events = [e for e in self.events if e.event_type == 'repair_completed']
        successful = [e for e in repair_events if e.details.get('success', False)]
        
        recovery_times = [
            e.details.get('recovery_time_seconds', 0) 
            for e in successful 
            if 'recovery_time_seconds' in e.details
        ]
        
        return {
            'total_events': len(self.events),
            'drift_detections': sum(1 for e in self.events if e.event_type == 'drift_detected'),
            'repairs_attempted': len(repair_events),
            'repairs_successful': len(successful),
            'success_rate': len(successful) / len(repair_events) if repair_events else 1.0,
            'quarantines': sum(1 for e in self.events if e.event_type == 'quarantine'),
            'active_issues': len(self.active_issues),
            'avg_recovery_time': float(np.mean(recovery_times)) if recovery_times else 0.0
        }
    
    def export_json(self) -> str:
        """Export timeline as JSON"""
        return json.dumps({
            'events': [e.to_dict() for e in self.events],
            'metrics': self.get_metrics()
        }, indent=2)


# =============================================================================
# INTEGRATED VISUALIZER
# =============================================================================

class B8Visualizer:
    """
    Integrated visualization toolkit for Booklet 8.
    
    Combines lineage, drift, and recovery visualizations.
    """
    
    def __init__(self):
        self.lineage = LineageVisualizer()
        self.heatmap = DriftHeatmap()
        self.timeline = RecoveryTimeline()
    
    def ingest_models(self, models: List[SelfModel]) -> None:
        """Add models to all visualizers"""
        self.lineage.add_models(models)
        self.heatmap.record_snapshot(models)
    
    def record_repair(self, capsule_id: str, strategy: str, success: bool, new_drift: float) -> None:
        """Record a repair operation"""
        self.timeline.record_repair_started(capsule_id, strategy)
        self.timeline.record_repair_completed(capsule_id, success, new_drift)
    
    def generate_report(self) -> str:
        """Generate comprehensive visualization report"""
        lines = [
            "=" * 80,
            "BOOKLET 8: VISUALIZATION REPORT",
            "=" * 80,
            "",
            ">>> LINEAGE TREE",
            self.lineage.generate_ascii_tree(),
            "",
            ">>> LINEAGE STATISTICS",
            json.dumps(self.lineage.get_statistics(), indent=2),
            "",
            ">>> DRIFT TRENDS",
            json.dumps(self.heatmap.get_drift_trends(), indent=2),
            "",
            ">>> RECOVERY TIMELINE",
            self.timeline.generate_timeline_ascii(),
            "",
            ">>> RECOVERY METRICS",
            json.dumps(self.timeline.get_metrics(), indent=2),
            "",
            "=" * 80
        ]
        return '\n'.join(lines)
    
    def export_all(self, prefix: str = "b8_viz") -> Dict[str, str]:
        """Export all visualizations to files"""
        return {
            f'{prefix}_lineage.dot': self.lineage.generate_dot(),
            f'{prefix}_lineage.json': self.lineage.generate_json(),
            f'{prefix}_heatmap.csv': self.heatmap.generate_matrix_csv(),
            f'{prefix}_timeline.json': self.timeline.export_json()
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BOOKLET 8: VISUAL LINEAGE AND DRIFT TOOLKIT DEMO")
    print("=" * 70)
    print()
    
    # Create test models
    np.random.seed(42)
    
    root = SelfModel("ROOT-001")
    root.drift_score = 0.1
    
    child1 = root.spawn_child("CHILD-001")
    child1.drift_score = 0.15
    
    child2 = root.spawn_child("CHILD-002")
    child2.drift_score = 0.35  # High drift
    child2.quarantine("high drift")
    
    grandchild = child1.spawn_child("GRANDCHILD-001")
    grandchild.drift_score = 0.08
    
    models = [root, child1, child2, grandchild]
    
    # Create visualizer
    viz = B8Visualizer()
    viz.ingest_models(models)
    
    # Record some events
    viz.timeline.record_drift_detected("CHILD-002", 0.35)
    viz.record_repair("CHILD-002", "ResetBaseline", False, 0.30)
    viz.timeline.record_quarantine("CHILD-002", "drift exceeded threshold")
    
    # Generate report
    print(viz.generate_report())
    
    # Show DOT output
    print()
    print(">>> DOT OUTPUT (for GraphViz)")
    print("-" * 40)
    print(viz.lineage.generate_dot("Demo Lineage")[:500] + "...")
    
    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
