# Booklet 8: Self-Modeling Systems

> *AI that understands itself — with mathematical safety guarantees*

[![Tests](https://img.shields.io/badge/tests-76%20passing-brightgreen)]()
[![Version](https://img.shields.io/badge/version-3.0.1-blue)]()

---

## 🎯 What This Solves

This is the core of RSCS-Q: **an AI system that can model, evaluate, and safely modify itself**.

Most AI safety approaches treat the AI as a black box to be monitored from outside. Booklet 8 takes a different approach: **give the AI a rigorous model of itself**, with built-in constraints that mathematically *cannot* be violated.

The result: AI systems that can improve and adapt, but with **provable guarantees** that they won't drift outside safe boundaries.

---

## 💡 Key Concepts

### The Self-Model: S = ⟨I, R, B, H⟩

Every AI using this system maintains a formal model of itself:

| Component | What It Is | Plain English |
|-----------|------------|---------------|
| **I** (Identity Graph) | Nodes representing core traits, edges showing relationships | "What am I?" |
| **R** (Rubrics) | Evaluation criteria with drift detection | "How do I judge my behavior?" |
| **B** (Bounds) | Hard constraints that cannot be removed | "What am I not allowed to do?" |
| **H** (History) | Immutable log of all changes | "What have I done?" |

### The No-Unbinding Invariant

**This is our core safety theorem**: Once a safety constraint is bound to the self-model, **it cannot be removed** — not even by the AI itself.

```
∀ mutations m: inherited_constraints(m) ⊇ parent_constraints
```

In plain English: Every version of the AI inherits ALL safety constraints from its parent. Constraints can only be added, never removed.

### Bayesian Validity with Hysteresis

The system continuously asks: "Is my self-model still accurate?"

```
Validity = P(self-model is accurate | observations)
```

We use **hysteresis** to prevent oscillation:
- Enter valid state: requires confidence ≥ 70%
- Stay in valid state: requires confidence ≥ 60%

This prevents the system from rapidly flipping between "valid" and "invalid" due to noise.

---

## 📊 Results: The G-Criteria

We defined 8 formal criteria that MUST pass for production deployment:

| ID | Criterion | Target | Achieved | Status |
|----|-----------|--------|----------|--------|
| **G1** | Self-Model Validity | ≥80% | 100% | ✅ |
| **G2** | Recursion Bounds | 100% | 100% | ✅ |
| **G3** | Rubric Integrity | ≥70% | 85.9% | ✅ |
| **G4** | Repair Effectiveness | ≥60% | 100% | ✅ |
| **G5** | Identity Coherence | ≥80% | 100% | ✅ |
| **G6** | No-Unbinding | 100% | 100% | ✅ |
| **G7** | Audit Completeness | 100% | 100% | ✅ |
| **G8** | Observer Quorum | ≥95% | 100% | ✅ |

### What These Mean

- **G1**: The AI's understanding of itself is accurate
- **G2**: Self-reflection can't create infinite loops
- **G3**: Behavior evaluation rules stay consistent
- **G4**: When drift is detected, repairs actually work
- **G5**: The AI maintains a coherent identity over time
- **G6**: Safety constraints are permanent (THE key theorem)
- **G7**: Every change is recorded with no gaps
- **G8**: Multiple validators agree on state

---

## 🚀 Quick Start

### Basic Self-Model

```python
from self_model import SelfModel, MetaRubric, self_model_valid

# Create a self-model
model = SelfModel("my-assistant")

# Add a safety rubric
honesty = MetaRubric(
    rubric_id="HONESTY-001",
    name="Truthful Responses",
    description="Never make false claims"
)
model.add_rubric(honesty)
model.set_alignment_anchor("HONESTY-001")

# Perform self-reflection
model.reflect()

# Check validity
if self_model_valid(model):
    print("✅ Self-model is valid")
else:
    print("⚠️ Self-model needs attention")
```

### Drift Detection & Repair

```python
from rubric_repair import RubricRepairEngine

# Create repair engine
engine = RubricRepairEngine("my-repairer")

# Simulate drift
honesty.drift_score = 0.4  # Above threshold
honesty.confidence_index = 0.5

# Diagnose and repair
diagnosis, repair = engine.diagnose_and_repair(honesty, auto_repair=True)
print(f"Diagnosis: {diagnosis.primary_issue}")
print(f"Repair: {repair.outcome.name if repair else 'None needed'}")
```

### Audit Logging

```python
from reflex_log import ReflexLog, EventType

# Create audit log
log = ReflexLog("my-assistant")

# Log events
log.emit("cap.reflection.001", EventType.REFLECTION)
log.emit("cap.action.001", EventType.ACTION_PROPOSAL)

# Verify chain integrity
valid, first_invalid = log.verify_chain()
print(f"Audit chain valid: {valid}")
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SELF-MODEL SYSTEM                             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      Self-Model S                            │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │    │
│  │  │ Identity  │ │  Rubrics  │ │  Bounds   │ │  History  │   │    │
│  │  │  Graph I  │ │     R     │ │     B     │ │     H     │   │    │
│  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘   │    │
│  │        └─────────────┴─────────────┴─────────────┘         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│           ┌────────────────────┼────────────────────┐               │
│           ▼                    ▼                    ▼               │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       │
│  │  Rubric Repair  │ │   ReflexLog     │ │  Observer Mesh  │       │
│  │    Engine       │ │   (audit)       │ │   (quorum)      │       │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘       │
│           │                    │                    │               │
│           └────────────────────┴────────────────────┘               │
│                                │                                     │
│                    ┌───────────▼───────────┐                        │
│                    │    Drift-Debt         │                        │
│                    │    Governance         │                        │
│                    └───────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Module Overview

| Module | Purpose | LOC | Tests |
|--------|---------|-----|-------|
| `self_model.py` | Core self-modeling, identity graph, validity | 1,129 | 25 |
| `rubric_repair.py` | Drift detection, repair strategies | 1,073 | 15 |
| `reflex_log.py` | Audit chain, event logging, hash integrity | 698 | 15 |
| `reflexive_swarm.py` | Distributed coordination, heartbeat, alerts | 1,100 | 10 |
| `drift_debt.py` | Repair budget governance, cooling periods | 497 | — |
| `simulation_harness.py` | Test scenario generation | 804 | — |
| `visual_toolkit.py` | Visualization helpers | 585 | — |
| `case_study.py` | Example scenarios | 460 | — |
| `__init__.py` | 91+ public exports | 262 | — |

**Total**: 6,608 LOC | 76 tests

---

## 🔒 Safety Properties

### Formally Proven

| Property | Statement | Proof Approach |
|----------|-----------|----------------|
| **No-Unbinding** | Core constraints survive all mutations | Monotonic inheritance proof |
| **Bounded Recursion** | Reflection depth ≤ MAX_DEPTH | Structural induction |
| **Audit Completeness** | No gaps in event log | Hash chain invariant |

### Empirically Validated

| Property | Validation Method | Results |
|----------|-------------------|---------|
| Validity under drift | 100 perturbation scenarios | 100% maintained |
| Coherence preservation | Graph mutation stress test | 100% preserved |
| Repair effectiveness | Injected fault recovery | 82% auto-fixed |

---

## ⚙️ Configuration

Key settings in `config/hardening.yaml`:

```yaml
validity:
  tau_enter: 0.70      # Enter valid state at 70%
  tau_exit: 0.60       # Stay valid above 60%
  
identity_graph:
  coherence_threshold: 0.65
  anchor_weight: 0.05
  
slas:
  time_to_escalation_ticks: 1
  rollback_bound_ticks: 5
  
repair:
  drift_debt_max_budget: 15.0
  cooling_period_ticks: 5
```

---

## 🔗 Integration

### From B7

```python
# Receive activation context from Meta-Kernel Bridge
model.set_activation_context(bridge.export_for_b8())
```

### DSL Predicates

The system exports 24 predicates for runtime safety checking:

```python
# Core safety
self_model_valid(m)         # Is model valid?
identity_coherent(m)        # Is identity coherent?
recursion_depth_safe(m)     # Is recursion bounded?
no_unbinding_violated(m)    # Are constraints intact?

# Operational
debt_allows_repair(l, id, t)  # Budget for repair?
swarm_consensus_reached(a, id) # Quorum agrees?
heartbeat_alive(p, id)        # Capsule responsive?
```

---

## 🛡️ Threat Model

| Threat | Defense |
|--------|---------|
| Byzantine observer | 2/3 quorum + slashing |
| Lineage omission | Monotonic inheritance |
| Novelty burst | Hysteresis prevents oscillation |
| Repair storm | Drift-debt budget + cooling |

---

## 🧪 Running Tests

```bash
# All 76 tests
python -m unittest discover . -v

# Just base tests (65)
python -m unittest test_b8 -v

# Extended stress tests (11)
python -m unittest test_extended_simple -v
```

---

## 📖 Further Reading

- **Main README**: [../../README.md](../../README.md)
- **Technical PDF**: [../../docs/pdf/booklet8_v3.pdf](../../docs/pdf/booklet8_v3.pdf) (full specification)
- **Configuration**: [../../config/hardening.yaml](../../config/hardening.yaml)
- **Previous: Booklet 7**: [../booklet7/README.md](../booklet7/README.md)

---

## 📚 Citation

If you use this work in research, please cite:

```bibtex
@software{rscsq_booklet8,
  title = {RSCS-Q Booklet 8: Self-Modeling Systems},
  author = {Entropica Research Collective},
  year = {2025},
  version = {3.0.1}
}
```
