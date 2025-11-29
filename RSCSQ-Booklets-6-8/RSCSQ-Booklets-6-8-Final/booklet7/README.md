# Booklet 7: Reflective Autonomy & Swarm Intelligence

> *Coordinating multiple AI validators to agree on what's safe*

---

## 🎯 What This Solves

How do you know an AI's self-assessment is trustworthy? **You don't trust just one perspective.**

Booklet 7 introduces **swarm-based validation** — multiple independent observers that must reach consensus before any significant change is accepted. It also manages **activation levels** that adjust how much autonomy the AI has based on current conditions.

---

## 💡 Key Concepts

### Swarm Coherence

**Plain English**: Do all the validators agree on what's happening?

Instead of trusting a single check, we use multiple independent observers that vote on the AI's state. Changes only proceed if enough observers agree (quorum).

```
Coherence κ ≥ 0.67  →  ✅ Quorum reached, proceed
Coherence κ < 0.67  →  ⚠️ Disagreement, hold for review
Coherence κ < 0.60  →  🚨 Fork detected, recovery needed
```

### Activation Levels

**Plain English**: How much freedom does the AI have right now?

Based on drift metrics (from B6) and coherence, the system adjusts autonomy:

| Level | Description | When Used |
|-------|-------------|-----------|
| **DORMANT** | Human approval for everything | High drift, low confidence |
| **GUARDED** | Frequent checkpoints, limited actions | Moderate uncertainty |
| **ACTIVE** | Normal operation, standard oversight | Stable conditions |
| **AUTONOMOUS** | Full autonomy within bounds | High confidence, low drift |

### Meta-Kernel Bridge

**Plain English**: The translator between different safety layers

The bridge receives metrics from B6 (drift/entropy), makes activation decisions, and exports state to B8 (self-modeling). It's the central coordination point.

---

## 📊 Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Quorum Agreement | ≥67% | ✅ 100% in tests |
| Fork Detection | ≤3 ticks | ✅ 2.1 ticks avg |
| Activation Response | ≤1 tick | ✅ Immediate |
| False Fork Rate | <2% | ✅ 0.8% |

---

## 🚀 Quick Start

```python
from meta_kernel_bridge import MetaKernelBridge, SwarmCoherence, SwarmMember

# Create the bridge
bridge = MetaKernelBridge("my-system")

# Feed it metrics from B6
bridge.receive_b6_metrics(evi=0.7, mds=0.2)
print(f"Activation: {bridge.current_profile.level.name}")
# Output: Activation: ACTIVE

# Add swarm members for consensus
for i in range(5):
    member = SwarmMember(f"validator-{i}")
    member.update_hash({'state': 'healthy', 'tick': 42})
    bridge.swarm.add_member(member)

# Check coherence
coherence = bridge.swarm.compute_coherence()
print(f"Coherence: {coherence:.2f} - State: {bridge.swarm.state.name}")
# Output: Coherence: 1.00 - State: COHERENT

# Export for B8
b8_input = bridge.export_for_b8()
```

---

## 🏗️ Architecture

```
                    From B6 (EVI, MDS)
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Meta-Kernel Bridge                      │
│                                                          │
│  ┌─────────────────┐         ┌─────────────────┐        │
│  │   Activation    │◄───────►│  Swarm Coherence │        │
│  │    Profile      │         │    (consensus)   │        │
│  │  ┌───────────┐  │         │  ┌───────────┐  │        │
│  │  │ DORMANT   │  │         │  │ Member 1  │  │        │
│  │  │ GUARDED   │  │         │  │ Member 2  │  │        │
│  │  │ ACTIVE    │  │         │  │ Member 3  │  │        │
│  │  │ AUTONOMOUS│  │         │  │    ...    │  │        │
│  │  └───────────┘  │         │  └───────────┘  │        │
│  └─────────────────┘         └─────────────────┘        │
│                                                          │
│  ┌─────────────────────────────────────────────┐        │
│  │            Reflexive Override                │        │
│  │     (bounded self-modification + audit)      │        │
│  └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼ exports to B8
```

---

## 📁 Files

| File | Description | LOC |
|------|-------------|-----|
| `meta_kernel_bridge.py` | Core implementation | 480 |
| `__init__.py` | Public API exports | 50 |

---

## 🔗 Integration

**Upstream**: Receives EVI/MDS from Booklet 6

**Downstream**: Exports activation profile and coherence to Booklet 8

```python
# B6 → B7 integration
bridge.receive_b6_metrics(evi=interface.evi_history[-1].value,
                          mds=interface.mds_history[-1].value)

# B7 → B8 integration
from booklet8 import SelfModel
model = SelfModel("my-ai")
model.set_activation_context(bridge.export_for_b8())
```

---

## 📚 API Reference

### Classes

| Class | Purpose |
|-------|---------|
| `SwarmMember` | Individual validator in the consensus swarm |
| `SwarmCoherence` | Manages hash-based agreement protocol |
| `ActivationProfile` | Defines behavior at each autonomy level |
| `MetaKernelBridge` | Central B6↔B7↔B8 coordinator |
| `OverrideRequest` | Request for bounded self-modification |
| `ReflexiveOverride` | Manages modification requests with audit |

### Enums

| Enum | Values |
|------|--------|
| `ActivationLevel` | DORMANT, GUARDED, ACTIVE, AUTONOMOUS |
| `SwarmState` | COHERENT, DIVERGING, FORKED, RECOVERING |

### Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `QUORUM_THRESHOLD` | 0.67 | Minimum agreement for consensus |
| `FORK_TIMEOUT` | 3 | Ticks before fork triggers recovery |
| `COHERENCE_MIN` | 0.6 | Below this = fork detected |

---

## 🔄 Activation State Machine

```
                    ┌─────────────┐
         MDS>0.5    │   DORMANT   │    MDS<0.35
        ┌──────────►│  (locked)   │◄──────────┐
        │           └──────┬──────┘           │
        │                  │ MDS<0.5          │
        │                  ▼                  │
        │           ┌─────────────┐           │
        │  MDS>0.35 │   GUARDED   │  EVI>0.6  │
        ├──────────►│ (cautious)  │───────────┤
        │           └──────┬──────┘           │
        │                  │ EVI>0.6          │
        │                  ▼                  │
        │           ┌─────────────┐           │
        │           │   ACTIVE    │           │
        └───────────│  (normal)   │───────────┘
                    └──────┬──────┘
                           │ Manual override only
                           ▼
                    ┌─────────────┐
                    │ AUTONOMOUS  │
                    │   (full)    │
                    └─────────────┘
```

---

## 🧪 Running Tests

```bash
python meta_kernel_bridge.py
# Output: B7 Tests: 5 run, 0 failed
```

---

## 📖 Further Reading

- **Main README**: [../../README.md](../../README.md)
- **Technical PDF**: [../../docs/pdf/booklet8_v3.pdf](../../docs/pdf/booklet8_v3.pdf) (Section 7)
- **Previous: Booklet 6**: [../booklet6/README.md](../booklet6/README.md)
- **Next: Booklet 8**: [../booklet8/README.md](../booklet8/README.md)
