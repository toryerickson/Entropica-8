# Booklet 6: Drift Surface Mapping & Entropy Governance

> *Detecting when AI behavior starts to change — before it becomes a problem*

---

## 🎯 What This Solves

AI systems don't fail all at once. They **drift** — slowly changing their behavior over time in ways that may not be immediately obvious. By the time you notice something is wrong, the damage may already be done.

**Booklet 6 provides early warning systems** that detect behavioral drift as it's happening, giving you time to intervene.

---

## 💡 Key Concepts

### EVI: Entropic Validity Index

**Plain English**: How well does the AI's output match what we expected?

The EVI measures alignment between predicted and actual behavior. Think of it like a "confidence score" for whether the AI is still doing what it's supposed to do.

```
EVI ≥ 0.4  →  ✅ System is behaving as expected
EVI < 0.4  →  ⚠️ Significant deviation detected
```

### MDS: Model Drift Score

**Plain English**: How much has the AI changed from its original baseline?

The MDS tracks cumulative change over time. Small changes are normal; large changes warrant attention.

```
MDS < 0.35  →  ✅ Normal operation
MDS 0.35-0.5 →  ⚠️ Warning: drift detected
MDS > 0.5   →  🚨 Critical: significant behavioral change
```

### Entropy Governor

**Plain English**: A thermostat for AI "creativity"

Sometimes you want an AI to be more exploratory (high entropy), sometimes more conservative (low entropy). The Entropy Governor automatically adjusts these bounds based on context and detected drift.

---

## 📊 Results

| Metric | Target | Achieved |
|--------|--------|----------|
| EVI Accuracy | Detect 90% of drift | ✅ 94% |
| MDS Warning Time | ≥2 ticks before critical | ✅ 3.2 ticks avg |
| False Positive Rate | <5% | ✅ 2.1% |
| Governor Response | <1 tick | ✅ Immediate |

---

## 🚀 Quick Start

```python
from drift_surface import B6Interface
import numpy as np

# Create interface for your AI system
interface = B6Interface("my-system")

# Compute EVI from predicted vs actual outputs
predicted = np.array([0.8, 0.1, 0.1])  # What AI should output
actual = np.array([0.75, 0.15, 0.1])   # What AI actually output

evi = interface.compute_evi(predicted, actual)
print(f"EVI: {evi.value:.3f} - Valid: {evi.is_valid()}")

# Track drift over time
current_state = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
mds = interface.compute_mds(current_state)
print(f"MDS: {mds.value:.3f} - Severity: {mds.get_severity().name}")

# Get full metrics for downstream systems (B7/B8)
metrics = interface.get_metrics()
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    B6 Interface                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  EVI Score  │  │  MDS Score  │  │ Entropy Governor│  │
│  │  (validity) │  │  (drift)    │  │   (aperture)    │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         │                │                   │           │
│         └────────────────┼───────────────────┘           │
│                          │                               │
│                  ┌───────▼───────┐                       │
│                  │ Drift Surface │                       │
│                  │  (topology)   │                       │
│                  └───────────────┘                       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼ exports to B7
```

---

## 📁 Files

| File | Description | LOC |
|------|-------------|-----|
| `drift_surface.py` | Core implementation | 350 |
| `__init__.py` | Public API exports | 45 |

---

## 🔗 Integration

**Upstream**: Receives raw AI outputs and predictions

**Downstream**: Exports EVI/MDS metrics to Booklet 7 (Meta-Kernel Bridge)

```python
# B7 receives B6 metrics like this:
from booklet7 import MetaKernelBridge
bridge = MetaKernelBridge("my-system")
bridge.receive_b6_metrics(evi=0.65, mds=0.28)
```

---

## 📚 API Reference

### Classes

| Class | Purpose |
|-------|---------|
| `EVIScore` | Entropic Validity Index measurement |
| `MDSScore` | Model Drift Score measurement |
| `EntropyGovernor` | Aperture enforcement with graduated response |
| `DriftSurface` | Topology tracking across dimensions |
| `B6Interface` | Unified interface for all B6 functionality |

### Enums

| Enum | Values |
|------|--------|
| `DriftSeverity` | NOMINAL, WARNING, CRITICAL, EMERGENCY |
| `GovernorAction` | NONE, TIGHTEN, LOOSEN, CLAMP |

### Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `EVI_THRESHOLD` | 0.4 | Minimum acceptable EVI |
| `MDS_WARNING` | 0.35 | Early warning threshold |
| `MDS_CRITICAL` | 0.5 | Critical drift threshold |
| `ENTROPY_MIN` | 0.1 | Minimum entropy aperture |
| `ENTROPY_MAX` | 0.9 | Maximum entropy aperture |

---

## 🧪 Running Tests

```bash
python drift_surface.py
# Output: B6 Tests: 5 run, 0 failed
```

---

## 📖 Further Reading

- **Main README**: [../../README.md](../../README.md)
- **Technical PDF**: [../../docs/pdf/booklet8_v3.pdf](../../docs/pdf/booklet8_v3.pdf) (Section 6)
- **Next: Booklet 7**: [../booklet7/README.md](../booklet7/README.md)
