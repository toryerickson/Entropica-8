# JSON Schemas

JSON Schema definitions for RSCS-Q data structures.

## Files

### reflexlog_schema.json

Schema for **ReflexLog audit events** (v3.0.1).

**Required fields:**
- `ts`: ISO 8601 timestamp
- `capsule_id`: Capsule identifier (pattern: `cap.[A-Za-z0-9_.]+`)
- `observer_phase`: Phase number (integer)
- `event_type`: One of `reflection`, `collapse`, `action_proposal`, `mutation`, `quarantine`, `repair`
- `quorum`: Number of observers
- `constraints_hash`: SHA-256 hash of active constraints
- `entropy`: Normalized entropy (0-1)
- `delta_state_hash`: SHA-256 of state delta
- `lineage_ptr`: Parent→child pointer
- `rci`, `psr`, `shy`: Runtime metrics

**v3.0 additions:**
- `valid_posterior`: Bayesian validity score
- `valid_band`: Hysteresis bounds [tau_exit, tau_enter]
- `grace_remaining`: Ticks in grace period
- `epoch_root`: Merkle root for epoch
- `certificate_id`: Observer certificate ID
- `repair_debt`: Current drift-debt level

**Example:**
```json
{
  "ts": "2025-11-29T21:00:00Z",
  "capsule_id": "cap.main.001",
  "observer_phase": 42,
  "event_type": "reflection",
  "quorum": 5,
  "constraints_hash": "sha256:abc123...",
  "entropy": 0.45,
  "delta_state_hash": "sha256:def456...",
  "lineage_ptr": "cap.main.000->cap.main.001",
  "rci": 0.85,
  "psr": 45.2,
  "shy": 0.08,
  "valid_posterior": 0.92,
  "valid_band": [0.60, 0.70],
  "grace_remaining": 0,
  "epoch_root": "merkle:789abc...",
  "certificate_id": "cert-epoch-42",
  "repair_debt": 3
}
```

### observer_cert_schema.json

Schema for **Observer quorum certificates** (v3.0.1).

**Required fields:**
- `epoch`: Epoch number
- `beacon`: Rotating epoch beacon hash
- `quorum_ratio`: Achieved ratio (0-1, must be ≥0.67)
- `max_clock_skew_ms`: Maximum clock skew (≤100ms)
- `votes`: Array of observer votes
- `epoch_root`: Merkle root for epoch

**Optional fields:**
- `witness_digests`: Witness hash array
- `rollup_id`: Attestation rollup ID
- `slashing_events`: Any slashing during epoch (v3.0.1)

**Example:**
```json
{
  "epoch": 42,
  "beacon": "beacon-hash-abc123",
  "quorum_ratio": 0.80,
  "max_clock_skew_ms": 50,
  "votes": [
    {
      "observer": "obs-001",
      "signature": "sig-abc",
      "ts": "2025-11-29T21:00:00Z",
      "vote": true
    }
  ],
  "epoch_root": "merkle:xyz789...",
  "rollup_id": "rollup-100",
  "slashing_events": []
}
```

## Validation

Use any JSON Schema validator:

```python
import json
import jsonschema

# Load schema
with open('schemas/reflexlog_schema.json') as f:
    schema = json.load(f)

# Validate event
event = {...}  # Your event
jsonschema.validate(event, schema)
```

## Schema Versioning

Schemas follow the same versioning as the main package:
- v3.0.1: Added `slashing_events` to observer cert
- v3.0.0: Added v3.0 fields (valid_posterior, epoch_root, etc.)
- v1.0.0: Initial schema
