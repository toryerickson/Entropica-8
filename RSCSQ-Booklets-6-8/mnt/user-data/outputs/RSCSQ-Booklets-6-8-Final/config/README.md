# Configuration

Production configuration files for RSCS-Q Booklets 6-8.

## Files

### hardening.yaml

**Main operational configuration** for production deployment.

Key sections:
- `validity`: Bayesian threshold with hysteresis (enter/exit)
- `identity_graph`: Coherence thresholds and regularization
- `small_n`: Policy for sparse identity graphs
- `observer_mesh`: Quorum, Byzantine tolerance, reputation
- `audit`: Merkle chains, epoch roots, external notary
- `slas`: Operational timing guarantees
- `repair`: Drift-debt budget and cooling periods
- `metrics`: Runtime gates (RCI, PSR, SHY)
- `bridge`: B7→B8 behavioral contracts
- `compatibility`: Version pinning
- `simulation`: Test scenario settings

**Example usage:**
```python
import yaml

with open('config/hardening.yaml') as f:
    config = yaml.safe_load(f)

validity_enter = config['validity']['tau_enter']  # 0.70
sla_escalation = config['slas']['time_to_escalation_ticks']  # 1
```

### baseline_config.json

**Test scenario configuration** for the capstone integration harness.

Contains:
- Scenario list (baseline, novelty_burst, slow_drift, etc.)
- B6/B7/B8 threshold settings
- G-criteria definitions with targets
- SLA thresholds
- Output paths

**Example usage:**
```python
import json

with open('config/baseline_config.json') as f:
    config = json.load(f)

scenarios = config['scenarios']
g1_target = config['g_criteria']['G1']['threshold']  # 0.80
```

## Key Parameters

### Validity Hysteresis

```yaml
validity:
  tau_enter: 0.70    # Must exceed 70% to become valid
  tau_exit: 0.60     # Must stay above 60% to remain valid
```

This prevents oscillation between valid/invalid states.

### Operational SLAs

```yaml
slas:
  time_to_escalation_ticks: 1     # Flag issues within 1 tick
  rollback_bound_ticks: 5         # Complete rollback within 5 ticks
  audit_ingestion_latency_ticks: 1  # Log events within 1 tick
  duplicate_collapse_budget: 0.0001  # Max 0.01% duplicates
```

### Drift-Debt Budget

```yaml
repair:
  drift_debt:
    max_budget: 15.0              # Maximum accumulated repair cost
    cooling_period_ticks: 5       # Ticks between repairs
    counterfactual:
      replay_percentage: 10       # Validate 10% of repairs
```

## Customization

To customize for your deployment:

1. Copy `hardening.yaml` to your config directory
2. Adjust thresholds based on your risk tolerance
3. Set appropriate SLA targets for your environment
4. Configure external notary endpoint if using attestation

```yaml
# Example: More conservative settings
validity:
  tau_enter: 0.80    # Higher bar to enter valid
  tau_exit: 0.70     # Higher bar to stay valid

slas:
  time_to_escalation_ticks: 1  # Keep tight
  rollback_bound_ticks: 3      # Faster rollback
```
