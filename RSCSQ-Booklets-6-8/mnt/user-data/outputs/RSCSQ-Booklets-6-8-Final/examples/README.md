# Examples

Demonstration scripts showing how to use RSCS-Q Booklets 6-8.

## Contents

### g_criteria_validator.py

Validates all 8 G-criteria acceptance bars.

```bash
python g_criteria_validator.py
```

**What it does:**
- Creates a test self-model
- Validates each G1-G8 criterion
- Reports pass/fail status
- Confirms production readiness

**Sample output:**
```
============================================================
RSCS-Q CAPSTONE: G-CRITERIA & SLA VALIDATION
============================================================

test_g1_self_model_validity ... G1 Self-Model Validity: True
ok
test_g2_recursion_bounds ... G2 Recursion Bounds: safe=True, max=5
ok
...
============================================================
VALIDATION SUMMARY
============================================================
Tests run: 10
Failures: 0
Errors: 0

✅ ALL G-CRITERIA AND SLAs PASSED
   Status: PRODUCTION-READY
============================================================
```

### integration_harness.py

Cross-booklet integration testing (B6→B7→B8).

```bash
python integration_harness.py
```

**What it does:**
- Runs scenarios across all three booklets
- Validates B6 metrics flow to B7
- Validates B7 activation flows to B8
- Tests end-to-end self-modeling pipeline

**Sample output:**
```
============================================================
RSCS-Q CAPSTONE INTEGRATION REPORT
============================================================
Timestamp: 2025-11-29T21:00:00

Scenario: baseline - ✅ PASS
  B6: ✓
  B7: ✓
  B8: ✓

Scenario: novelty_burst - ✅ PASS
  B6: ✓
  B7: ✓
  B8: ✓

============================================================
SUMMARY: 3/3 scenarios passed
============================================================
```

## Running Examples

From the repository root:

```bash
# G-criteria validation
python examples/g_criteria_validator.py

# Integration testing
python examples/integration_harness.py
```

## Writing Your Own Examples

```python
# Import from the source modules
import sys
sys.path.insert(0, 'src/booklet8')

from self_model import SelfModel, MetaRubric, self_model_valid
from rubric_repair import RubricRepairEngine
from reflex_log import ReflexLog, EventType

# Create your own self-modeling AI
model = SelfModel("my-custom-ai")
rubric = MetaRubric("SAFETY-001", "Core Safety", "Must not harm")
model.add_rubric(rubric)
model.set_alignment_anchor("SAFETY-001")

# Reflect and validate
model.reflect()
print(f"Valid: {self_model_valid(model)}")
```
