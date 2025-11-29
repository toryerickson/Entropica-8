# Tests

This directory contains all test suites for RSCS-Q Booklets 6-8.

## Test Summary

| Suite | Tests | Description |
|-------|-------|-------------|
| `test_b8.py` | 65 | Core Booklet 8 tests (self-model, rubrics, audit) |
| `test_extended_simple.py` | 11 | Stress tests and edge cases |
| `test_g_suite_extended.py` | — | Extended G-criteria tests (reference) |

**Total: 76 tests passing**

## Running Tests

### All Tests

```bash
cd RSCSQ-Booklets-6-8
python -m unittest discover tests/ -v
```

### Individual Suites

```bash
# Core tests
python -m unittest tests.test_b8 -v

# Stress tests
python -m unittest tests.test_extended_simple -v
```

### From Source Directory

```bash
cd src/booklet8
python -m unittest discover ../../tests/ -v
```

## Test Categories

### test_b8.py (65 tests)

**Self-Model Tests** (25 tests)
- `test_creation` - Basic model creation
- `test_add_rubric` - Rubric management
- `test_reflect` - Self-reflection
- `test_validity` - Validity checking
- `test_identity_coherence` - Graph coherence
- `test_spawn_child` - Child model creation
- `test_no_unbinding` - Constraint inheritance

**Rubric Repair Tests** (15 tests)
- `test_diagnosis` - Drift diagnosis
- `test_repair_strategies` - All 4 repair strategies
- `test_effectiveness` - Repair success rates

**ReflexLog Tests** (15 tests)
- `test_emit` - Event emission
- `test_chain_integrity` - Hash chain validation
- `test_verify` - Chain verification

**Observer Mesh Tests** (10 tests)
- `test_add_observer` - Observer management
- `test_quorum` - Quorum validation
- `test_collapse_validation` - State collapse

### test_extended_simple.py (11 tests)

**Stress Distributions**
- `test_validity_distribution` - 100 models under reflection
- `test_coherence_by_graph_size` - N=1,2,3,5,10,20
- `test_repair_outcome_distribution` - 50 repair scenarios

**Edge Cases**
- `test_empty_graph` - 0 nodes
- `test_single_node` - 1 node
- `test_high_drift` - drift=0.9
- `test_observer_mesh_edge` - 1 vs 10 observers

**Audit Chain**
- `test_chain_under_load` - 200 events
- `test_tamper_detection` - Modified event detection

**Swarm Stress**
- `test_sudden_novelty` - Burst of new inputs
- `test_gradual_drift` - 20 ticks of slow change

## G-Criteria Validation

Each G-criterion maps to specific tests:

| G | Criterion | Primary Tests |
|---|-----------|---------------|
| G1 | Self-Model Validity | `test_validity`, `test_validity_distribution` |
| G2 | Recursion Bounds | `test_max_recursion_depth` |
| G3 | Rubric Integrity | `test_drift_detection`, `test_baseline` |
| G4 | Repair Effectiveness | `test_repair_*`, `test_repair_outcome_distribution` |
| G5 | Identity Coherence | `test_identity_coherence`, `test_coherence_by_graph_size` |
| G6 | No-Unbinding | `test_no_unbinding`, `test_constraint_inheritance` |
| G7 | Audit Completeness | `test_chain_*`, `test_tamper_detection` |
| G8 | Observer Quorum | `test_quorum`, `test_observer_mesh_edge` |

## Adding New Tests

Follow the existing patterns:

```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Initialize test fixtures"""
        self.model = SelfModel("TEST")
    
    def test_feature_basic(self):
        """Test basic functionality"""
        result = self.model.new_feature()
        self.assertTrue(result)
    
    def test_feature_edge_case(self):
        """Test edge case"""
        # ...
```

## Coverage

The test suite covers:
- ✅ All public API methods
- ✅ All DSL predicates
- ✅ All G-criteria
- ✅ Edge cases (empty, single, extreme values)
- ✅ Stress scenarios (high load, drift, novelty)
- ✅ Security scenarios (tampering, Byzantine)
