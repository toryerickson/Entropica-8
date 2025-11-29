# Documentation

## Contents

### Technical Specification

**[pdf/booklet8_v3.pdf](pdf/booklet8_v3.pdf)** (13 pages)

The complete technical specification for Booklet 8, including:

- Section 1: Architectural Reflection
- Section 2: Formal Safety Guarantees
- Section 3: Metacognitive Constructs
- Section 4: Runtime Constraints
- Section 5: System-Level Test Cases (B8-T1 to B8-T5)
- Section 6: Implementation Notes
- Section 7: Bridge B7→B8
- Section 8: Stakeholder Readiness
- Section 9: Hardening Addendum
  - 9.1 Validity Hardening
  - 9.2 Identity Graph Hardening
  - 9.3 Observer Mesh Resilience
  - 9.4 Audit Integrity
  - 9.5 Operational SLAs
  - 9.6 Repair Governance
  - 9.7 Calibration Protocol
  - 9.8 Behavioral Bridge Contracts
- Section 10: Acceptance Criteria (Normative)
- Appendix A: Glossary
- Appendix B: Schemas
- Appendix C: Calibration Playbook
- Appendix D: Threat Model Scenarios
- Appendix E: DSL Predicate Inventory

### LaTeX Source

**[booklet8_v3.tex](booklet8_v3.tex)**

Full LaTeX source for the technical specification. Compile with:

```bash
pdflatex booklet8_v3.tex
pdflatex booklet8_v3.tex  # Run twice for references
```

## Quick Links by Topic

| Topic | PDF Section | Source File |
|-------|-------------|-------------|
| Self-Model Formalism | Section 2 | `src/booklet8/self_model.py` |
| G-Criteria | Section 10 | `tests/test_b8.py` |
| Rubric Repair | Section 9.6 | `src/booklet8/rubric_repair.py` |
| Audit System | Section 9.4 | `src/booklet8/reflex_log.py` |
| Observer Mesh | Section 9.3 | `src/booklet8/reflex_log.py` |
| Drift-Debt | Section 9.6 | `src/booklet8/drift_debt.py` |
| Threat Model | Appendix D | `config/hardening.yaml` |
| Calibration | Appendix C | `config/baseline_config.json` |

## Versioning

| Version | Date | Changes |
|---------|------|---------|
| 3.0.1 | 2025-11-29 | Micro-polish: notary SLA, reputation retention, version matrix |
| 3.0.0 | 2025-11-29 | Production hardening, drift-debt, calibration playbook |
| 1.0.0 | 2025-11-28 | Initial release |
