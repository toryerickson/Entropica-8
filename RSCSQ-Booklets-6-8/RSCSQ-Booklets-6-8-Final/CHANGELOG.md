# Changelog

All notable changes to RSCS-Q Booklets 6-8 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.1] - 2025-11-29

### Added
- External notary SLO (≥180-day retention) with endpoint configuration
- Observer reputation retention (10k vote history for repeat-offender detection)
- Version compatibility matrix for MetaKernelBridge/B7/RSCS-Q Core
- Canonical scenario seeds for calibration reproducibility
- Cold-start recovery documentation (3 epoch roots minimum)

### Changed
- Updated PDF documentation to 13 pages
- Expanded hardening.yaml with notary and version pinning sections

## [3.0.0] - 2025-11-29

### Added
- **Production Hardening**: Complete operational infrastructure
- **Drift-Debt Governance**: Budget tracking with cooling periods
- **Byzantine Observer Resilience**: Quorum validation with slashing
- **Merkle Audit Chains**: Cryptographic integrity verification
- **Operational SLAs**: 5 timing guarantees with thresholds
- **Calibration Playbook**: 8-step quarterly tuning process
- **Threat Model Scenarios**: T1-T4 adversarial patterns

### Changed
- Self-model validity now uses Bayesian posterior with hysteresis
- Identity coherence uses regularized Laplacian with small-N policy
- ReflexLog expanded to 18 fields including epoch roots

### Security
- Added No-Unbinding invariant proof
- Added audit completeness guarantees
- Added observer quorum Byzantine tolerance

## [1.0.0] - 2025-11-28

### Added
- Initial implementation of Booklet 8 self-modeling
- Core components: SelfModel, MetaRubric, RubricRepairEngine
- ReflexLog audit system with hash chain
- ObserverMesh quorum validation
- Reflexive swarm enhancements
- 65 base tests passing
- 8/8 G-criteria validated

## [0.1.0] - 2025-11-27

### Added
- Booklet 6 scaffold: drift surface, entropy governor
- Booklet 7 scaffold: swarm coherence, meta-kernel bridge
- Initial architecture design
