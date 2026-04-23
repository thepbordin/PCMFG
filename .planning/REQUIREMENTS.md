# Requirements: PCMFG Narrative Time-Series Normalization & Shape-Based Clustering

**Defined:** 2026-04-23
**Core Value:** Cross-narrative emotional trajectory comparison must work regardless of source length or medium.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Normalization

- [x] **NORM-01**: System maps each source narrative emotion time-series to a normalized progress axis in range [0.0, 1.0] with a configurable number of resampling points
- [x] **NORM-02**: System preserves event order while rescaling different source lengths into a comparable timeline
- [x] **NORM-03**: System handles variable-length input sequences without failing due to sequence size mismatch
- [x] **NORM-04**: System uses nearest-neighbor or rounded interpolation for ordinal 1-5 emotion scores (not linear interpolation which creates fractional meaningless values)

### DTW Distance

- [ ] **DTW-01**: System uses DTW as a supported distance metric for emotional time-series comparison via tslearn
- [ ] **DTW-02**: System supports Sakoe-Chiba warping constraint with configurable radius to prevent narratively absurd alignments
- [ ] **DTW-03**: System supports multi-dimensional DTW (9 emotions × 2 directions = 18 dimensions) per trajectory
- [ ] **DTW-04**: System allows switching between distance metrics (Euclidean, DTW, Soft-DTW) via configuration

### Shape-Based Clustering

- [ ] **CLST-01**: System clusters narratives by emotional trajectory shape similarity using TimeSeriesKMeans with DTW metric
- [ ] **CLST-02**: System produces cluster barycenters (DTW Barycenter Averaging) as representative "prototypical arcs" per cluster
- [ ] **CLST-03**: System outputs cluster assignments mapping each narrative to its cluster with interpretable labels
- [ ] **CLST-04**: System identifies at least two distinct pattern classes from sample data
- [ ] **CLST-05**: System is reproducible for the same inputs and configuration (random_state)

### Overlay Visualization

- [ ] **VIS-01**: System overlays two or more normalized emotional trajectories on the same axes at equivalent progress points
- [ ] **VIS-02**: System supports per-emotion overlay plots for all 9 base emotions
- [ ] **VIS-03**: System supports per-direction overlay (A→B vs B→A) for asymmetry analysis
- [ ] **VIS-04**: System visualizes cluster barycenters alongside member trajectories

### Integration

- [x] **INTG-01**: New normalization and DTW clustering modules are additive — existing SceneClusterer and TrajectoryClusterer continue working unchanged
- [x] **INTG-02**: System accepts existing AnalysisResult JSON files as input without modification
- [ ] **INTG-03**: tslearn >=0.8.1 added as a project dependency

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Enhanced Analysis

- **VIS-05**: DTW warping path visualization showing which narrative moments align between two stories
- **VIS-06**: Per-emotion distance matrix revealing which emotions drive inter-narrative similarity
- **CLST-06**: Recurring pattern identification with automatic emotion-based cluster labeling (e.g., "Trust-building arcs," "Hostility-to-passion flips")

## Out of Scope

| Feature | Reason |
|---------|--------|
| Z-normalization of emotion values | 1-5 ordinal scale has semantic meaning; Z-normalization destroys interpretability |
| Subsequence DTW (find matching scenes within narratives) | PCMFG chunks are irregular; existing SceneClusterer handles scene-level comparison |
| Derivative-based DTW (DDTW) | Emotion scores are integers 1-5; derivatives are mostly 0 with noisy jumps |
| Frequency-domain analysis (FFT/DWT) | Emotional trajectories are not periodic signals |
| Real-time/streaming DTW | PCMFG is explicitly batch/offline processing |
| PCA/t-SNE dimensionality reduction | Loses emotion-specific interpretability researchers need |
| Automatic optimal-k selection | With <20 narratives, silhouette scores are unreliable; k is a literary interpretive question |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| NORM-01 | Phase 1 | Complete |
| NORM-02 | Phase 1 | Complete |
| NORM-03 | Phase 1 | Complete |
| NORM-04 | Phase 1 | Complete |
| INTG-01 | Phase 1 | Complete |
| INTG-02 | Phase 1 | Complete |
| INTG-03 | Phase 1 | Pending |
| DTW-01 | Phase 2 | Pending |
| DTW-02 | Phase 2 | Pending |
| DTW-03 | Phase 2 | Pending |
| DTW-04 | Phase 2 | Pending |
| CLST-01 | Phase 2 | Pending |
| CLST-02 | Phase 2 | Pending |
| CLST-03 | Phase 2 | Pending |
| CLST-04 | Phase 2 | Pending |
| CLST-05 | Phase 2 | Pending |
| VIS-01 | Phase 3 | Pending |
| VIS-02 | Phase 3 | Pending |
| VIS-03 | Phase 3 | Pending |
| VIS-04 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 17 total
- Mapped to phases: 17
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-23*
*Last updated: 2026-04-23 after roadmap creation*
