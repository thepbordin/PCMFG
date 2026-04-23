# Project Research Summary

**Project:** PCMFG — Narrative Time-Series Normalization & Shape-Based Clustering
**Domain:** Computational literary analysis — DTW-based emotional trajectory comparison across narratives
**Researched:** 2026-04-23
**Confidence:** HIGH

## Executive Summary

PCMFG needs to compare emotional trajectories across narratives of different lengths (novels, novellas, films). The existing pipeline produces per-chunk emotion scores on a 1-5 ordinal scale with a `position` field (0.0-1.0), but has no mechanism to normalize variable-length time-series to a common grid or compare shapes using temporally-aware distance metrics. The recommended approach is a **post-pipeline processing layer** — completely separate from the existing Phase 1→2→3 extraction pipeline — that (1) resamples all trajectories to a uniform N-point grid using scipy interpolation, (2) computes pairwise DTW distances via tslearn, and (3) clusters by shape using tslearn's `TimeSeriesKMeans` with DTW Barycenter Averaging (DBA) for correct centroid computation.

The stack addition is minimal: **one new dependency** (`tslearn>=0.8.1`). Everything else — scipy for interpolation, scikit-learn for pipeline integration, matplotlib for overlay visualization, numpy/pandas for data handling — is already present. The architecture adds four new modules (`NarrativeNormalizer`, `DTWDistanceComputer`, `ShapeClusterer`, `OverlayPlotter`) that consume `AnalysisResult` outputs without modifying the existing pipeline.

The key risks are (1) **ordinal scale violation** — linear interpolation creates semantically meaningless fractional emotion scores (2.3 Joy), which can be mitigated by nearest-neighbor resampling or explicit documentation of the simplification, (2) **forward-fill plateau distortion** — the existing imputation creates flat segments that DTW treats as real emotional data, requiring a "raw mode" that excludes forward-filled segments for cross-narrative comparison, and (3) **unconstrained DTW warping** — without Sakoe-Chiba constraints, DTW produces narratively absurd alignments (matching a scene at 10% progress to one at 90%).

## Key Findings

### Recommended Stack

Only one new dependency is needed. The existing Python scientific stack (numpy, pandas, scipy, scikit-learn, matplotlib, pydantic) covers all other needs.

**Core technologies:**
- **tslearn>=0.8.1**: DTW computation, `TimeSeriesKMeans` clustering with DBA centroids, variable-length time-series support — chosen over `dtaidistance` (narrower scope, no scikit-learn integration) and `fastdtw` (abandoned since 2019, no Python 3.12+ wheels)
- **scipy.interpolate.interp1d**: Resampling variable-length trajectories to uniform [0.0, 1.0] grid — already available as a transitive dependency via scikit-learn; chosen over tslearn's `TimeSeriesResampler` because it handles non-uniform x-values (per-chunk positions) natively

### Expected Features

**Must have (table stakes):**
- Narrative progress normalization — resample all trajectories to uniform [0.0, 1.0] grid (cross-narrative comparison is impossible without this)
- DTW distance metric with `TimeSeriesKMeans` — shape-based clustering that aligns peaks/valleys temporally
- Multi-emotion DTW (18-dimensional: 9 emotions x 2 directions) — tslearn supports this natively
- Overlay comparison visualization — plot 2+ normalized trajectories on same axes per emotion
- Configurable metric (Euclidean vs DTW vs Soft-DTW) — analysts need to compare approaches
- Cluster output with DTW barycenters — "prototypical emotional arc" per cluster

**Should have (competitive differentiators):**
- Directed emotion overlay — separate A→B and B→A trajectories revealing asymmetry patterns
- DTW warping path visualization — shows which narrative moments align across stories
- Warping constraint (Sakoe-Chiba) — prevents narratively absurd temporal alignments
- Emotion-specific distance matrix — reveals which emotions drive cluster similarity

**Defer (v2+):**
- Recurring pattern identification with emotion labels (HIGH complexity, partly a literary theory problem)
- Canonical Time Warping (CTW) — advanced, niche use case
- Kernel K-Means with GAK — exploratory alternative clustering

### Architecture Approach

The new functionality is a **post-pipeline processing layer** that consumes `AnalysisResult` outputs from the existing 3-phase extraction pipeline. This is the natural boundary because normalization is a cross-narrative concern (meaningless for a single narrative) and the existing pipeline should remain untouched for backward compatibility.

**Major components:**
1. **NarrativeNormalizer** (`pcmfg/analysis/normalizer.py`) — Resamples variable-length emotion time-series to uniform N-point grid on [0.0, 1.0] using scipy interpolation. Outputs `NormalizedTrajectory` model.
2. **DTWDistanceComputer** (`pcmfg/analysis/dtw_distance.py`) — Computes pairwise DTW distance matrix from normalized trajectories. Supports configurable metrics and Sakoe-Chiba constraints.
3. **ShapeClusterer** (`pcmfg/analysis/dtw_clusterer.py`) — Clusters trajectories by shape using tslearn's `TimeSeriesKMeans` (with DBA centroids). Coexists with existing `SceneClusterer` and `TrajectoryClusterer`.
4. **OverlayPlotter** (`pcmfg/visualization/overlay.py`) — Visualizes normalized trajectories overlaid on common axes with optional cluster coloring.

### Critical Pitfalls

1. **Linear interpolation on ordinal data** — Creates meaningless fractional emotion scores (2.3 Joy). Mitigate with nearest-neighbor resampling or document as deliberate simplification. Address in normalization phase.
2. **Forward-fill plateau distortion** — Long flat segments from character-absent scenes get treated as real emotional data by DTW. Mitigate by flagging forward-filled data and providing a "raw mode" that excludes imputed segments. Address in normalization phase.
3. **Unconstrained DTW produces nonsensical alignments** — Without Sakoe-Chiba band, DTW warps beginning of one narrative to end of another. Default to Sakoe-Chiba radius of 10-15%. Address in clustering phase.
4. **Joint 9-emotion DTW dominated by high-variance emotions** — Sparse ordinal data means DTW is driven by whichever emotions happen to be active. Mitigate with per-emotion z-normalization before concatenation or per-emotion DTW with aggregation. Address in clustering phase.
5. **Wrong centroid computation** — Using sklearn's `KMeans` on DTW distance matrices produces arithmetic mean centroids, not DTW barycenters. Must use tslearn's `TimeSeriesKMeans(metric="dtw")` which uses DBA internally. Address in clustering phase.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Normalization Foundation
**Rationale:** All downstream components (DTW, clustering, visualization) consume `NormalizedTrajectory`. This must come first. It also forces early resolution of the interpolation strategy and forward-fill handling — the two most consequential design decisions flagged by pitfalls research.
**Delivers:** `NormalizedTrajectory` model, `NarrativeNormalizer` class, `NormalizationConfig`, unit tests
**Addresses:** Narrative progress normalization, common grid resampling (FEATURES P1)
**Avoids:** Pitfall 1 (ordinal interpolation) and Pitfall 2 (forward-fill plateaus)

### Phase 2: DTW Distance Computation
**Rationale:** Depends on normalized trajectories from Phase 1. Adding tslearn dependency and implementing the distance layer is self-contained — no visualization or clustering needed yet.
**Delivers:** `DTWDistanceComputer`, `DTWConfig`, pairwise distance matrix computation, integration tests
**Uses:** tslearn>=0.8.1 (STACK), `NormalizedTrajectory` from Phase 1
**Implements:** DTW distance metric, configurable metrics, Sakoe-Chiba constraints (FEATURES P1)
**Avoids:** Pitfall 3 (unconstrained DTW), Pitfall 4 (joint emotion dominance)

### Phase 3: Shape-Based Clustering
**Rationale:** Depends on distance matrix from Phase 2. This is where the core analytical value materializes — grouping narratives by emotional arc shape.
**Delivers:** `ShapeClusterer`, `ShapeClusterResult`, silhouette scoring, cluster barycenters, integration tests
**Uses:** tslearn `TimeSeriesKMeans` with DBA (STACK)
**Implements:** DTW clustering with barycenters, optimal-k analysis (FEATURES P1)
**Avoids:** Pitfall 5 (wrong centroid computation) and Pitfall 6 (unstable centroids)

### Phase 4: Overlay Visualization
**Rationale:** Can be developed in parallel with Phase 2/3 (only depends on `NormalizedTrajectory` from Phase 1), but logically follows clustering so overlay plots can show cluster coloring. This is the primary user-facing output.
**Delivers:** `OverlayPlotter`, multi-trajectory overlay plots, cluster comparison plots, dendrogram visualization
**Uses:** matplotlib (existing), `NormalizedTrajectory` and `ShapeClusterResult` from prior phases
**Implements:** Overlay comparison visualization (FEATURES P1)

### Phase 5: Integration & CLI
**Rationale:** Wires everything together — end-to-end pipeline from text input to clustered overlay visualization. Includes CLI subcommands and public API exports.
**Delivers:** CLI subcommands, `__init__.py` exports, end-to-end integration test, documentation
**Uses:** All prior components

### Phase Ordering Rationale

- **Phase 1 is the gate** — every other component depends on `NormalizedTrajectory`. Building it first forces resolution of the interpolation strategy (Pitfall 1) and forward-fill handling (Pitfall 2) before they propagate into DTW and clustering.
- **Phases 2 and 3 are tightly coupled** — DTW distance is meaningless without clustering, and clustering needs the distance layer. Keeping them sequential ensures each layer is tested in isolation.
- **Phase 4 can partially parallel Phase 2/3** — overlay visualization only needs normalized trajectories, not cluster results. But delivering it after clustering enables cluster-colored overlays, which is more valuable.
- **Phase 5 is the integration gate** — nothing ships without end-to-end validation.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Interpolation strategy decision (nearest-neighbor vs linear vs documented simplification) — this is a domain-specific decision with no clear "right answer" in the literature. Needs `gsd-discuss-phase` to resolve.
- **Phase 2:** Multi-dimensional vs per-emotion DTW strategy — tslearn supports both, but the choice affects clustering quality. Needs experimentation during planning.

Phases with standard patterns (skip research-phase):
- **Phase 3:** tslearn `TimeSeriesKMeans` is well-documented with standard API. The pitfall (use DBA not arithmetic mean) is clearly resolved by using tslearn.
- **Phase 4:** matplotlib overlay plotting is straightforward with existing infrastructure.
- **Phase 5:** Standard CLI integration pattern.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | tslearn verified via PyPI (v0.8.1, Mar 2026), official docs, GitHub (3.1k stars). scipy interpolation is standard practice. |
| Features | MEDIUM-HIGH | tslearn API verified from official docs. Domain-specific features (narrative overlay, directed emotion analysis) are novel but well-reasoned. Anti-features (Z-normalization, subsequence DTW, DDTW) have clear justifications. |
| Architecture | HIGH | Clean separation from existing pipeline. Component boundaries are well-defined. Data flow is linear and testable. Build order has clear dependency chain. |
| Pitfalls | MEDIUM-HIGH | All pitfalls derived from verified tslearn docs and direct codebase inspection. Ordinal interpolation concern is domain-specific and lacks authoritative literature — flagged as "document decision" rather than "avoid entirely." |

**Overall confidence:** HIGH

### Gaps to Address

- **Interpolation strategy for ordinal data:** No authoritative source on how to handle interpolation of 1-5 ordinal emotion scores for DTW. Research identifies the problem clearly but the resolution (nearest-neighbor vs linear vs documented simplification) requires a deliberate decision during Phase 1 planning. Recommend `gsd-discuss-phase` to surface assumptions.
- **Joint vs per-emotion DTW tradeoff:** Research identifies the problem (high-variance emotions dominating joint DTW) but doesn't prescribe a single solution. Both approaches (per-emotion with aggregation vs z-normalized joint) are valid. Needs experimentation during Phase 2 planning.
- **Forward-fill cap threshold:** Research recommends capping consecutive forward-filled chunks at 3-5, but the exact number is a heuristic. Needs validation against real narrative data during Phase 1 implementation.
- **Sakoe-Chiba radius tuning:** Research recommends 10-15% of narrative length as default, but optimal value depends on narrative type. Needs empirical validation during Phase 2.

## Sources

### Primary (HIGH confidence)
- tslearn PyPI — v0.8.1, Mar 2026, Python 3.10+, BSD-2-Clause
- tslearn GitHub — 3.1k stars, 1,750 commits, active development
- tslearn official documentation — API reference for TimeSeriesKMeans, dtw(), TimeSeriesResampler, dtw_barycenter_averaging
- scipy PyPI — v1.17.1, Feb 2026, already transitive dependency
- PCMFG codebase — direct inspection of clusterer.py, synthesizer.py, schemas.py, plotter.py, feature_extractor.py
- PROJECT.md — project constraints and requirements

### Secondary (MEDIUM confidence)
- dtaidistance PyPI — v2.4.0, Feb 2026 (alternative library, not recommended)
- fastdtw PyPI — v0.3.4, Oct 2019, abandoned (anti-recommendation verified)
- Sakoe & Chiba (1978) — original Sakoe-Chiba band constraint
- Petitjean et al. (2011) — DTW Barycenter Averaging (DBA)
- Ordinal data interpolation best practices — general statistical knowledge, not verified against narrative analysis literature

### Tertiary (LOW confidence)
- Herrmann & Webb (2023) — Amerced DTW (ADTW) — recent paper, mentioned as future alternative
- Optimal k selection for small corpora — literary theory insight, not statistically rigorous

---
*Research completed: 2026-04-23*
*Ready for roadmap: yes*
