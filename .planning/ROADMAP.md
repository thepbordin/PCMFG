# Roadmap: PCMFG Narrative Normalization & Shape Clustering

## Overview

Extends the existing PCMFG pipeline with three new capabilities layered on top of current outputs: (1) normalize variable-length emotion time-series to a uniform grid so narratives of any length become directly comparable, (2) cluster narratives by emotional arc shape using DTW distance instead of raw statistical features, and (3) visualize multiple normalized trajectories overlaid on the same timeline with cluster coloring. Every new module is purely additive — existing SceneClusterer, TrajectoryClusterer, and Phase 1-3 pipeline remain untouched.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Normalization Foundation** - Resample any-length emotion time-series to a uniform [0.0, 1.0] grid using nearest-neighbor interpolation
- [ ] **Phase 2: DTW Distance & Shape Clustering** - Cluster narratives by emotional arc shape using DTW distance via tslearn
- [ ] **Phase 3: Overlay Visualization** - Overlay multiple normalized emotional trajectories on the same axes with cluster coloring

## Phase Details

### Phase 1: Normalization Foundation
**Goal**: Any PCMFG emotion time-series can be resampled to a uniform N-point grid on [0.0, 1.0], enabling direct cross-narrative comparison regardless of source length.
**Depends on**: Nothing (first phase)
**Requirements**: NORM-01, NORM-02, NORM-03, NORM-04, INTG-01, INTG-02, INTG-03
**Success Criteria** (what must be TRUE):
  1. User can load an existing AnalysisResult JSON and produce a NormalizedTrajectory with a configurable number of resampling points
  2. Normalized trajectories from narratives of different lengths (e.g., 47 chunks vs 12 chunks) produce arrays of identical length
  3. Ordinal emotion scores remain integers (1-5) after resampling — no fractional values like 2.3 Joy
  4. System handles variable-length input sequences without throwing sequence size mismatch errors
  5. Existing SceneClusterer, TrajectoryClusterer, and Phase 1-3 pipeline continue working unchanged
**Plans:** 2 plans

Plans:
- [ ] 01-01-PLAN.md — Data model contract, tslearn dependency, and test scaffold
- [ ] 01-02-PLAN.md — NarrativeNormalizer implementation, exports, and full test verification

### Phase 2: DTW Distance & Shape Clustering
**Goal**: Narratives cluster by emotional arc shape similarity, preserving temporal alignment of emotional peaks and valleys across different story lengths.
**Depends on**: Phase 1
**Requirements**: DTW-01, DTW-02, DTW-03, DTW-04, CLST-01, CLST-02, CLST-03, CLST-04, CLST-05
**Success Criteria** (what must be TRUE):
  1. User can compute a pairwise DTW distance matrix for a set of normalized trajectories (multi-dimensional: 9 emotions x 2 directions)
  2. User can cluster narratives into k groups using TimeSeriesKMeans with DTW metric and retrieve cluster assignments mapping each narrative to its cluster
  3. User can retrieve a DTW barycenter (prototypical emotional arc) for each cluster
  4. User can switch between Euclidean, DTW, and Soft-DTW distance metrics and configure Sakoe-Chiba warping radius via configuration
  5. Running the same clustering with the same inputs and configuration produces identical results (reproducible with random_state)
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD
- [ ] 02-03: TBD

### Phase 3: Overlay Visualization
**Goal**: Users can visually compare emotional trajectories from multiple narratives aligned on the same normalized timeline, with cluster coloring and barycenter overlays.
**Depends on**: Phase 2
**Requirements**: VIS-01, VIS-02, VIS-03, VIS-04
**Success Criteria** (what must be TRUE):
  1. User can overlay two or more normalized emotional trajectories on the same plot at equivalent progress points
  2. User can generate per-emotion overlay plots for any of the 9 base emotions
  3. User can generate separate A-to-B and B-to-A direction overlay plots for asymmetry analysis
  4. User can visualize cluster barycenters alongside their member trajectories on the same axes
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Normalization Foundation | 0/3 | Not started | - |
| 2. DTW Distance & Shape Clustering | 0/3 | Not started | - |
| 3. Overlay Visualization | 0/2 | Not started | - |
