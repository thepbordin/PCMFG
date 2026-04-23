# Feature Research: Narrative Time-Series Normalization & Shape-Based Clustering

**Domain:** Computational literary analysis — emotional trajectory comparison across narratives
**Researched:** 2026-04-23
**Confidence:** MEDIUM-HIGH (domain-specific features are novel; tslearn API verified from official docs)

## Feature Landscape

### Table Stakes (Users Expect These)

Features researchers/analysts assume exist in a cross-narrative emotional comparison system. Missing these = the system doesn't solve the stated core value.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Narrative progress normalization** | Cross-narrative comparison is impossible without a common axis. The `position` field (0.0-1.0) already exists per chunk but no resampling/interpolation to a common grid exists. | MEDIUM | Resample all trajectories to a uniform grid (e.g., 100 points) on [0.0, 1.0]. `tslearn.preprocessing.TimeSeriesResampler` handles this natively. |
| **DTW distance metric** | The whole point is shape-based comparison. Euclidean distance (current approach) fails when two narratives have the same shape but different durations or phase offsets. | LOW | `tslearn.metrics.dtw()` and `tslearn.clustering.TimeSeriesKMeans(metric="dtw")` provide this out of the box. |
| **Multi-emotion DTW distance** | PCMFG produces 9 emotions × 2 directions = 18 dimensions. DTW must operate on the full multidimensional vector, not one emotion at a time. | LOW | `tslearn` DTW supports multidimensional time-series natively (shape: `n_ts, sz, d` where d=18). |
| **Overlay comparison visualization** | Researchers need to see two or more normalized trajectories on the same axes to visually assess similarity. The existing `plot_comparison()` method only plots two analyses side-by-side, not overlaid on a common normalized grid. | MEDIUM | Extend `EmotionPlotter` with overlay method. Already has matplotlib infrastructure and EMOTION_CONFIG. |
| **Cluster assignment output** | After clustering, the system must tell you which narratives belong to which cluster and provide cluster centroids (representative shapes). | LOW | `TimeSeriesKMeans` returns `labels_` and `cluster_centers_` directly. Existing `ClusterResult` Pydantic model is close to what's needed. |
| **Backward-compatible module** | Existing `SceneClusterer` and `TrajectoryClusterer` must continue working. New shape-based clustering is additive, not replacing. | MEDIUM | New `TrajectoryShapeClusterer` class in `clusterer.py` that operates on normalized time-series, not statistical features. |
| **Configurable distance metric** | Analysts need to compare DTW vs Euclidean vs Soft-DTW to understand which metric produces the most meaningful narrative groupings. | LOW | `TimeSeriesKMeans` accepts `metric="euclidean"`, `"dtw"`, or `"softdtw"` as a parameter. Pass through from config. |

### Differentiators (Competitive Advantage)

Features that make PCMFG uniquely valuable for narrative analysis — not found in generic time-series tooling.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Directed emotion overlay** | Plot A→B and B→A as separate overlaid trajectories across narratives, revealing asymmetry patterns (e.g., "slow-burn where A falls first, B follows"). No generic time-series tool does this because they don't have the directed relationship concept. | MEDIUM | Extend overlay visualization to support per-direction filtering. Already have `plot_directional_comparison()` as reference. |
| **DTW barycenter (average trajectory)** | `tslearn.barycenters.dtw_barycenter_averaging` computes the "average shape" of a cluster — a representative emotional arc. This is the narrative equivalent of "show me the prototypical enemies-to-lovers arc." | LOW | Direct call to `dtw_barycenter_averaging(X, barycenter_size=100)`. One-line integration. |
| **Recurring pattern identification with emotion labels** | Automatically label clusters with dominant emotion patterns (e.g., "Trust-building arcs," "Hostility-to-passion flips") by analyzing cluster centroids' emotion profiles. | HIGH | Requires combining DTW clustering centroids with the existing `_describe_centroid()` logic from `SceneClusterer`, adapted for full trajectory centroids. |
| **DTW warping path visualization** | Show the optimal alignment between two trajectories — which narrative moments "match" across stories. This is powerful for literary analysis (e.g., "the confrontation scene in novel A aligns with chapter 8 of novel B"). | MEDIUM | `tslearn.metrics.dtw_path()` returns the alignment path. Visualize as a heatmap or connected lines between two overlaid trajectories. |
| **Warping constraint (Sakoe-Chiba)** | For narrative analysis, you don't want DTW to warp a scene at 10% progress to one at 90% — that's narratively absurd. Sakoe-Chiba band limits warping to ±k% of narrative progress, preserving temporal coherence. | MEDIUM | `tslearn.metrics.dtw(s1, s2, global_constraint="sakoe_chiba", sakoe_chiba_radius=0.1)` constrains alignment to ±10% of timeline. Critical for narrative validity. |
| **Emotion-specific DTW distance** | Compute DTW per emotion and produce a per-emotion distance matrix, revealing which emotions drive similarity (e.g., "novels A and B cluster together because their Trust arcs match, even though Anger differs"). | MEDIUM | Loop over 9 emotions, compute DTW per emotion, compare distance matrices. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem useful for time-series analysis but are wrong for narrative emotional trajectories.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Z-normalization of emotion values** | Standard practice in time-series DTW to normalize amplitude. | PCMFG uses a 1-5 ordinal scale with semantic meaning. Z-normalizing destroys the baseline=1 concept and makes scores non-interpretable ("what does a Trust score of -0.3 mean?"). | Min-max normalization to [0,1] preserves ordinal relationships. Or no amplitude normalization at all — DTW handles scale differences, and the 1-5 scale is already bounded. |
| **Automatic optimal-k selection** | Generic clustering tools often auto-select k via silhouette analysis. | With <20 narratives (typical corpus size), silhouette scores are noisy and unreliable. The "right" number of narrative pattern clusters is a literary interpretive question, not a statistical one. | Provide silhouette scores as advisory only. Default to k=3 (common in narratology: positive arc, negative arc, flat arc) and let the analyst override. |
| **Real-time/progressive DTW** | Some time-series tools support streaming DTW for live data. | PCMFG is explicitly batch/offline processing (stated in PROJECT.md constraints). Narratives are analyzed in full, not streamed. | N/A — out of scope per project constraints. |
| **PCA/t-SNE dimensionality reduction for visualization** | Common technique to visualize high-dimensional time-series clusters in 2D. | Reducing 18 emotion dimensions to 2D loses the emotion-specific interpretability that researchers need. A cluster might look tight in 2D but have wildly different Joy vs Anger profiles. | Instead: visualize per-emotion overlay plots and provide per-emotion distance matrices. Preserve interpretability. |
| **Subsequence DTW (find matching scenes within narratives)** | Powerful for generic time-series — find the best-matching subsequence of one series within another. | Tempting for "find the equivalent confrontation scene," but PCMFG chunks are irregular (different lengths, different numbers per narrative). Subsequence DTW would match across chunks with no narrative coherence. | If scene-level matching is needed, use the existing `SceneClusterer` which compares scenes by emotional profile, not temporal alignment. |
| **Derivative-based DTW (DDTW)** | Compares the shape of the derivative (rate of change) rather than raw values. | PCMFG emotion scores are integers 1-5. The derivative is mostly 0 with occasional jumps of ±1. DDTW on discrete ordinal data produces noisy, unreliable alignments. | Use raw DTW on the normalized trajectory. If trend detection matters, the existing DELTA feature type already captures emotion changes. |
| **Frequency-domain analysis (FFT/DWT)** | Common in signal processing for pattern discovery. | Emotional trajectories are not periodic signals. Applying frequency analysis to narrative arcs is academically unfounded and produces meaningless results. | Stick to time-domain DTW and statistical features. |

## Feature Dependencies

```
[Narrative Progress Normalization]
    └──requires──> [Common Grid Resampling]
                       └──required by──> [DTW Distance Computation]
                                            └──required by──> [Shape-Based Clustering]
                                                                  └──required by──> [Cluster Barycenters]
                                                                  └──required by──> [Pattern Labeling]

[Overlay Visualization]
    └──requires──> [Narrative Progress Normalization]
    └──enhances──> [Shape-Based Clustering]

[Warping Constraint (Sakoe-Chiba)]
    └──enhances──> [DTW Distance Computation]

[Emotion-Specific DTW Distance]
    └──requires──> [DTW Distance Computation]

[DTW Warping Path Visualization]
    └──requires──> [DTW Distance Computation]

[Pattern Labeling]
    └──requires──> [Shape-Based Clustering]
    └──requires──> [Cluster Barycenters]
    └──enhances──> [Overlay Visualization]

[Configurable Distance Metric]
    └──enhances──> [Shape-Based Clustering]
```

### Dependency Notes

- **Narrative Progress Normalization requires Common Grid Resampling:** Before DTW can compare shapes, all trajectories must be resampled to the same number of points on [0.0, 1.0]. Without this, DTW handles variable length (tslearn supports it natively), but overlay visualization requires a common grid. **Decision point:** use DTW on variable-length data directly (more accurate) vs normalize to common grid first (enables overlay). Recommendation: do both — normalize for visualization, use raw variable-length for DTW.
- **Pattern Labeling requires Cluster Barycenters:** To generate human-readable labels like "Trust-building arc," we need the average shape (barycenter) of each cluster to analyze dominant emotion patterns.
- **Warping Constraint enhances DTW:** Without constraints, DTW might match narrative progress 0.1 to 0.9, which is narratively meaningless. Sakoe-Chiba band is strongly recommended for this domain.
- **Emotion-Specific DTW enhances but is not required by base clustering:** Base DTW operates on all 18 dimensions simultaneously. Per-emotion analysis is an exploratory layer on top.

## MVP Definition

### Launch With (v1)

Minimum features to validate "cross-narrative emotional trajectory comparison works."

- [x] **Narrative progress normalization** — Resample all trajectories to uniform [0.0, 1.0] grid. Without this, no cross-narrative comparison is possible.
- [x] **DTW distance metric with TimeSeriesKMeans** — `tslearn.clustering.TimeSeriesKMeans(metric="dtw")` on normalized trajectories. This is the core new capability.
- [x] **Multi-emotion DTW** — Full 18-dimensional DTW (9 emotions × 2 directions). Not one emotion at a time.
- [x] **Overlay comparison visualization** — Plot 2+ normalized trajectories on same axes per emotion. Must work with the existing `EmotionPlotter` infrastructure.
- [x] **Configurable metric (Euclidean vs DTW)** — Allow switching to compare results.
- [x] **Cluster output with barycenters** — `TimeSeriesKMeans` cluster_centers_ are DTW barycenters by default.

### Add After Validation (v1.x)

Features to add once core clustering and visualization work on real data.

- [ ] **Warping constraint (Sakoe-Chiba)** — Add after validating that unconstrained DTW produces narratively unreasonable alignments. Trigger: analyst reports "these two narratives match but the alignment makes no sense."
- [ ] **Soft-DTW alternative** — `metric="softdtw"` provides differentiable clustering. Useful if DBA (DTW Barycenter Averaging) convergence is problematic. Trigger: DBA barycenters look noisy or don't converge.
- [ ] **DTW warping path visualization** — Add after basic overlay works. Shows which narrative moments align between two stories. Trigger: analyst asks "which scenes correspond across these two novels?"
- [ ] **Directed emotion overlay** — Separate A→B and B→A overlay plots. Trigger: analyst needs to compare asymmetry patterns across narratives.
- [ ] **Emotion-specific distance matrix** — Per-emotion DTW distances. Trigger: analyst asks "which emotion drives the similarity between these clusters?"

### Future Consideration (v2+)

Features requiring significant additional research or infrastructure.

- [ ] **Recurring pattern identification with emotion labels** — Automated cluster labeling. Requires defining a vocabulary of narrative emotional patterns. HIGH complexity because it's partly a literary theory problem, not just engineering.
- [ ] **Canonical Time Warping (CTW)** — `tslearn.metrics.ctw()` learns optimal rotation before warping. Could help when different emotion axes are correlated differently across narratives. Trigger: DTW clusters seem driven by emotion scale differences rather than shape.
- [ ] **Kernel K-Means with GAK** — `tslearn.clustering.KernelKMeans(kernel="gak")` offers non-linear clustering. Could reveal patterns DTW misses. Trigger: DTW produces only 1-2 meaningful clusters but analyst suspects more nuanced groupings exist.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Narrative progress normalization | HIGH | MEDIUM | P1 |
| DTW distance with TimeSeriesKMeans | HIGH | LOW | P1 |
| Multi-emotion DTW (18-dim) | HIGH | LOW | P1 |
| Overlay comparison visualization | HIGH | MEDIUM | P1 |
| Configurable metric (Euclidean/DTW/Soft-DTW) | MEDIUM | LOW | P1 |
| Cluster output with barycenters | MEDIUM | LOW | P1 |
| Warping constraint (Sakoe-Chiba) | MEDIUM | LOW | P2 |
| DTW warping path visualization | MEDIUM | MEDIUM | P2 |
| Directed emotion overlay | MEDIUM | MEDIUM | P2 |
| Emotion-specific distance matrix | MEDIUM | MEDIUM | P2 |
| Soft-DTW alternative | LOW | LOW | P2 |
| Recurring pattern labeling | HIGH | HIGH | P3 |
| Canonical Time Warping | LOW | HIGH | P3 |
| Kernel K-Means (GAK) | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for cross-narrative comparison to work
- P2: Adds analytical depth, build after P1 is validated
- P3: Advanced/exploratory, defer until product value is established

## Narrative-Specific Design Decisions

These decisions differentiate this system from generic time-series clustering:

### 1. Normalization Strategy: Resample to Common Grid, Not Z-Normalize

**Decision:** Resample all trajectories to N points on [0.0, 1.0] using linear interpolation. Do NOT z-normalize emotion values.

**Why:** The 1-5 scale has ordinal semantic meaning. "Trust = 4" means something specific. Z-normalizing would destroy interpretability. The X-axis (narrative progress) is already semantically normalized (0.0 = start, 1.0 = end). Only the sampling density needs alignment.

**Implementation:** Use `numpy.interp()` for resampling or `tslearn.preprocessing.TimeSeriesResampler(sz=N)`.

### 2. DTW Constraints: Sakoe-Chiba Band Recommended

**Decision:** Default to Sakoe-Chiba radius of 10-15% of narrative length. Allow disabling for experimentation.

**Why:** Without constraints, DTW can warp the beginning of one narrative to the end of another. For narrative analysis, temporal ordering is meaningful — a "falling in love" scene at 20% progress should not align with a "falling in love" scene at 80% progress, because their narrative consequences are different.

**Implementation:** `tslearn.metrics.dtw(s1, s2, global_constraint="sakoe_chiba", sakoe_chiba_radius=int(0.1 * sz))`

### 3. Clustering Granularity: Per-Novel, Not Per-Scene

**Decision:** The new shape-based clustering operates at the **novel level** — each novel is one time-series to be clustered. This complements the existing `SceneClusterer` which operates at the chunk level.

**Why:** The core value is "cross-narrative comparison." Clustering individual scenes across novels conflates local emotional moments with global arc structure. The existing system already handles scene-level clustering. The new feature should handle narrative-level shape clustering.

**Implementation:** Each `AnalysisResult` produces one normalized trajectory per direction per emotion. For clustering, concatenate all 9 emotions into an 18-dimensional time-series (or operate per-emotion separately for analysis).

### 4. Barycenter as "Prototypical Arc"

**Decision:** DTW barycenter averaging (DBA) produces the cluster centroid — interpret this as the "prototypical emotional arc" for that narrative pattern class.

**Why:** This is the most interpretable output for literary researchers. "Show me the average enemies-to-lovers arc" is a natural research question. The barycenter IS the answer.

**Implementation:** `tslearn.barycenters.dtw_barycenter_averaging(X_cluster, barycenter_size=N)` — direct pass-through.

## Competitor Feature Analysis

There are no direct competitors doing "DTW-based emotional trajectory clustering for romance narratives." The comparison is against generic time-series tools and narrative analysis approaches:

| Feature | Generic TS Tools (tslearn) | Existing PCMFG | Our Approach |
|---------|---------------------------|----------------|--------------|
| Distance metric | DTW, Soft-DTW, GAK, LCSS, Frechet | Euclidean (statistical features) | DTW with Sakoe-Chiba constraint, configurable |
| Normalization | Z-normalize, min-max | None (raw chunk values) | Narrative progress resampling, no amplitude normalization |
| Visualization | Generic time-series plots | Per-novel emotion grids, directional comparison | Multi-narrative overlay on normalized grid |
| Interpretability | Cluster labels are numeric | Emotion-based scene descriptions | Barycenter as prototypical arc + emotion pattern labels |
| Domain awareness | None (generic) | Romance-specific 9-emotion model | Leverages directed emotions (A→B vs B→A) for asymmetry analysis |

## Sources

- **tslearn 0.8.1 official documentation** — https://tslearn.readthedocs.io/en/stable/ (HIGH confidence — verified API signatures for `TimeSeriesKMeans`, `dtw()`, `TimeSeriesResampler`, `dtw_barycenter_averaging`)
- **tslearn PyPI** — https://pypi.org/project/tslearn/ (HIGH confidence — version 0.8.1, released Mar 2026, Python 3.10+, BSD-2-Clause)
- **dtaidistance documentation** — https://dtaidistance.readthedocs.io/en/latest/ (MEDIUM confidence — alternative library, has C-optimized DTW and built-in hierarchical/KMeans clustering with DTW)
- **Existing PCMFG codebase** — `clusterer.py`, `feature_extractor.py`, `plotter.py`, `schemas.py` (HIGH confidence — directly inspected)
- **PROJECT.md** — Project constraints and requirements (HIGH confidence — primary specification)

---
*Feature research for: Narrative time-series normalization & shape-based clustering*
*Researched: 2026-04-23*
