# Phase 2: DTW Distance & Shape Clustering - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Cluster narratives by emotional arc shape similarity using DTW distance via tslearn. This operates on `NormalizedTrajectory` output from Phase 1's `NarrativeNormalizer` — it does NOT modify the existing pipeline, existing `SceneClusterer`, or existing `TrajectoryClusterer`.

Deliverables: pairwise DTW distance matrix computation, TimeSeriesKMeans clustering with DTW metric, DTW barycenter extraction, and configurable metric switching (Euclidean/DTW/Soft-DTW) with Sakoe-Chiba warping constraints.

</domain>

<decisions>
## Implementation Decisions

### Multi-dimensional DTW Strategy
- **D-01:** Use joint multivariate DTW on all 18 dimensions (9 emotions x 2 directions) as a single multivariate time series per narrative. tslearn's `TimeSeriesKMeans` natively supports multivariate input via shape `(n_samples, n_timestamps, n_features)`. This preserves cross-emotion and cross-direction correlations that independent per-emotion DTW would destroy.
- **D-02:** The 18 features are ordered: for each of the 9 `BASE_EMOTIONS`, first A_to_B then B_to_A. So feature axis is `[Joy_A2B, Joy_B2A, Trust_A2B, Trust_B2A, ...]`. This groups emotion pairs together for interpretability.
- **D-03:** The clustering input is constructed from `list[NormalizedTrajectory]` — group by source narrative, stack the 18 emotion arrays into a single `(100, 18)` matrix per narrative.

### DTW Warping Constraints
- **D-04:** Default Sakoe-Chiba radius: 20% of series length (i.e., `int(0.2 * n_points)`). For the default 100-point grid, this allows warping up to +/-20 positions — enough for meaningful emotional peak alignment but prevents narratively absurd mappings (e.g., mapping the opening to the finale). Configurable via parameter.
- **D-05:** No constraint (full DTW) is available as an option when `sakoe_chiba_radius=None`, but not the default. Full DTW risks pathological alignments especially on plateau-heavy data from forward-fill imputation.

### Distance Metric Configuration
- **D-06:** Support three metrics via string enum: `"euclidean"`, `"dtw"`, `"soft-dtw"`. Default to `"dtw"`. Soft-DTW is differentiable and avoids the "curse of winning" problem but may produce blurrier barycenters. Euclidean serves as a baseline comparison against DTW.
- **D-07:** Follow existing config patterns — metric choice and Sakoe-Chiba radius are constructor parameters on the clustering class (like `SceneClusterer` takes `algorithm`, `n_clusters`, `random_state`). No new YAML config section needed — these are programmatic parameters.

### Cluster Output Model
- **D-08:** New Pydantic model `DTWClusterResult` (in `pcmfg/models/schemas.py` or a new file — planner decides based on size). Contains: cluster labels per narrative, cluster barycenters (as normalized trajectory arrays), distance matrix, metric used, warping parameters, cluster sizes, and silhouette score.
- **D-09:** Cluster assignments map `source` (narrative identifier from `NormalizedTrajectory.source`) to cluster label. This is the primary output users need for grouping narratives.
- **D-10:** Barycenters are returned as numpy arrays of shape `(n_points, 18)` — one per cluster. Users can pass these directly to Phase 3 visualization. Barycenter computation uses DTW Barycenter Averaging (DBA) via tslearn.

### Pre-DTW Processing
- **D-11:** No Z-normalization of emotion values. The 1-5 ordinal scale has semantic meaning (per REQUIREMENTS.md out-of-scope decision). Emotion values go into DTW as-is after Phase 1 normalization.
- **D-12:** No derivative preprocessing. Emotion scores are integers 1-5; derivatives would be mostly 0 with noisy jumps (per REQUIREMENTS.md out-of-scope).

### Missing Direction Handling
- **D-13:** If a narrative only has one direction (e.g., A_to_B but no B_to_A), fill the missing direction with baseline (all 1s) to maintain the 18-dimensional structure. Log a warning. This keeps the multivariate shape consistent across all narratives.

### Module Structure
- **D-14:** New file `pcmfg/analysis/dtw_clusterer.py` for the DTW clustering logic. Sits alongside `clusterer.py` and `normalizer.py` in the `analysis/` package. Follows the class-based pattern established by `SceneClusterer` and `TrajectoryClusterer`.
- **D-15:** Zero modifications to `clusterer.py`, `normalizer.py`, or any existing pipeline files. DTW clustering is purely additive. Only `__init__.py` gets updated with new exports.

### Reproducibility
- **D-16:** All clustering accepts `random_state` parameter (default 42), matching existing `SceneClusterer` and `TrajectoryClusterer` patterns. Same inputs + same config = identical results.

### Claude's Discretion
- Exact Sakoe-Chiba default radius percentage (20% is recommended but planner may adjust based on tslearn docs)
- Whether `DTWClusterResult` lives in `schemas.py` or a separate file
- Internal helper method naming and API surface design
- Whether to provide a convenience method that takes `list[AnalysisResult]` directly (normalizing + clustering in one call) or keep normalize and cluster as separate steps
- Exact error handling for edge cases (single narrative, all-identical trajectories, etc.)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `NormalizedTrajectory` model (`schemas.py:266`): Per-emotion, per-direction resampled trajectory with `source`, `main_pairing`, `direction`, `emotion`, `x`, `y`, `original_length`, `n_points`
- `NarrativeNormalizer` (`normalizer.py:69`): Produces `list[NormalizedTrajectory]` from `AnalysisResult`. Has `normalize()` (single) and `normalize_all()` (batch) methods
- `BASE_EMOTIONS` constant (`schemas.py:32`): List of 9 emotion names, reusable for constructing the 18-dimensional feature axis
- `ClusterResult` model (`clusterer.py:50`): Existing Pydantic model for clustering results — new `DTWClusterResult` should follow similar structure
- `ClusteringAlgorithm` enum (`clusterer.py:42`): Pattern for metric enum — new `DistanceMetric` enum should follow same style
- `SKLEARN_AVAILABLE` pattern (`clusterer.py:33`): Graceful import handling — same pattern should be used for tslearn availability check

### Established Patterns
- Class-based API: `SceneClusterer(algorithm=..., n_clusters=..., random_state=...)` → `.cluster(features)` → `ClusterResult`
- Optional dependency guarding: try/except import with `*_AVAILABLE` flag and descriptive error message
- Pydantic models with `Field()` descriptors and `ConfigDict(frozen=True)` for immutable outputs
- Module-level `logger = logging.getLogger(__name__)` in every module
- Constructor parameters for all configurable options (not global config)
- `random_state` parameter for reproducibility on all stochastic operations

### Integration Points
- Input: `list[NormalizedTrajectory]` from `NarrativeNormalizer.normalize_all()`
- Output: `DTWClusterResult` → consumed by Phase 3 visualization
- `pcmfg/analysis/__init__.py` needs new exports for DTW clusterer classes
- `pyproject.toml` needs `tslearn >=0.8.1` confirmed (INTG-03 from Phase 1)

</code_context>

<specifics>
## Specific Ideas

- The STATE.md blocker on "Joint vs per-emotion DTW strategy for 18-dimensional trajectories" is resolved: joint multivariate DTW preserves cross-emotion correlations
- The STATE.md blocker on "Forward-fill plateau distortion" is acknowledged but not addressed in this phase — DTW will operate on the normalized output as-is. A future "raw mode" flag (skipping forward-fill before normalization) is deferred
- tslearn's `TimeSeriesKMeans` with `metric="dtw"` and `tslearn.metrics.dtw_path` for warping path visualization (deferred to VIS-05 in v2)

</specifics>

<deferred>
## Deferred Ideas

- **VIS-05**: DTW warping path visualization — showing which narrative moments align between two stories (v2 requirement)
- **VIS-06**: Per-emotion distance matrix revealing which emotions drive inter-narrative similarity (v2 requirement)
- **CLST-06**: Automatic emotion-based cluster labeling (e.g., "Trust-building arcs") (v2 requirement)
- **Raw mode flag**: Option to skip forward-fill imputation before normalization, reducing plateau distortion in DTW input
- **Subsequence DTW**: Finding matching emotional patterns within a single narrative (explicitly out of scope per REQUIREMENTS.md)
- **Automatic optimal-k selection**: With small narrative counts, silhouette scores are unreliable (explicitly out of scope per REQUIREMENTS.md)

</deferred>

---

*Phase: 02-dtw-distance-shape-clustering*
*Context gathered: 2026-04-23*
