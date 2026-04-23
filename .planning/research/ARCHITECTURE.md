# Architecture Patterns

**Domain:** Narrative time-series normalization and DTW-based shape clustering
**Researched:** 2026-04-23

## Recommended Architecture

```
EXISTING PIPELINE (unchanged)
==============================

INPUT TEXT → Phase 1 (World Builder) → Phase 2 (Emotion Extraction) → Phase 3 (Synthesizer)
                                                                         ↓
                                                                  AnalysisResult
                                                                  (timeseries:
                                                                   A_to_B, B_to_A
                                                                   + chunks with
                                                                   position 0.0-1.0)
                                                                         ↓
                                                    ┌────────────────────────────────────┐
NEW: Post-Pipeline Processing Layer   ─────────────│  pcmfg/analysis/normalizer.py      │
                                                    │  (NarrativeNormalizer)             │
                                                    └────────────────────────────────────┘
                                                                         ↓
                                                    ┌────────────────────────────────────┐
NEW: DTW Distance Computation     ─────────────────│  pcmfg/analysis/dtw_distance.py     │
                                                    │  (DTWDistanceComputer)              │
                                                    └────────────────────────────────────┘
                                                                         ↓
                                                    ┌────────────────────────────────────┐
NEW: DTW Clustering                ─────────────────│  pcmfg/analysis/dtw_clusterer.py    │
                                                    │  (ShapeClusterer)                   │
                                                    └────────────────────────────────────┘
                                                                         ↓
                                                    ┌────────────────────────────────────┐
NEW: Overlay Visualization          ───────────────│  pcmfg/visualization/overlay.py     │
                                                    │  (OverlayPlotter)                   │
                                                    └────────────────────────────────────┘
```

## Component Boundaries

| Component | Responsibility | Input | Output | Communicates With |
|-----------|---------------|-------|--------|-------------------|
| `NarrativeNormalizer` | Resample variable-length time-series to uniform N-point grid on [0.0, 1.0] | `AnalysisResult` | `NormalizedTrajectory` | `Synthesizer` output only (read) |
| `DTWDistanceComputer` | Compute pairwise DTW distance matrix from normalized trajectories | `list[NormalizedTrajectory]` | `NDArray` distance matrix | `NarrativeNormalizer` output only (read) |
| `ShapeClusterer` | Cluster trajectories by shape using DTW distance | Distance matrix + `list[AnalysisResult]` | `ShapeClusterResult` | `DTWDistanceComputer` (read) |
| `OverlayPlotter` | Visualize normalized trajectories aligned at equivalent progress points | `list[NormalizedTrajectory]` + optional cluster labels | PNG/SVG | `NarrativeNormalizer`, `ShapeClusterer` |

## Key Design Decision: Post-Pipeline Layer, Not Pipeline Extension

**Decision:** Normalization and DTW clustering live *outside* the Phase 1→2→3 pipeline as a separate post-processing layer.

**Why:**

1. **The existing pipeline produces `AnalysisResult`** — a clean, validated Pydantic model with `timeseries` (A_to_B, B_to_A) and `chunks` with `position` (0.0-1.0). This is the natural boundary.
2. **Normalization is a cross-narrative concern** — it only makes sense when you have *multiple* `AnalysisResult` objects to compare. A single narrative doesn't need normalization.
3. **The existing `SceneClusterer` and `TrajectoryClusterer` remain untouched** — they operate on single-narrative scene features or per-novel statistical summaries. The new `ShapeClusterer` is a fundamentally different operation (shape-based cross-narrative clustering).
4. **Deterministic Phase 3 stays clean** — no new logic leaks into `Synthesizer`.

## Data Flow

### Single-Narrative Flow (existing, unchanged)
```
Text → PCMFGAnalyzer.analyze() → AnalysisResult
```

### Multi-Narrative Normalization + Clustering Flow (new)
```
results: list[AnalysisResult]  (from analyzing multiple narratives)
    ↓
NarrativeNormalizer.normalize_all(results, n_points=100)
    ↓
normalized: list[NormalizedTrajectory]
    ↓
DTWDistanceComputer.compute_distance_matrix(normalized)
    ↓
distance_matrix: NDArray  (NxN symmetric matrix)
    ↓
ShapeClusterer.cluster(distance_matrix, results, n_clusters=3)
    ↓
ShapeClusterResult  (labels, cluster members, silhouette)
    ↓
OverlayPlotter.plot_overlay(normalized, labels, output_path)
```

### Detailed Data Types

```python
# NEW: Normalized trajectory — the core intermediate data structure
class NormalizedTrajectory(BaseModel):
    """A single narrative's emotion time-series resampled to a uniform grid."""
    source: str                          # Source identifier
    main_pairing: list[str]              # Character names
    direction: str                       # "A_to_B" or "B_to_A"
    emotion: str                         # "Joy", "Trust", etc.
    x: list[float]                       # Uniform grid [0.0, ..., 1.0] (n_points)
    y: list[float]                       # Interpolated emotion values
    original_length: int                 # Original number of data points
    n_points: int                        # Normalized grid size

# NEW: Shape cluster result
class ShapeClusterResult(BaseModel):
    """Result of shape-based trajectory clustering."""
    n_clusters: int
    labels: list[int]                    # Cluster assignment per narrative
    distance_matrix: list[list[float]]   # Pairwise DTW distances
    silhouette_score: float | None
    cluster_members: dict[int, list[str]]  # cluster_id → source names
    cluster_interpretations: dict[int, str]  # cluster_id → description
```

## Where Each Component Lives

### `pcmfg/analysis/normalizer.py` — NEW MODULE

**Responsibility:** Resample variable-length emotion time-series to a uniform N-point grid.

**Key functions:**
```python
class NarrativeNormalizer:
    def __init__(self, n_points: int = 100, method: str = "linear") -> None:
        """
        Args:
            n_points: Number of uniformly-spaced points on [0.0, 1.0].
            method: Interpolation method ("linear", "cubic", "pchip").
        """
        ...

    def normalize(self, result: AnalysisResult) -> list[NormalizedTrajectory]:
        """Normalize a single AnalysisResult into per-emotion, per-direction trajectories."""
        ...

    def normalize_all(
        self, results: list[AnalysisResult]
    ) -> list[NormalizedTrajectory]:
        """Normalize multiple AnalysisResults into a flat list of trajectories."""
        ...

    def normalize_emotion(
        self,
        values: list[float],
        source: str,
        main_pairing: list[str],
        direction: str,
        emotion: str,
    ) -> NormalizedTrajectory:
        """Normalize a single emotion series using scipy.interpolate."""
        ...
```

**Implementation approach:** Use `scipy.interpolate.interp1d` with `np.linspace(0, 1, n_points)` as the target grid. The existing `position` field per chunk (0.0-1.0) provides the x-coordinates. The emotion score is the y-coordinate.

**Why `scipy.interpolate` over `tslearn.TimeSeriesResampler`:**
- tslearn's `TimeSeriesResampler` requires tslearn's internal array format (`(n_ts, sz, d)`). Converting to/from that format adds unnecessary coupling.
- `scipy.interpolate.interp1d` is already a transitive dependency (through scikit-learn and numpy) and provides full control over interpolation method.
- The PCMFG data model uses per-chunk positions (potentially non-uniform), not fixed-length arrays. `interp1d` handles non-uniform x-values natively.

**Confidence:** HIGH — scipy interpolation is the standard approach for resampling irregularly-spaced time-series to a uniform grid.

### `pcmfg/analysis/dtw_distance.py` — NEW MODULE

**Responsibility:** Compute pairwise DTW distances between normalized trajectories.

**Key functions:**
```python
class DTWDistanceComputer:
    def __init__(
        self,
        metric: str = "dtw",
        global_constraint: str = "sakoe_chiba",
        sakoe_chiba_radius: int | None = None,
    ) -> None:
        """
        Args:
            metric: Distance metric ("dtw", "euclidean", "soft_dtw").
            global_constraint: DTW constraint type ("itakura", "sakoe_chiba", or None).
            sakoe_chiba_radius: Band width for Sakoe-Chiba constraint.
        """
        ...

    def compute_pairwise(
        self, trajectories: list[NormalizedTrajectory]
    ) -> NDArray[np.float64]:
        """Compute NxN distance matrix from trajectories."""
        ...

    def compute_distance_matrix(
        self,
        results: list[AnalysisResult],
        normalizer: NarrativeNormalizer,
        emotions: list[str] | None = None,
        direction: str = "A_to_B",
    ) -> NDArray[np.float64]:
        """End-to-end: normalize + compute distance matrix."""
        ...
```

**Library choice: `tslearn` over `dtaidistance`:**

| Criterion | tslearn | dtaidistance |
|-----------|---------|--------------|
| scikit-learn integration | Native (extends sklearn estimators) | Separate API |
| TimeSeriesKMeans with DTW | Built-in | Separate KMeans implementation |
| DBSCAN with custom metric | Built-in (`TimeSeriesDBSCAN`) | Not provided |
| Variable-length support | Native (padded arrays) | Native |
| Silhouette score with DTW | Built-in | Not provided |
| Dependencies | numpy, scipy, scikit-learn (all existing) | numpy, Cython (new) |
| Barycenter computation (DBA) | Built-in | Built-in |

**Recommendation: Use `tslearn`** for DTW computation and clustering. It integrates natively with the existing scikit-learn dependency, provides `TimeSeriesKMeans` and `TimeSeriesDBSCAN` that accept custom distance metrics, and handles variable-length time-series natively. The `dtaidistance` library is faster for raw DTW distance computation (C implementation), but its clustering API is separate from scikit-learn and doesn't provide silhouette scoring.

**Confidence:** HIGH — tslearn is the de facto standard for time-series ML in Python, builds on scikit-learn, and is actively maintained.

### `pcmfg/analysis/dtw_clusterer.py` — NEW MODULE

**Responsibility:** Cluster normalized trajectories by shape using DTW-based distance.

**Key functions:**
```python
class ShapeClusterer:
    def __init__(
        self,
        n_clusters: int = 3,
        algorithm: str = "hierarchical",  # "kmeans_dtw", "hierarchical", "dbscan"
        metric: str = "dtw",
        random_state: int = 42,
    ) -> None:
        ...

    def cluster(
        self,
        distance_matrix: NDArray[np.float64],
        sources: list[str],
    ) -> ShapeClusterResult:
        """Cluster using pre-computed distance matrix."""
        ...

    def cluster_from_results(
        self,
        results: list[AnalysisResult],
        normalizer: NarrativeNormalizer,
        distance_computer: DTWDistanceComputer | None = None,
    ) -> ShapeClusterResult:
        """End-to-end: normalize → distance matrix → cluster."""
        ...

    def find_optimal_k(
        self,
        distance_matrix: NDArray[np.float64],
        k_range: tuple[int, int] = (2, 8),
    ) -> dict[int, float]:
        """Find optimal k using silhouette analysis with DTW distance."""
        ...
```

**Relationship to existing `TrajectoryClusterer`:**

The existing `TrajectoryClusterer` in `clusterer.py` reduces each novel to a single row of **statistical features** (mean, std, min, max, range per emotion) — this is a 90-dimensional vector (9 emotions × 5 stats × 2 directions) that loses all shape/temporal information. It then clusters these feature vectors using standard sklearn algorithms with Euclidean distance.

The new `ShapeClusterer` is fundamentally different:
- It operates on **full time-series shapes**, not statistical summaries
- It uses **DTW distance** (or derivative DTW), not Euclidean distance
- It preserves the temporal evolution of emotions
- It clusters **narratives** (not scenes within a narrative)

**They serve complementary purposes and should coexist.** The existing `TrajectoryClusterer` answers "do these novels have similar overall emotional intensity profiles?" while the new `ShapeClusterer` answers "do these novels have similar emotional arc shapes?" Both are valid questions.

**Confidence:** HIGH — architectural separation is clear and both approaches serve different analytical questions.

### `pcmfg/visualization/overlay.py` — NEW MODULE

**Responsibility:** Visualize multiple normalized trajectories overlaid on the same [0.0, 1.0] progress axis.

**Key functions:**
```python
class OverlayPlotter:
    def plot_overlay(
        self,
        trajectories: list[NormalizedTrajectory],
        output_path: str | Path,
        title: str = "Normalized Trajectory Overlay",
        cluster_labels: list[int] | None = None,
    ) -> None:
        """Plot multiple normalized trajectories on same axes, grouped by emotion."""
        ...

    def plot_cluster_comparison(
        self,
        trajectories: list[NormalizedTrajectory],
        labels: list[int],
        output_path: str | Path,
    ) -> None:
        """Plot trajectories colored/ grouped by cluster assignment."""
        ...

    def plot_dendrogram(
        self,
        distance_matrix: NDArray[np.float64],
        sources: list[str],
        output_path: str | Path,
    ) -> None:
        """Plot hierarchical clustering dendrogram from DTW distance matrix."""
        ...
```

**Confidence:** HIGH — matplotlib already handles this well. No new dependencies needed.

## Integration Points with Existing Code

### What changes in existing files:

| File | Change | Why |
|------|--------|-----|
| `pcmfg/analysis/__init__.py` | Add exports for new modules | Public API surface |
| `pcmfg/models/schemas.py` | Add `NormalizedTrajectory`, `ShapeClusterResult` | Data validation for new outputs |
| `pcmfg/config.py` | Add `NormalizationConfig` section | Configuration for n_points, interpolation method, DTW params |
| `pyproject.toml` | Add `tslearn>=0.6.0` dependency | DTW computation and clustering |

### What does NOT change:

| File | Reason |
|------|--------|
| `pcmfg/analyzer.py` | Pipeline orchestrator is unchanged; new modules consume its output |
| `pcmfg/phase3/synthesizer.py` | Synthesis produces `AnalysisResult`; normalization is downstream |
| `pcmfg/analysis/clusterer.py` | Existing `SceneClusterer` and `TrajectoryClusterer` remain for their use cases |
| `pcmfg/analysis/feature_extractor.py` | Feature extraction for scene-level analysis remains unchanged |
| `pcmfg/visualization/plotter.py` | Existing plot methods remain; overlay plotter is additive |
| `pcmfg/phase1/`, `pcmfg/phase2/` | LLM pipeline untouched |

## New Dependency: `tslearn`

```toml
# pyproject.toml addition
dependencies = [
    # ... existing deps ...
    "tslearn>=0.6.0",
]
```

`tslearn` brings in: `numpy`, `scipy`, `scikit-learn` (all already present). No net new transitive dependencies.

## Patterns to Follow

### Pattern 1: Strategy Selection via Config

**What:** Allow switching between Euclidean and DTW distance via configuration.

**When:** The distance metric should be configurable per clustering run.

```python
class NormalizationConfig(BaseModel):
    n_points: int = Field(default=100, ge=10, description="Uniform grid resolution")
    interpolation: Literal["linear", "cubic", "pchip"] = "linear"
    emotions: list[str] | None = Field(
        default=None,
        description="Emotions to include in DTW comparison. None = all 9."
    )
    direction: Literal["A_to_B", "B_to_A", "both"] = "A_to_B"

class DTWConfig(BaseModel):
    metric: Literal["dtw", "soft_dtw", "euclidean"] = "dtw"
    global_constraint: Literal["sakoe_chiba", "itakura", "none"] = "sakoe_chiba"
    sakoe_chiba_radius: int | None = None
    n_clusters: int = Field(default=3, ge=2)
    algorithm: Literal["hierarchical", "kmeans_dtw", "dbscan"] = "hierarchical"
```

### Pattern 2: End-to-End Convenience Method

**What:** Provide a single method that chains normalize → distance → cluster.

**When:** Users want one-call clustering of multiple narratives.

```python
class ShapeClusterer:
    def cluster_from_results(
        self,
        results: list[AnalysisResult],
        normalizer: NarrativeNormalizer | None = None,
        distance_computer: DTWDistanceComputer | None = None,
    ) -> ShapeClusterResult:
        """One-shot: normalize multiple results and cluster by shape."""
        normalizer = normalizer or NarrativeNormalizer()
        distance_computer = distance_computer or DTWDistanceComputer()

        # Normalize
        trajectories = normalizer.normalize_all(results)

        # Compute distance matrix
        # Group by (source, direction, emotion) → one distance matrix per grouping
        # For simplicity, default: use A_to_B direction, all 9 emotions concatenated
        distance_matrix = distance_computer.compute_pairwise(trajectories)

        # Cluster
        sources = [t.source for t in trajectories]
        return self.cluster(distance_matrix, sources)
```

### Pattern 3: Composable Distance Metrics

**What:** The DTW distance computer should accept both single-emotion trajectories and multi-dimensional (all 9 emotions) trajectories.

**When:** Users may want to cluster by a single dominant emotion (e.g., "Trust arc shape") or by the full 9-emotion profile.

```python
def compute_pairwise(
    self,
    trajectories: list[NormalizedTrajectory],
    multi_dim: bool = False,
) -> NDArray[np.float64]:
    """
    If multi_dim=True, group trajectories by source and compute
    multidimensional DTW across all 9 emotions simultaneously.
    If multi_dim=False, compute per-emotion and average.
    """
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Modifying `AnalysisResult` to include normalized data
**What:** Adding normalized time-series fields to the existing `AnalysisResult` model.
**Why bad:** Violates single responsibility. `AnalysisResult` is the output of the extraction pipeline. Normalization is a cross-narrative operation. Mixing concerns makes the model bloated and couples pipeline output to analysis configuration.
**Instead:** Return `NormalizedTrajectory` as a separate model from `NarrativeNormalizer`.

### Anti-Pattern 2: Building DTW from scratch
**What:** Implementing DTW algorithm manually with nested loops.
**Why bad:** O(n²) naive implementation is slow. Production DTW needs Sakoe-Chiba banding, derivative DTW, and fast C implementations.
**Instead:** Use `tslearn.metrics.dtw` which provides optimized implementations with constraints.

### Anti-Pattern 3: Replacing existing `TrajectoryClusterer`
**What:** Removing the statistical-feature-based trajectory clustering.
**Why bad:** Statistical features answer a different question than shape-based clustering. Both have valid use cases. Removing breaks backward compatibility.
**Instead:** Add `ShapeClusterer` alongside existing clusterers. Let users choose.

### Anti-Pattern 4: Normalizing inside `Synthesizer`
**What:** Adding normalization as a step in Phase 3 synthesis.
**Why bad:** Normalization requires knowing the target grid size, which is a cross-narrative concern. The synthesizer operates on a single narrative and shouldn't know about other narratives.
**Instead:** Normalization is a separate post-processing step that consumes `AnalysisResult` output.

## Scalability Considerations

| Concern | At 10 narratives | At 100 narratives | At 1000 narratives |
|---------|-------------------|-------------------|---------------------|
| DTW distance matrix (O(n² × m²)) | Instant (<1s) | ~2-5s | ~3-10 min |
| Memory for distance matrix | Negligible | 100×100×8 bytes = 80KB | 1000×1000×8 bytes = 8MB |
| Hierarchical clustering | Instant | Instant | ~1s |
| KMeans with DTW barycenter | Instant | ~5s | ~5-10 min |
| Normalization (interpolation) | Instant | Instant | ~1s |

**Key bottleneck:** DTW pairwise distance is O(n² × m²) where n = number of narratives and m = normalized grid size. With `n_points=100` and 100 narratives, this is 100² × 100² = 10⁸ operations — manageable. With 1000 narratives, it becomes 10¹² — may need approximate DTW (FastDTW) or lower `n_points`.

**Mitigation strategies for large scale:**
1. Use Sakoe-Chiba banding (default) to reduce DTW to O(n² × m × w) where w = band radius
2. Reduce `n_points` for exploratory analysis (50 instead of 100)
3. Use `tslearn`'s built-in parallelism for distance matrix computation
4. Cache normalized trajectories to avoid re-interpolation

## Suggested Build Order (Dependencies Between Components)

```
Phase A: Data Models + Normalizer
  └── Add NormalizedTrajectory, ShapeClusterResult to schemas.py
  └── Implement NarrativeNormalizer with scipy.interpolate
  └── Add NormalizationConfig to config.py
  └── Tests: normalize single narrative, verify grid uniformity

Phase B: DTW Distance Layer
  └── Add tslearn dependency
  └── Implement DTWDistanceComputer
  └── Add DTWConfig to config.py
  └── Tests: pairwise distance, verify symmetry, compare to known DTW values

Phase C: Shape Clustering
  └── Implement ShapeClusterer using tslearn clustering
  └── Tests: cluster known-similar trajectories together, silhouette scores

Phase D: Overlay Visualization
  └── Implement OverlayPlotter
  └── Tests: overlay 2+ normalized trajectories, verify axis alignment

Phase E: Integration + CLI
  └── Wire into cli.py (new subcommands)
  └── End-to-end test: analyze 3 texts → normalize → cluster → visualize
  └── Update __init__.py exports
```

**Build order rationale:**
- A must come first — all other components consume `NormalizedTrajectory`
- B depends on A (needs normalized trajectories to compute distances)
- C depends on B (needs distance matrix)
- D depends on A (needs normalized trajectories for overlay), but can run parallel to B/C
- E depends on everything

## Sources

- tslearn documentation: https://tslearn.readthedocs.io/en/stable/ (HIGH confidence — official docs)
- tslearn preprocessing (TimeSeriesResampler): https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.preprocessing.html (HIGH confidence)
- tslearn clustering (TimeSeriesKMeans, TimeSeriesDBSCAN): https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.clustering.html (HIGH confidence)
- dtaidistance documentation: https://dtaidistance.readthedocs.io/en/latest/ (HIGH confidence — official docs)
- Existing PCMFG codebase analysis (HIGH confidence — directly read source files)
