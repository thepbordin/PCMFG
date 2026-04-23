# Phase 3: Overlay Visualization - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can visually compare emotional trajectories from multiple narratives aligned on the same normalized timeline, with cluster coloring and barycenter overlays. This is a post-clustering visualization layer — it consumes `NormalizedTrajectory` (from Phase 1) and `DTWClusterResult` (from Phase 2) and produces PNG overlay plots.

Deliverables: overlay plotter class with methods for 9x2 emotion grid, per-emotion, per-direction, and per-cluster views. Full suite output generates all variants from a single call.

</domain>

<decisions>
## Implementation Decisions

### Plot Layout Strategy
- **D-01:** Primary layout is 9 rows x 2 columns (emotions x directions) — same grid structure as existing `EmotionPlotter.plot_timeseries()`. Each subplot overlays all narrative lines for one emotion+direction combination.
- **D-02:** One emotion per subplot — no grouped or stacked emotion views.
- **D-03:** API is a `NarrativeOverlayPlotter` class with methods: `plot_overlay()` (9x2 grid), `plot_emotion()` (single emotion, both directions), `plot_direction()` (all emotions for one direction), `plot_cluster()` (members + barycenter for one cluster).
- **D-04:** Figure size and DPI match `EmotionPlotter`: `figsize=(16, 20)`, `dpi=300`, `tight_layout`.
- **D-05:** Plots save to file (like `EmotionPlotter`), not return `Figure` objects.
- **D-06:** New file `pcmfg/visualization/overlay_plotter.py`. The overlay is about visualizing normalized trajectories, fitting the visualization package.

### Cluster Visual Encoding
- **D-07:** When cluster assignments exist, narrative lines are colored by cluster using the existing `CLUSTER_COLORS` palette from `analysis/plotter.py`. All members of the same cluster share a color.
- **D-08:** Member narrative lines use low alpha (0.3-0.4) so the overall shape pattern emerges from overlap. Barycenters get full alpha (1.0) with bold lines — clear visual hierarchy.
- **D-09:** Legends show cluster labels only (e.g., "Cluster 0 (5 narratives)", "Cluster 1 (3 narratives)"). Individual narrative names are NOT in the legend — keeps plots clean with many narratives.
- **D-10:** When no `DTWClusterResult` is provided, use a sequential color cycle (e.g., `tab10`) with each narrative getting a unique color. No cluster-specific styling.

### Barycenter Overlay Style
- **D-11:** Barycenter line is drawn on the same axes as its cluster members. Solid line, linewidth=3-4, with large diamond markers at regular intervals. Members are thin (linewidth=1) and translucent behind it.
- **D-12:** Barycenters are always shown when `DTWClusterResult` is provided. No opt-in flag needed — it's the primary value of clustering visualization.
- **D-13:** Only the joint 18-dimensional barycenter from `DTWClusterResult.barycenters` is visualized. No additional per-emotion mean computation.

### Output Granularity
- **D-14:** A single "generate all" method produces the full suite: 9x2 grid PNG, 9 per-emotion PNGs (each with A-to-B and B-to-A subplots), 2 per-direction PNGs (one per direction with all 9 emotions), and per-cluster views.
- **D-15:** Output file naming uses `overlay_` prefix: `overlay_grid.png`, `overlay_joy.png`, `overlay_trust.png`, `overlay_atob.png`, `overlay_btoa.png`, `overlay_cluster_0.png`, etc.
- **D-16:** PNG only — no PDF, CSV, or metadata JSON output. Matches existing plotter conventions.
- **D-17:** Only overlay visualization is in scope. DTW warping path visualization (VIS-05) and per-emotion distance matrices (VIS-06) are v2 requirements and out of scope.

### Claude's Discretion
- Exact alpha value for member lines (0.3-0.4 range recommended)
- Barycenter marker frequency (every N points)
- Whether per-cluster views are 9x2 grids or a condensed layout
- Method parameter naming and API surface details
- How to handle edge cases (single narrative, single cluster, empty clusters)
- Whether to add a convenience method that takes `list[AnalysisResult]` directly (normalize + cluster + overlay in one call)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `EmotionPlotter` (`visualization/plotter.py`, 653 lines): Established 9x2 grid pattern, `EMOTION_CONFIG` color map (9 emotion colors), baseline reference line, fill_between styling, DPI/figsize conventions
- `CLUSTER_COLORS` palette (`analysis/plotter.py`): 10-color palette for cluster visualization, plus `NOISE_COLOR`
- `NormalizedTrajectory` model (`schemas.py:266`): Per-emotion, per-direction resampled trajectory with `source`, `main_pairing`, `direction`, `emotion`, `x`, `y`, `original_length`, `n_points`
- `DTWClusterResult` model (`dtw_clusterer.py:45`): `assignments` (source→cluster), `barycenters` (list of shape `(n_points, 18)` arrays), `sources` (ordered list), `distance_matrix`, `n_clusters`, `metric`, `cluster_sizes`
- `BASE_EMOTIONS` constant (`schemas.py:32`): List of 9 emotion names for iteration
- `build_dtw_dataset()` function (`analysis/dtw_clusterer.py`): Stacks `NormalizedTrajectory` list into `(n_narratives, n_points, 18)` array — reusable for organizing overlay data

### Established Patterns
- Class-based plotters: `EmotionPlotter(dpi=300, figsize=(16, 20))` with methods that save to file
- Inline styling (no global rcParams or plt.style.use())
- `ax.fill_between()` for area under curves, `ax.axhline(y=1)` for baseline
- Y-axis locked to [0.5, 5.5] with integer ticks [1,2,3,4,5]
- X-axis always `np.linspace(0, 1, len(values))` — narrative progress
- `plt.tight_layout(rect=[0, 0, 1, 0.96])` for suptitle space
- `fig.savefig(path, dpi=self.dpi, bbox_inches="tight")` + `plt.close(fig)`
- `output_path.parent.mkdir(parents=True, exist_ok=True)` for auto-creating directories
- Two separate plotter modules with different conventions (visualization saves to file, analysis returns Figure)

### Integration Points
- Input: `list[NormalizedTrajectory]` from `NarrativeNormalizer.normalize_all()`
- Optional input: `DTWClusterResult` from `DTWClusterer.cluster()` for cluster coloring + barycenters
- `pcmfg/visualization/__init__.py` needs new export for `NarrativeOverlayPlotter`
- The 18-dimensional barycenter needs to be unpacked into per-emotion, per-direction arrays for plotting — planner figures out the reshaping logic

</code_context>

<specifics>
## Specific Ideas

- The barycenter shape `(n_points, 18)` maps to the Phase 2 feature ordering: for each of the 9 `BASE_EMOTIONS`, first A_to_B then B_to_A (i.e., `[Joy_A2B, Joy_B2A, Trust_A2B, Trust_B2A, ...]`). The overlay plotter needs to slice this correctly.
- Full suite output means a convenience method like `plot_all()` that generates grid + per-emotion + per-direction + per-cluster views into a specified output directory.

</specifics>

<deferred>
## Deferred Ideas

- **VIS-05**: DTW warping path visualization — showing which narrative moments align between two stories (v2 requirement, already deferred from Phase 2)
- **VIS-06**: Per-emotion distance matrix heatmap revealing which emotions drive inter-narrative similarity (v2 requirement)
- **CLST-06**: Automatic emotion-based cluster labeling (e.g., "Trust-building arcs") (v2 requirement, already deferred from Phase 2)
- **Raw mode flag**: Option to skip forward-fill imputation before normalization (deferred from Phase 2)
- **PDF/vector output**: For publication-quality figures (future enhancement)
- **Interactive/hover tooltips**: For narrative identification in crowded overlays (out of scope — CLI tool only)
- **Convenience end-to-end method**: normalize + cluster + overlay in one call (planner may evaluate)

</deferred>

---

*Phase: 03-overlay-visualization*
*Context gathered: 2026-04-24*
