# Pitfalls Research

**Domain:** Narrative time-series normalization and shape-based DTW clustering
**Researched:** 2026-04-23
**Confidence:** MEDIUM-HIGH

## Critical Pitfalls

### Pitfall 1: Linear Interpolation Creates Fake Emotion Intensities on an Ordinal Scale

**What goes wrong:**
When resampling emotion time-series to a uniform grid (e.g., [0.0, 1.0] with N points), standard linear interpolation produces fractional values like 2.3 or 3.7. The PCMFG system uses a **strict 1-5 ordinal integer scale** — there is no "2.3 Joy" in the scoring rubric. These interpolated values are mathematically convenient but semantically meaningless. A DTW distance computed over interpolated ordinal data treats a difference of 0.3 the same as a difference of 0.3 in continuous data, which is incorrect.

**Why it happens:**
The `position` field exists per chunk (0.0-1.0), and `np.linspace` is already used in the plotter (line 146 of `plotter.py`) to map chunks to progress points. The natural next step is to use `scipy.interpolate.interp1d(kind='linear')` or tslearn's `TimeSeriesResampler` (which uses linear interpolation internally) to create a uniform grid. But the 1-5 scale is ordinal, not interval — the distance between 1→2 is not necessarily the same as 3→4 in emotional magnitude.

**How to avoid:**
1. **Nearest-neighbor resampling** instead of linear interpolation. Use `scipy.interpolate.interp1d(kind='nearest')` to snap to actual scored values. This preserves ordinal integrity.
2. **Weighted-slope interpolation** with clamping: interpolate but round to nearest integer before computing DTW. This acknowledges some continuity in emotion space while staying grounded in the scoring rubric.
3. **Document the decision explicitly** in the normalizer module: "We treat the 1-5 scale as interval data for the purpose of cross-narrative comparison, accepting that this is a simplification."
4. Consider whether the DTW metric itself should use a **custom ground metric** (tslearn's `dtw_path_from_metric`) that respects ordinal distance — e.g., |a-b|^2 weighted by a step function.

**Warning signs:**
- Normalized time-series contain non-integer values (2.3, 1.7, etc.) in emotion columns
- DTW distances are very small (because interpolation smooths away the discrete jumps that make narratives distinct)
- Clustering produces one giant blob instead of meaningful groups (everything looks similar after smoothing)

**Phase to address:**
Normalization module implementation (first phase building the normalizer)

---

### Pitfall 2: Forward Fill Creates Illusory Emotional Stability

**What goes wrong:**
The existing `impute_missing_emotions()` in `synthesizer.py` uses forward fill: when character B is absent from a scene, B→A emotions are carried forward from the last known state. This is correct for **within-narrative** continuity (B's emotions persist while off-screen). But when two narratives of very different lengths are compared via DTW, a long forward-filled flat segment in one narrative will be warped against actual emotional variation in the other. The DTW alignment will treat the flat segment as "real" emotional data, producing misleading similarity scores.

More critically: a 50-chapter novel with character B absent for 20 chapters creates a 20-point flat plateau at whatever score was last observed. A 10-chapter novella with B present in every scene has no such plateaus. When DTW aligns these, it may match the plateau to a rising/falling segment in the novella, declaring them "similar" because the warping hides the plateau.

**Why it happens:**
Forward fill was designed for within-narrative analysis where the goal is "what is B feeling right now?" It wasn't designed for cross-narrative comparison where the question is "do these two emotional arcs have similar shapes?" The flat segments are artifacts of the data collection process (one character being off-screen), not features of the emotional arc.

**How to avoid:**
1. **Mark forward-filled data** with a flag. The `justification_quote` already contains `"[FORWARD FILLED - character absent from scene]"` — use this metadata to distinguish real from imputed data.
2. **Provide a "raw" mode** for cross-narrative comparison that excludes forward-filled segments or replaces them with NaN. DTW implementations can handle NaN by skipping those points (or the time-series can be truncated at gaps).
3. **Cap consecutive forward-fill runs** — if more than N consecutive chunks are forward-filled, treat the gap as "unknown" rather than carrying forward. A reasonable N for narrative analysis is 3-5 chunks (the character is briefly off-screen, not gone for an arc).
4. **Compute DTW only on "observed" segments** — split each trajectory at forward-fill boundaries and compute DTW on the segments that have real data from both narratives.

**Warning signs:**
- DTW clustering groups a novel where one character is mostly absent with novels that have genuine low-intensity emotional arcs
- Normalized trajectories have suspiciously long flat segments at non-baseline values (e.g., Trust=4.0 for 15 consecutive points)
- Overlay plots show one narrative's plateau perfectly aligned with another's actual emotional variation

**Phase to address:**
Normalization module (when designing the normalization pipeline that feeds into DTW)

---

### Pitfall 3: DTW Without Constraints Produces Nonsensical Narrative Alignments

**What goes wrong:**
Unconstrained DTW allows any temporal warping — a single emotional peak in narrative A could be matched against 50 flat points in narrative B, and vice versa. For narrative analysis, this is semantically wrong. A climactic confrontation scene at position 0.8 in one novel should not be warped to match a slow-building tension arc spanning positions 0.3-0.9 in another. The narrative structure has inherent temporal meaning: "the ending" should align with "the ending," not with "the middle."

Without constraints, DTW will find mathematically optimal but narratively meaningless alignments. The Sakoe-Chiba band or Itakura parallelogram constraints exist specifically to prevent this.

**Why it happens:**
DTW was originally designed for speech recognition where temporal stretching is expected (people speak at different speeds). Narrative progress is different — the `position` field (0.0-1.0) already normalizes for length, so the main remaining variation is in how many "events" happen at each progress point. The permissible warping should be small: a climax at 0.75 in one narrative should align within roughly 0.65-0.85 in another, not at 0.2.

**How to avoid:**
1. **Use Sakoe-Chiba band by default** with a radius proportional to narrative length. For emotion trajectories with ~50 points, a radius of 5-10 is reasonable (allowing ±10-20% temporal shift). tslearn supports this directly: `dtw(x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=r)`.
2. **Make the radius configurable** and expose it as a hyperparameter in the clusterer. Different narrative types may need different warping tolerances.
3. **Visualize the warping path** for at least one pair in each cluster. If the alignment path looks like a zigzag rather than a near-diagonal line, the constraint is too loose.
4. **Consider Amerced DTW (ADTW)** as an alternative — it penalizes each warping action rather than using a hard window, which may be more appropriate for narratives where some temporal shift is expected but extreme warping should be penalized rather than forbidden. (Reference: Herrmann & Webb, 2023)

**Warning signs:**
- DTW path visualization shows extreme diagonal excursions (matching beginning of one narrative to end of another)
- Clustering results don't match intuitive groupings (e.g., "enemies to lovers" novels not clustering together)
- Silhouette scores are suspiciously high (because unconstrained DTW can always find a "good" alignment)

**Phase to address:**
DTW clustering implementation phase (when choosing DTW parameters)

---

### Pitfall 4: Treating All 9 Emotions as Independent Dimensions in DTW

**What goes wrong:**
DTW operates on multivariate time-series. If all 9 emotions are concatenated into a single 9-dimensional vector per time step, DTW computes a joint alignment that warps all emotions simultaneously along the same temporal path. But emotions in a narrative don't always change at the same time — Joy may spike at position 0.7 while Trust doesn't rise until 0.8. A joint DTW alignment forces a single warping that must compromise across all emotions, potentially misaligning each individual emotion's key events.

Additionally, the 9 emotions have very different dynamic ranges in practice. Most narratives spend most of their time at baseline (1.0) for most emotions, with occasional spikes. This creates highly sparse vectors where DTW's Euclidean ground metric is dominated by the few emotions that happen to be active, drowning out the meaningful shape information.

**Why it happens:**
The existing `EmotionTimeSeries` model has 9 parallel arrays. The natural approach is to stack them into a (N, 9) matrix and pass to DTW. tslearn expects exactly this format. But the emotional dynamics are not synchronized, and the variance structure is highly uneven across emotions.

**How to avoid:**
1. **Per-emotion DTW with aggregation**: Compute DTW separately for each emotion, then aggregate distances (e.g., weighted average). This allows each emotion to find its own optimal temporal alignment. More expensive (9x DTW calls) but semantically superior.
2. **Standardize per-emotion before joint DTW**: z-normalize each emotion independently (subtract mean, divide by std) before concatenating. This equalizes the influence of high-variance emotions (like Arousal) and low-variance ones (like Surprise).
3. **Weighted DTW**: Use a custom ground metric that assigns higher weight to emotions more relevant to the specific analysis. For romance narratives, Trust, Joy, Arousal, and Anger might get weight 2x while Surprise gets 0.5x.
4. **Dimensionality reduction first**: Apply PCA to the 9-emotion space before DTW. This captures the dominant emotional axes while reducing noise.

**Warning signs:**
- Changing which emotions are included dramatically changes clustering results
- Clustering seems driven by one or two "loud" emotions rather than overall shape
- DTW distances are dominated by a single emotion's contribution

**Phase to address:**
DTW clustering implementation phase (when designing the distance computation strategy)

---

### Pitfall 5: The "Same Shape, Different Baseline" Problem

**What goes wrong:**
Two narratives may have identical emotional *shapes* (same ups and downs) but at completely different absolute levels. Narrative A: Joy goes 1→3→5→2→1. Narrative B: Joy goes 1→2→3→1→1. These have the same shape (single peak) but different amplitudes. DTW with Euclidean ground metric will report a significant distance because the absolute values differ, even though the *pattern* is the same.

For narrative analysis, the shape often matters more than the absolute intensity. A "slow burn" romance where emotions peak at 3 and a "passionate" romance where emotions peak at 5 may have identical narrative structures, just at different intensities.

**Why it happens:**
DTW's ground metric is Euclidean distance, which is sensitive to both shape and offset. There's no built-in mechanism to say "ignore absolute levels, compare only shape." The existing `StandardScaler` in `clusterer.py` is applied per-feature across samples, not per-time-series, so it doesn't address this.

**How to avoid:**
1. **Per-time-series z-normalization**: For each narrative's emotion trajectory, subtract the mean and divide by std before DTW. This removes baseline and scale differences, comparing only shape. tslearn supports this via `tslearn.preprocessing.TimeSeriesScalerMeanVariance`.
2. **Derivative DTW (DDTW)**: Compute DTW on the first derivative of the time-series rather than the raw values. Derivatives capture the *direction and rate of change*, completely ignoring absolute levels. A Joy score going 1→3 has the same derivative as 4→5 (both +2).
3. **Provide both options**: Some analyses care about intensity (e.g., "which novels have the most passionate arcs?") and some care only about shape (e.g., "which novels follow the enemies-to-lovers pattern?"). The normalizer should support both modes.

**Warning signs:**
- Clustering groups novels by overall emotional intensity rather than narrative pattern
- High-arousal novels form their own cluster even when their narrative structures differ
- Low-intensity "quiet" novels cluster together despite having completely different arcs

**Phase to address:**
Normalization module (when designing normalization strategies) and clustering module (when exposing distance options)

---

### Pitfall 6: DTW Clustering with k-means Produces Unstable Centroids

**What goes wrong:**
Standard k-means computes centroids as the arithmetic mean of cluster members. But the arithmetic mean of DTW-aligned time-series is not meaningful — DTW doesn't define a proper vector space (it violates the triangle inequality). The correct approach is DTW Barycenter Averaging (DBA), which iteratively refines a centroid by aligning all members to it and averaging the aligned values.

The existing `TrajectoryClusterer` in `clusterer.py` (line 324-458) reduces each novel to a single row of statistical features and clusters those. This loses shape information entirely. If the new DTW clusterer uses sklearn's `KMeans` directly on DTW distance matrices, the centroids will be computed incorrectly.

**Why it happens:**
sklearn's `KMeans` expects Euclidean space. It computes centroids as means. There's no way to override this behavior. Using `KMeans(metric='precomputed')` with a DTW distance matrix still computes centroids as means of the distance matrix rows, which is meaningless. The correct tool is tslearn's `TimeSeriesKMeans` with `metric="dtw"`, which uses DBA internally.

**How to avoid:**
1. **Use tslearn's `TimeSeriesKMeans`** instead of sklearn's `KMeans` when DTW is the distance metric. It handles DBA centroid computation correctly.
2. **If using hierarchical clustering** with a precomputed DTW distance matrix, use `AgglomerativeClustering(metric='precomputed', linkage='average')`. Average linkage is more robust than Ward's method for non-Euclidean distances.
3. **Never compute cluster centers manually** from DTW distances. If you need a representative trajectory for a cluster, use DBA or pick the medoid (the time-series with minimum average DTW distance to all other members).
4. **Validate centroids visually**: Plot the computed centroid alongside cluster members. If the centroid looks like a blurred average rather than a representative shape, the centroid computation is wrong.

**Warning signs:**
- Cluster centroids look like flat averages with no distinctive shape
- Adding/removing one time-series dramatically changes the centroid
- Silhouette scores are inconsistent across runs with different `random_state`

**Phase to address:**
DTW clustering implementation phase (when choosing the clustering algorithm)

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Using linear interpolation for normalization | Simpler code, tslearn's `TimeSeriesResampler` works out of the box | Semantically meaningless fractional emotion scores; subtle clustering errors | Only for initial prototyping, must be replaced before production |
| Using sklearn `KMeans` with precomputed DTW matrix | No new dependency (tslearn not needed) | Incorrect centroids; meaningless cluster centers; misleading results | Never — use tslearn's `TimeSeriesKMeans` |
| Skipping Sakoe-Chiba constraints | Simpler DTW calls, one fewer hyperparameter | Nonsensical alignments, over-permissive warping | Only for quick sanity checks on very similar-length series |
| Treating 1-5 scale as continuous for DTW | Avoids clamping/rounding complexity | Ordinal semantics lost; small meaningless differences inflate distances | Document as deliberate simplification; revisit if clustering quality is poor |
| Computing joint 9-emotion DTW | One DTW call per pair instead of 9 | Single alignment compromises all emotions; dominated by high-variance emotions | Acceptable if per-emotion z-normalization is applied first |
| Hardcoding forward-fill behavior for cross-narrative comparison | Reuses existing `impute_missing_emotions()` | Flat plateaus distort DTW; absent-character artifacts masquerade as emotional stability | Never for cross-narrative comparison; only for within-narrative analysis |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| tslearn `TimeSeriesResampler` | Assuming it handles non-uniform input positions | It only handles uniform time-series — you must first map your chunk positions to a uniform grid using `np.interp` or similar |
| tslearn `TimeSeriesKMeans` | Passing raw lists instead of (n_ts, sz, d) arrays | tslearn expects shape `(n_timestamps, n_timesteps, n_dimensions)` — reshape your 9-emotion series into `(n_novels, n_points, 9)` |
| tslearn + scikit-learn version conflicts | tslearn pins older numpy/scikit-learn | Pin `tslearn>=0.6.0` and check compatibility matrix; tslearn 0.6+ supports scikit-learn 1.3+ |
| Existing `ClusterResult` model | Trying to extend it for DTW without modifying the Pydantic schema | The `ClusterResult.cluster_centers` field is `list[list[float]]` — for DTW centroids (which are time-series), this needs to be `list[list[list[float]]]` or a new field |
| Existing `TrajectoryClusterer` | Replacing it entirely and breaking existing scene-level clustering | Extend the module with a new `DTWClusterer` class; keep `SceneClusterer` and `TrajectoryClusterer` untouched |
| Pydantic frozen models | Trying to modify `DirectedEmotionScores` (frozen=True) during normalization | Create new instances rather than mutating; the `model_config = ConfigDict(frozen=True)` on line 76 of schemas.py prevents in-place changes |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| O(N*M) pairwise DTW for clustering | Minutes to compute distance matrix for >50 novels | Use Sakoe-Chiba band (reduces to O(N*band)); use `cdist_dtw` for vectorized computation; cache distance matrices | At ~100 novels with ~100 points each, full pairwise DTW takes ~30 seconds with band, ~5 minutes without |
| No lower-bound pruning | Clustering takes forever on large datasets | Use LB_Keogh or LB_Enhanced lower bounds to prune DTW computations (tslearn doesn't expose these directly, but `tempo` library does) | At >500 novels, even with band constraints |
| Resampling inside a hot loop | Re-interpolating the same novel for every pairwise comparison | Pre-normalize all time-series to the target grid once, cache the result | With >50 novels and >1000 DTW comparisons |
| Redundant DTW path computation | Computing both `dtw()` and `dtw_path()` separately | Use `dtw_path()` which returns both; or cache the distance matrix | When both distances and visualizations are needed |

## "Looks Done But Isn't" Checklist

- [ ] **Normalization to [0.0, 1.0] grid**: Often missing — verify that the resampled grid uses the *actual chunk positions* (not just uniform `np.linspace`), and that short novels aren't over-interpolated into noise. Check that the resampling method handles non-uniform original positions.
- [ ] **Forward-fill flag propagation**: Often missing — verify that when forward-filled data is excluded for DTW, the remaining segments are still long enough for meaningful DTW (minimum ~5 points per segment).
- [ ] **DTW constraint tuning**: Often missing — verify that the Sakoe-Chiba radius was actually tested across a range, not just set to a default. Plot DTW paths for a few pairs to visually confirm reasonable alignment.
- [ ] **Emotion dimension handling**: Often missing — verify whether per-emotion or joint DTW was used, and why. Check that z-normalization is applied per-emotion, not globally.
- [ ] **Centroid validation**: Often missing — verify that cluster centroids (if using k-means) were computed with DBA, not arithmetic mean. Plot centroids alongside members.
- [ ] **Distance matrix symmetry**: Often missing — verify that DTW(A, B) ≈ DTW(B, A). While DTW is theoretically symmetric, numerical precision can cause tiny asymmetries. Large asymmetries indicate a bug.
- [ ] **Cluster stability**: Often missing — verify that running clustering multiple times with different `random_state` produces similar results. DTW k-means with DBA can converge to different local optima.
- [ ] **Overlay comparison correctness**: Often missing — verify that overlay plots actually show data aligned at equivalent narrative positions, not just two lines drawn on the same axes.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Linear interpolation artifacts | LOW | Switch to nearest-neighbor resampling; re-run clustering; compare results |
| Forward-fill plateau distortion | MEDIUM | Add forward-fill detection to normalizer; re-compute time-series with capped forward-fill; re-run DTW |
| Missing Sakoe-Chiba constraint | LOW | Add `global_constraint="sakoe_chiba"` with tuned radius; re-compute distance matrix; re-cluster |
| Wrong centroid computation (sklearn KMeans) | LOW | Replace with tslearn `TimeSeriesKMeans(metric="dtw")`; re-cluster |
| Joint vs per-emotion DTW choice | MEDIUM | Re-compute distances with alternative strategy; compare cluster assignments using adjusted Rand index |
| Ordinal scale violation | MEDIUM | Re-normalize with rounding or nearest-neighbor; re-run full pipeline; document tradeoff decision |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Linear interpolation artifacts | Normalization module | Unit test: resampled values must all be integers 1-5 (if nearest-neighbor) |
| Forward-fill plateau distortion | Normalization module | Unit test: time-series with >5 consecutive forward-filled chunks must be flagged or split |
| Unconstrained DTW | DTW clustering module | Integration test: DTW path for known-similar narratives must stay within ±20% of diagonal |
| Joint 9-emotion DTW dominance | DTW clustering module | Compare per-emotion vs joint DTW cluster assignments; verify per-emotion z-normalization is applied |
| Same-shape-different-baseline | Normalization module | Unit test: two trajectories with same shape but different baselines must have DTW distance < threshold after z-normalization |
| Wrong centroid computation | DTW clustering module | Unit test: `TimeSeriesKMeans` from tslearn must be used, not sklearn `KMeans` |
| tslearn array shape mismatch | DTW clustering module | Unit test: input shape must be (n_novels, n_points, n_emotions) |
| Pydantic frozen model mutation | Any phase touching schemas | Type checker (mypy) will catch; unit test must verify new instances created |

## Sources

- tslearn official documentation: DTW metrics, clustering, and preprocessing — HIGH confidence (official docs, verified 2026-04-23)
- Sakoe & Chiba (1978): Original Sakoe-Chiba band constraint — HIGH confidence (foundational paper)
- Petitjean et al. (2011): DTW Barycenter Averaging (DBA) — HIGH confidence (widely cited)
- Herrmann & Webb (2023): Amerced DTW (ADTW) — MEDIUM confidence (recent paper, verified via tslearn docs)
- Wikipedia: Dynamic Time Warping — MEDIUM confidence (general reference, verified against tslearn docs)
- PCMFG codebase analysis: `synthesizer.py`, `clusterer.py`, `schemas.py`, `plotter.py`, `feature_extractor.py` — HIGH confidence (direct code inspection)
- Ordinal data interpolation best practices — MEDIUM confidence (general statistical knowledge, not verified against a specific narrative analysis paper)

---
*Pitfalls research for: Narrative time-series normalization and shape-based DTW clustering*
*Researched: 2026-04-23*
