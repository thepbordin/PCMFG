# Stack Research

**Domain:** Narrative time-series normalization, DTW-based shape clustering
**Researched:** 2026-04-23
**Confidence:** HIGH

## Recommended Stack

### Core Addition: tslearn (DTW + Clustering)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **tslearn** | >=0.8.1 | DTW computation, time-series clustering, preprocessing, resampling | One-stop solution that provides `TimeSeriesKMeans` (with DTW metric), `dtw()` distance function, `TimeSeriesResampler`, and `TimeSeriesScalerMinMax`. Scikit-learn-compatible API means it integrates directly with the existing `TrajectoryClusterer` infrastructure. Supports variable-length time-series natively — the exact problem PCMFG has (novels of different lengths). 3.1k GitHub stars, active development (v0.8.1 released Mar 2026), BSD-2-Clause license. |
| **scipy** | >=1.15.0 | Interpolation (`scipy.interpolate`), hierarchical linkage | Already a transitive dependency via scikit-learn. Provides `scipy.interpolate.interp1d` for resampling variable-length emotion time-series to a uniform [0.0, 1.0] grid with cubic or linear interpolation. `scipy.cluster.hierarchy` for dendrogram-based clustering. No new dependency needed. |

### Existing Stack (No Changes Needed)

| Technology | Version | Role in This Milestone |
|------------|---------|------------------------|
| numpy | >=1.24.0 | Array operations for time-series data — already the backbone |
| pandas | >=2.0.0 | DataFrames for emotion time-series storage and manipulation |
| scikit-learn | >=1.3.0 | Pipeline infrastructure, silhouette scoring, `GridSearchCV` for hyperparameter tuning of tslearn models |
| matplotlib | >=3.7.0 | Overlay visualization of normalized trajectories |
| pydantic | >=2.0.0 | Schema validation for new normalization/clustering config models |

### New Dependency Summary

Only **one new dependency** is needed:

```python
# pyproject.toml additions
dependencies = [
    # ... existing deps ...
    "tslearn>=0.8.1",  # DTW + time-series clustering
]
```

No new dependency for interpolation — `scipy.interpolate` is already available via the scikit-learn transitive dependency.

## Alternatives Considered

### DTW Libraries

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **tslearn** | dtaidistance | If you need raw DTW speed on very large datasets (millions of time-series) with C-optimized distance matrix computation. dtaidistance 2.4.0 (Feb 2026) has fast C implementations and parallelization. However, it provides only distance computation, not clustering algorithms — you'd need to build your own KMeans wrapper or use scipy linkage. |
| **tslearn** | fastdtw | Never. fastdtw 0.3.4 was last released Oct 2019, builds only for Python 3.7, has no wheels for Python 3.12+, and is effectively abandoned. Its O(N) approximation is also available in tslearn via the `global_constraint` parameter. |
| **tslearn** | scipy (no DTW) | If you want to avoid all new dependencies and use only Euclidean distance. But this defeats the purpose — Euclidean cannot align temporal shifts, which is the core problem this milestone solves. |

### Why tslearn Over dtaidistance

1. **Batteries-included**: tslearn provides `TimeSeriesKMeans`, `KShape`, `KernelKMeans` — ready-made clustering with DTW distance. dtaidistance only provides `dtw.distance_matrix` and basic hierarchical wrappers.
2. **Scikit-learn compatibility**: tslearn models follow `fit`/`predict`/`transform` API and work with `GridSearchCV`, `Pipeline`, `cross_val_score`. This means the existing `TrajectoryClusterer` can be extended rather than rewritten.
3. **Preprocessing built-in**: `TimeSeriesResampler` handles the resampling problem (aligning variable-length sequences to uniform length), and `TimeSeriesScalerMinMax` normalizes amplitude. No need to write custom resampling code.
4. **Variable-length support**: tslearn natively handles datasets where time-series have different lengths via NaN-padding internally. PCMFG's core problem (50-chapter novel vs 10-chapter novella) is handled out of the box.
5. **Active maintenance**: v0.8.1 released Mar 2026, 77 open issues being triaged, regular releases. dtaidistance is also maintained (v2.4.0 Feb 2026) but has a narrower scope.

### Interpolation Methods

| Method | Library | Recommendation |
|--------|---------|----------------|
| `scipy.interpolate.interp1d` (linear) | scipy | **Default choice**. Linear interpolation preserves the shape of emotional trajectories without introducing artifacts. Fast, predictable, no overshooting. |
| `scipy.interpolate.interp1d` (cubic) | scipy | Use when trajectories are smooth and you want to reduce staircase artifacts from linear interpolation. Risk: cubic can overshoot on noisy emotion scores (1-5 integer scale), creating values outside [1, 5]. |
| `numpy.interp` | numpy | Simpler than scipy, but only does linear interpolation. Use if you want zero-dependency resampling. Since scipy is already available, prefer `interp1d` for its flexibility. |
| `pandas.DataFrame.interpolate` | pandas | Good for filling NaN gaps within a series, but not designed for resampling to a new grid. Use for the forward-fill imputation (Phase 3 existing), not for normalization resampling. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **fastdtw** | Abandoned since Oct 2019. No Python 3.12+ wheels. Only provides approximate DTW — the approximation is available in tslearn. | tslearn with `global_constraint="sakoe_chiba"` parameter for bounded DTW |
| **tsfresh** | Designed for feature extraction from time-series for classical ML, not for DTW distance computation or time-series clustering. Wrong abstraction level. | tslearn for DTW + clustering; existing `feature_extractor.py` for feature extraction |
| **pydtw** | Minimal API, pure Python (slow), no clustering support, last meaningful update years ago. | tslearn |
| **Custom DTW implementation** | Reinventing the wheel. DTW has well-known edge cases (window constraints, multivariate handling, barycenter averaging) that tslearn has already solved. | tslearn's `dtw()` function |
| **D3.js / Plotly for overlay viz** | The project already uses matplotlib. Adding a JS-based visualization layer adds complexity without benefit for a CLI tool. | matplotlib with `alpha` transparency for trajectory overlays |

## Stack Patterns by Variant

**If clustering fewer than 50 narratives:**
- Use tslearn's `TimeSeriesKMeans` with `metric="dtw"` directly
- No speed optimizations needed — exact DTW is fast enough

**If clustering 50+ narratives:**
- Use tslearn's `TimeSeriesKMeans` with `metric="dtw"` and `global_constraint="sakoe_chiba"` with `sakoe_chiba_radius` set to ~10% of sequence length
- This bounds the warping window, reducing O(N²) to approximately O(N×W) where W is the window width
- Consider `n_jobs=-1` for parallel distance computation

**If you need silhouette scoring with DTW distance:**
- Use tslearn's `tslearn.metrics.dtw()` as the metric function in `sklearn.metrics.silhouette_score`
- This works because tslearn's dtw function accepts two 1D arrays and returns a float — compatible with sklearn's metric API

**For normalization/resampling:**
- Primary: `scipy.interpolate.interp1d(kind='linear')` to resample to uniform grid
- Primary: `scipy.interpolate.interp1d(kind='cubic')` if smoother curves are needed and overshooting is acceptable
- The existing `position` field (0.0-1.0) in chunks provides the x-axis for interpolation

## Version Compatibility

| Package | Version | Compatible With | Notes |
|-----------|-----------------|-------|
| tslearn 0.8.1 | Python >=3.10 | scikit-learn >=1.3.0, numpy, scipy | Requires `numpy` and `scikit-learn` as dependencies (both already in project) |
| scipy 1.17.1 | Python >=3.11 | numpy >=1.25.0 | Already pulled in transitively by scikit-learn |
| dtaidistance 2.4.0 | Python >=3.8 | numpy (optional), Cython (optional) | NOT recommended — listed here for reference |

## Installation

```bash
# Single new dependency
uv add "tslearn>=0.8.1"

# Or with pip
pip install "tslearn>=0.8.1"
```

No other new packages needed. scipy is already available via scikit-learn's dependency chain.

## Sources

- [tslearn on PyPI](https://pypi.org/project/tslearn/) — Verified v0.8.1, Mar 13 2026, Python >=3.10, BSD-2-Clause (HIGH confidence)
- [tslearn GitHub](https://github.com/tslearn-team/tslearn) — 3.1k stars, active development, 1,750 commits (HIGH confidence)
- [dtaidistance on PyPI](https://pypi.org/project/dtaidistance/) — Verified v2.4.0, Feb 12 2026, Apache-2.0 (HIGH confidence)
- [fastdtw on PyPI](https://pypi.org/project/fastdtw/) — Verified v0.3.4, Oct 7 2019, abandoned, no Python 3.12+ wheels (HIGH confidence)
- [scipy on PyPI](https://pypi.org/project/scipy/) — Verified v1.17.1, Feb 23 2026, Python >=3.11 (HIGH confidence)
- tslearn documentation (readthedocs) — API reference for TimeSeriesKMeans, dtw, TimeSeriesResampler, preprocessing module (MEDIUM confidence — some doc URLs 404'd but GitHub README confirms API)

---
*Stack research for: Narrative time-series normalization and DTW-based clustering*
*Researched: 2026-04-23*
