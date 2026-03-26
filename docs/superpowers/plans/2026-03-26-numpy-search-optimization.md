# Numpy Search Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-query library preprocessing by precomputing normalized search matrices, replacing N scalar Python operations with two BLAS matmuls per search call.

**Architecture:** Add a lazy matrix cache on `MSToolkit` keyed by weighting scheme. At first search with a given scheme, preprocess and normalize all library spectra into a float32 matrix once. Subsequent searches extract a preselector-selected submatrix and call `matrix @ query_vec`. For composite similarity, cosine and overlap batch as matmuls; ratio factors remain a per-compound loop but operate on numpy arrays instead of dicts.

**Tech Stack:** Python 3.8+, NumPy, scikit-learn (KMeans already used), gensim (Word2Vec already used), pytest

---

## File Map

| File | Change |
|---|---|
| `ms_toolkit/preprocessing.py` | Add `preprocess_to_vector()` |
| `ms_toolkit/similarity.py` | Add `batch_cosine_similarity()`, `batch_overlap()`, `composite_ratio_factors()` |
| `ms_toolkit/api.py` | Add matrix cache attrs to `__init__`, add `_build_search_matrix()`, `_get_search_matrix()`, `_build_w2v_matrix()`, update `search_vector()`, `search_w2v()`, `load_w2v()` |
| `tests/test_search.py` | Replace placeholder with real tests |

---

## Task 1: Add `preprocess_to_vector` to preprocessing.py

**Files:**
- Modify: `ms_toolkit/preprocessing.py` (append after line 126)
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing tests**

Replace the contents of `tests/test_search.py` with:

```python
import numpy as np
import pytest
from ms_toolkit.models import Compound
from ms_toolkit.preprocessing import (
    preprocess_spectrum, preprocess_to_vector, align_spectra
)
from ms_toolkit.similarity import (
    dot_product_similarity, compare_spectra,
    batch_cosine_similarity, batch_overlap, composite_ratio_factors,
    composite_weighted_cosine_similarity,
)
from ms_toolkit.api import MSToolkit
from sklearn.cluster import KMeans
from ms_toolkit.preselector import ClusterPreselector


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SPECTRA = {
    'ethanol':  [(31, 1000.0), (45, 800.0), (46, 400.0), (27, 200.0)],
    'methanol': [(31, 1000.0), (29, 600.0), (32, 200.0), (28, 100.0)],
    'acetone':  [(43, 1000.0), (58, 800.0), (15, 600.0), (42, 400.0)],
    'benzene':  [(78, 1000.0), (77, 600.0), (51, 400.0), (50, 200.0)],
    'toluene':  [(91, 1000.0), (92, 800.0), (65, 400.0), (63, 200.0)],
}

QUERY = [(31, 900.0), (45, 700.0), (46, 350.0), (27, 150.0)]


@pytest.fixture
def small_library():
    return {name: Compound(name=name, spectrum=spec) for name, spec in SPECTRA.items()}


@pytest.fixture
def toolkit(small_library):
    tk = MSToolkit(vector_max_mz=1000, n_clusters=2)
    tk.library = small_library
    tk.vectorize_library()
    mat = np.vstack(list(tk.vectors.values()))
    keys = list(tk.vectors.keys())
    tk.preselector = ClusterPreselector(mat, library_keys=keys, n_clusters=2, random_state=42)
    return tk


# ---------------------------------------------------------------------------
# Task 1: preprocess_to_vector
# ---------------------------------------------------------------------------

def test_preprocess_to_vector_shape_and_dtype():
    vec = preprocess_to_vector(QUERY, weighting_scheme="None", max_mz=1000)
    assert vec.shape == (1001,), f"Expected shape (1001,), got {vec.shape}"
    assert vec.dtype == np.float32


def test_preprocess_to_vector_is_unit_norm():
    vec = preprocess_to_vector(QUERY, weighting_scheme="None", max_mz=1000)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_preprocess_to_vector_cosine_matches_old_path():
    """Dot product of two preprocess_to_vector results == old cosine similarity."""
    spec_a = QUERY
    spec_b = SPECTRA['ethanol']
    new_sim = float(np.dot(
        preprocess_to_vector(spec_a, "None"),
        preprocess_to_vector(spec_b, "None"),
    ))
    old_sim = dot_product_similarity(
        preprocess_spectrum(spec_a, weighting_scheme="None"),
        preprocess_spectrum(spec_b, weighting_scheme="None"),
    )
    assert abs(new_sim - old_sim) < 1e-4, f"new={new_sim}, old={old_sim}"


def test_preprocess_to_vector_nist_gc_weighting():
    """Weighting scheme affects the output."""
    vec_none = preprocess_to_vector(QUERY, weighting_scheme="None")
    vec_gc   = preprocess_to_vector(QUERY, weighting_scheme="NIST_GC")
    assert not np.allclose(vec_none, vec_gc), "NIST_GC should differ from None"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py::test_preprocess_to_vector_shape_and_dtype \
       tests/test_search.py::test_preprocess_to_vector_is_unit_norm \
       tests/test_search.py::test_preprocess_to_vector_cosine_matches_old_path \
       tests/test_search.py::test_preprocess_to_vector_nist_gc_weighting \
       -v 2>&1 | head -40
```

Expected: `ImportError: cannot import name 'preprocess_to_vector'`

- [ ] **Step 3: Implement `preprocess_to_vector`**

Append to the end of `ms_toolkit/preprocessing.py`:

```python
def preprocess_to_vector(spectrum, weighting_scheme="None", max_mz=1000, bin_width=1.0):
    """
    Preprocess spectrum and return a normalized dense float32 vector.
    Combines filter → tic_scale → mass_weight → vectorize → L2-normalize in one pass.
    Used for library matrix construction and query preprocessing at search time.
    """
    spectrum = filter_low_intensity_peaks(spectrum)
    spectrum = tic_scaling(spectrum)
    a, b = weighting_schemes.get(weighting_scheme, (0, 1))
    spectrum = mass_weighting(spectrum, a, b)
    vec = spectrum_to_vector(spectrum, max_mz=max_mz, bin_width=bin_width).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py::test_preprocess_to_vector_shape_and_dtype \
       tests/test_search.py::test_preprocess_to_vector_is_unit_norm \
       tests/test_search.py::test_preprocess_to_vector_cosine_matches_old_path \
       tests/test_search.py::test_preprocess_to_vector_nist_gc_weighting \
       -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
git add ms_toolkit/preprocessing.py tests/test_search.py
git commit -m "feat: add preprocess_to_vector for batch library preprocessing"
```

---

## Task 2: Add batch similarity functions to similarity.py

**Files:**
- Modify: `ms_toolkit/similarity.py` (append after line 83)
- Test: `tests/test_search.py`

- [ ] **Step 1: Append the failing tests to `tests/test_search.py`**

```python
# ---------------------------------------------------------------------------
# Task 2: batch similarity functions
# ---------------------------------------------------------------------------

def test_batch_cosine_similarity_matches_dot_product_similarity():
    """batch_cosine_similarity on a 1-row matrix == dot_product_similarity."""
    spec_a = QUERY
    spec_b = SPECTRA['ethanol']
    q_vec = preprocess_to_vector(spec_a, "None")
    l_vec = preprocess_to_vector(spec_b, "None")
    library_matrix = l_vec.reshape(1, -1)

    batch_score = batch_cosine_similarity(q_vec, library_matrix)[0]
    old_score = dot_product_similarity(
        preprocess_spectrum(spec_a, weighting_scheme="None"),
        preprocess_spectrum(spec_b, weighting_scheme="None"),
    )
    assert abs(float(batch_score) - old_score) < 1e-4


def test_batch_cosine_similarity_returns_one_score_per_row():
    vecs = np.vstack([preprocess_to_vector(s, "None") for s in SPECTRA.values()])
    q = preprocess_to_vector(QUERY, "None")
    scores = batch_cosine_similarity(q, vecs)
    assert scores.shape == (5,)
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0 + 1e-5)


def test_batch_overlap_counts_common_nonzero_peaks():
    q_vec = preprocess_to_vector(QUERY, "None")
    l_vec = preprocess_to_vector(SPECTRA['ethanol'], "None")
    q_bool = (q_vec > 0).astype(np.float32)
    l_bool = (l_vec > 0).astype(np.float32).reshape(1, -1)
    overlap = batch_overlap(q_bool, l_bool)[0]
    # Manual count: common non-zero m/z between QUERY and ethanol
    common = set(mz for mz, _ in QUERY) & set(mz for mz, _ in SPECTRA['ethanol'])
    assert int(overlap) == len(common)


def test_composite_ratio_factors_returns_float_in_zero_one():
    q_vec = preprocess_to_vector(QUERY, "None")
    l_vec = preprocess_to_vector(SPECTRA['ethanol'], "None")
    rf = composite_ratio_factors(q_vec, l_vec)
    assert isinstance(rf, float)
    assert 0.0 <= rf <= 1.0


def test_composite_ratio_factors_one_when_no_common_peaks():
    q_vec = preprocess_to_vector([(10, 100.0)], "None")
    l_vec = preprocess_to_vector([(900, 100.0)], "None")
    assert composite_ratio_factors(q_vec, l_vec) == 1.0


def test_composite_ratio_factors_matches_old_path():
    """avg_ratio_factor from composite_ratio_factors matches the old dict-based path."""
    spec_a = QUERY
    spec_b = SPECTRA['ethanol']

    # Old path extracts avg_ratio_factor as part of composite score.
    # We reconstruct it by comparing composite_sim to cosine_sim.
    proc_a = preprocess_spectrum(spec_a, weighting_scheme="None")
    proc_b = preprocess_spectrum(spec_b, weighting_scheme="None")

    dict_a = {mz: i for mz, i in proc_a}
    dict_b = {mz: i for mz, i in proc_b}
    common_keys = sorted(set(dict_a) & set(dict_b))
    ratio_factors = []
    for i in range(1, len(common_keys)):
        mz_prev, mz_curr = common_keys[i-1], common_keys[i]
        if dict_a.get(mz_prev, 0) and dict_b.get(mz_prev, 0):
            r_a = dict_a[mz_curr] / dict_a[mz_prev]
            r_b = dict_b[mz_curr] / dict_b[mz_prev]
            ratio_factors.append(min(r_a, r_b) / max(r_a, r_b))
    expected_rf = float(np.mean(ratio_factors)) if ratio_factors else 1.0

    q_vec = preprocess_to_vector(spec_a, "None")
    l_vec = preprocess_to_vector(spec_b, "None")
    new_rf = composite_ratio_factors(q_vec, l_vec)

    assert abs(new_rf - expected_rf) < 1e-4, f"new={new_rf}, old={expected_rf}"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py -k "batch or ratio_factors" -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'batch_cosine_similarity'`

- [ ] **Step 3: Append the three functions to `ms_toolkit/similarity.py`**

```python
# --- Batch Similarity (fast path for matrix search) ---

def batch_cosine_similarity(query_vec, library_matrix):
    """
    Cosine similarity between query_vec and every row of library_matrix.
    Both must be L2-normalized. Returns shape (n_compounds,).
    """
    return library_matrix @ query_vec


def batch_overlap(query_bool, library_bool_matrix):
    """
    Count of common non-zero m/z peaks between query and each library row.
    query_bool: float32 binary vector shape (n_mz,)
    library_bool_matrix: float32 binary matrix shape (n_compounds, n_mz)
    Returns shape (n_compounds,).
    """
    return library_bool_matrix @ query_bool


def composite_ratio_factors(query_vec, lib_vec):
    """
    Compute avg_ratio_factor for one query/library pair using dense numpy arrays.
    Replaces the dict-construction + set-intersection + Python inner loop in
    composite_weighted_cosine_similarity. Ratio is scale-invariant so pre-normalized
    vectors give the same result as raw preprocessed vectors.
    Returns float in [0, 1]; returns 1.0 if fewer than 2 common peaks.
    """
    common_idx = np.where((query_vec > 0) & (lib_vec > 0))[0]
    if len(common_idx) < 2:
        return 1.0
    r_q = query_vec[common_idx[1:]] / query_vec[common_idx[:-1]]
    r_l = lib_vec[common_idx[1:]] / lib_vec[common_idx[:-1]]
    rf = np.minimum(r_q, r_l) / np.maximum(r_q, r_l)
    return float(np.mean(rf))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py -k "batch or ratio_factors" -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
git add ms_toolkit/similarity.py tests/test_search.py
git commit -m "feat: add batch_cosine_similarity, batch_overlap, composite_ratio_factors"
```

---

## Task 3: Add matrix cache infrastructure to MSToolkit

**Files:**
- Modify: `ms_toolkit/api.py` lines 74–78 (`__init__` body after `self.preselector = None`)
- Test: `tests/test_search.py`

- [ ] **Step 1: Append the failing tests**

```python
# ---------------------------------------------------------------------------
# Task 3: matrix cache infrastructure
# ---------------------------------------------------------------------------

def test_build_search_matrix_shape(toolkit):
    toolkit._build_search_matrix("None")
    assert "None" in toolkit._search_matrices
    assert toolkit._search_matrices["None"].shape == (5, 1001)
    assert toolkit._search_matrices["None"].dtype == np.float32


def test_build_search_matrix_rows_are_unit_norm(toolkit):
    toolkit._build_search_matrix("None")
    mat = toolkit._search_matrices["None"]
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_get_search_matrix_lazy(toolkit):
    assert "None" not in toolkit._search_matrices
    toolkit._get_search_matrix("None")
    assert "None" in toolkit._search_matrices


def test_get_search_matrix_cached(toolkit):
    m1 = toolkit._get_search_matrix("None")
    m2 = toolkit._get_search_matrix("None")
    assert m1 is m2  # same object — not rebuilt


def test_get_search_matrix_raises_without_library():
    tk = MSToolkit()
    with pytest.raises(RuntimeError, match="Library must be loaded"):
        tk._get_search_matrix("None")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py -k "search_matrix" -v 2>&1 | head -30
```

Expected: `AttributeError: 'MSToolkit' object has no attribute '_search_matrices'`

- [ ] **Step 3: Add cache attributes to `__init__` and implement `_build_search_matrix` / `_get_search_matrix`**

In `ms_toolkit/api.py`, find the line `self.preselector = None` (line 78) and add after it:

```python
        self._search_matrices: dict = {}          # weighting_scheme → (n, max_mz+1) float32
        self._search_matrix_keys: list = []       # compound keys in row order
        self._search_matrix_key_to_idx: dict = {} # key → row index
        self._w2v_matrices: dict = {}             # (n_decimals, intensity_power) → (n, vec_size) float32
        self._w2v_matrix_keys: list = []
        self._w2v_matrix_key_to_idx: dict = {}
```

Then add these two methods to the `MSToolkit` class, before `search_vector` (around line 465):

```python
    def _build_search_matrix(self, weighting_scheme: str) -> None:
        """
        Build and cache a preprocessed, normalized float32 matrix for weighting_scheme.
        Rows correspond to self.library compounds in insertion order.
        """
        from .preprocessing import preprocess_to_vector
        keys = list(self.library.keys())
        rows = [
            preprocess_to_vector(self.library[k].spectrum,
                                 weighting_scheme=weighting_scheme,
                                 max_mz=self.max_mz)
            for k in keys
        ]
        self._search_matrices[weighting_scheme] = np.vstack(rows).astype(np.float32)
        self._search_matrix_keys = keys
        self._search_matrix_key_to_idx = {k: i for i, k in enumerate(keys)}

    def _get_search_matrix(self, weighting_scheme: str) -> np.ndarray:
        """Return cached search matrix, building it if not yet cached."""
        if self.library is None:
            raise RuntimeError("Library must be loaded before building search matrix")
        if weighting_scheme not in self._search_matrices:
            self._build_search_matrix(weighting_scheme)
        return self._search_matrices[weighting_scheme]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py -k "search_matrix" -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
git add ms_toolkit/api.py tests/test_search.py
git commit -m "feat: add lazy search matrix cache to MSToolkit"
```

---

## Task 4: Update `search_vector` — weighted_cosine fast path

**Files:**
- Modify: `ms_toolkit/api.py` — `search_vector` method (lines 465–539)
- Test: `tests/test_search.py`

- [ ] **Step 1: Append the failing test**

```python
# ---------------------------------------------------------------------------
# Task 4: search_vector weighted_cosine fast path
# ---------------------------------------------------------------------------

def test_search_vector_weighted_cosine_matches_old_path(toolkit):
    """Fast path (weighted_cosine) returns same top compound as old compare_spectra."""
    from ms_toolkit.similarity import compare_spectra
    from ms_toolkit.preprocessing import preprocess_spectrum

    query = QUERY
    query_spectrum = [(mz + toolkit.mz_shift, i) for mz, i in query]
    proc_query = preprocess_spectrum(query_spectrum, weighting_scheme="None")

    # Old path: compare against all 5 compounds directly
    old_results = compare_spectra(
        query_spectrum,
        toolkit.library,
        max_mz=toolkit.max_mz,
        weighting_scheme="None",
        similarity_measure="weighted_cosine",
        unmatched_method="keep_all",
    )

    # New path via search_vector (preselector selects all 2-cluster members)
    new_results = toolkit.search_vector(
        query, top_n=5, weighting_scheme="None",
        composite=False, unmatched_method="keep_all", top_k_clusters=2
    )

    # Top compound should match
    assert old_results[0][0] == new_results[0][0], (
        f"Old top: {old_results[0][0]}, New top: {new_results[0][0]}"
    )
    # Scores should be close
    assert abs(old_results[0][1] - new_results[0][1]) < 1e-4
```

- [ ] **Step 2: Run test to confirm it currently passes (or note current behavior)**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py::test_search_vector_weighted_cosine_matches_old_path -v
```

Note the result — it may already pass since the old path is still active. This test is the regression guard for the next step.

- [ ] **Step 3: Replace the body of `search_vector` with the fast path**

In `ms_toolkit/api.py`, replace the `search_vector` method body with:

```python
    def search_vector(
        self,
        query_input,
        top_n=10,
        weighting_scheme="None",
        composite=False,
        unmatched_method="keep_all",
        top_k_clusters=1):

        if self.preselector is None:
            raise RuntimeError("Preselector must be loaded first")

        # Build query spectrum and vector (same as before)
        if isinstance(query_input, np.ndarray):
            if len(query_input) != (self.max_mz + 1):
                query_spectrum = vector_to_spectrum(query_input, shift=self.mz_shift)
                query_vector = spectrum_to_vector(query_spectrum, max_mz=self.max_mz)
            else:
                query_vector = query_input
                query_spectrum = vector_to_spectrum(query_input, shift=self.mz_shift)
        else:
            query_spectrum = [(mz + self.mz_shift, intensity) for mz, intensity in query_input]
            query_vector = spectrum_to_vector(query_spectrum, max_mz=self.max_mz)

        # Preselector
        if isinstance(self.preselector, ClusterPreselector):
            selected_keys = self.preselector.select(
                query_vector, list(self.library.keys()), top_k_clusters=top_k_clusters)
        elif isinstance(self.preselector, GMMPreselector):
            selected_keys = self.preselector.select(
                query_vector, list(self.library.keys()), top_k_components=top_k_clusters)
        else:
            selected_keys = self.preselector.select(query_vector, list(self.library.keys()))

        # Fast path: keep_all alignment only (default), batched matmul
        if unmatched_method == "keep_all":
            from .preprocessing import preprocess_to_vector
            from .similarity import (batch_cosine_similarity, batch_overlap,
                                     composite_ratio_factors)

            matrix = self._get_search_matrix(weighting_scheme)
            query_vec = preprocess_to_vector(
                query_spectrum, weighting_scheme=weighting_scheme, max_mz=self.max_mz)

            # Map selected keys to matrix row indices
            valid_keys = [k for k in selected_keys if k in self._search_matrix_key_to_idx]
            indices = [self._search_matrix_key_to_idx[k] for k in valid_keys]
            sub_matrix = matrix[indices]  # (n_selected, max_mz+1)

            if not composite:
                # weighted_cosine: single matmul
                scores = batch_cosine_similarity(query_vec, sub_matrix)
                results = [
                    (self.library[k].name, float(scores[i]))
                    for i, k in enumerate(valid_keys)
                ]
            else:
                # composite: batch cosine + overlap, per-compound ratio loop
                cosine_scores = batch_cosine_similarity(query_vec, sub_matrix)
                query_bool = (query_vec > 0).astype(np.float32)
                lib_bool = (sub_matrix > 0).astype(np.float32)
                overlap_scores = batch_overlap(query_bool, lib_bool)
                N = int(np.sum(query_vec > 0))

                results = []
                for i, k in enumerate(valid_keys):
                    cos = float(cosine_scores[i])
                    ov = float(overlap_scores[i])
                    rf = composite_ratio_factors(query_vec, sub_matrix[i])
                    score = (N * cos + ov * rf) / (N + ov) if (N + ov) > 0 else 0.0
                    results.append((self.library[k].name, score))

            return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

        # Fallback: non-default unmatched_method — use old compare_spectra path
        subset = {k: self.library[k] for k in selected_keys if k in self.library}
        similarity_measure = "composite" if composite else "weighted_cosine"
        results = compare_spectra(
            query_spectrum,
            subset,
            max_mz=self.max_mz,
            weighting_scheme=weighting_scheme,
            similarity_measure=similarity_measure,
            unmatched_method=unmatched_method,
        )
        return results[:top_n]
```

- [ ] **Step 4: Run all search tests so far**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py -v
```

Expected: all previously passing tests still pass, plus the new test.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
git add ms_toolkit/api.py tests/test_search.py
git commit -m "perf: use batched matmul in search_vector fast path"
```

---

## Task 5: Verify composite fast path and fallback

**Files:**
- Test: `tests/test_search.py`
- No code changes needed — composite path was included in Task 4's `search_vector`

- [ ] **Step 1: Append the composite and fallback tests**

```python
# ---------------------------------------------------------------------------
# Task 5: composite fast path + fallback
# ---------------------------------------------------------------------------

def test_search_vector_composite_matches_old_path(toolkit):
    """composite=True fast path returns same top compound as old compare_spectra."""
    from ms_toolkit.similarity import compare_spectra

    query_spectrum = [(mz + toolkit.mz_shift, i) for mz, i in QUERY]

    old_results = compare_spectra(
        query_spectrum,
        toolkit.library,
        max_mz=toolkit.max_mz,
        weighting_scheme="None",
        similarity_measure="composite",
        unmatched_method="keep_all",
    )
    new_results = toolkit.search_vector(
        QUERY, top_n=5, weighting_scheme="None",
        composite=True, unmatched_method="keep_all", top_k_clusters=2
    )

    assert old_results[0][0] == new_results[0][0], (
        f"Old top: {old_results[0][0]}, New top: {new_results[0][0]}"
    )
    assert abs(old_results[0][1] - new_results[0][1]) < 1e-3


def test_search_vector_fallback_remove_all(toolkit):
    """unmatched_method='remove_all' still returns results via old path."""
    results = toolkit.search_vector(
        QUERY, top_n=3, weighting_scheme="None",
        composite=False, unmatched_method="remove_all", top_k_clusters=2
    )
    assert len(results) > 0
    assert all(isinstance(name, str) and isinstance(score, float) for name, score in results)


def test_search_vector_matrix_cached_after_search(toolkit):
    """Running search_vector populates _search_matrices."""
    assert "None" not in toolkit._search_matrices
    toolkit.search_vector(QUERY, weighting_scheme="None", top_k_clusters=2)
    assert "None" in toolkit._search_matrices


def test_search_vector_matrix_not_rebuilt_on_second_call(toolkit):
    """Second call with same weighting_scheme reuses cached matrix."""
    toolkit.search_vector(QUERY, weighting_scheme="None", top_k_clusters=2)
    m1 = toolkit._search_matrices["None"]
    toolkit.search_vector(QUERY, weighting_scheme="None", top_k_clusters=2)
    m2 = toolkit._search_matrices["None"]
    assert m1 is m2
```

- [ ] **Step 2: Run tests**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py -v
```

Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
git add tests/test_search.py
git commit -m "test: add composite fast path, fallback, and cache behavior tests"
```

---

## Task 6: Precompute Word2Vec embedding matrix and update `search_w2v`

**Files:**
- Modify: `ms_toolkit/api.py` — add `_build_w2v_matrix()`, update `load_w2v()`, replace `search_w2v()` body

This task has no isolated unit test for the w2v path (it requires a trained model file), so the test verifies the error guard. Score equivalence is confirmed by the manual timing smoke test in Task 7.

- [ ] **Step 1: Append the w2v infrastructure test**

```python
# ---------------------------------------------------------------------------
# Task 6: w2v matrix (structure test only — no model file required)
# ---------------------------------------------------------------------------

def test_w2v_matrix_raises_without_model(toolkit):
    """_build_w2v_matrix raises if w2v_model is None."""
    with pytest.raises(RuntimeError, match="Word2Vec model must be loaded"):
        toolkit._build_w2v_matrix()
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py::test_w2v_matrix_raises_without_model -v 2>&1 | head -20
```

Expected: `AttributeError: 'MSToolkit' object has no attribute '_build_w2v_matrix'`

- [ ] **Step 3: Add `_build_w2v_matrix` to `api.py`**

Add this method to `MSToolkit`, just after `_get_search_matrix`:

```python
    def _build_w2v_matrix(self, n_decimals=2, intensity_power=0.6):
        """
        Precompute normalized float32 embedding matrix for all library compounds.
        Keyed by (n_decimals, intensity_power) in self._w2v_matrices.
        Call after load_w2v() to amortize per-compound calc_embedding cost.
        """
        if self.w2v_model is None:
            raise RuntimeError("Word2Vec model must be loaded before building embedding matrix")
        key = (n_decimals, intensity_power)
        if key in self._w2v_matrices:
            return
        from .models import SpectrumDocument
        from .w2v import calc_embedding
        lib_keys = list(self.library.keys())
        rows = []
        for k in lib_keys:
            doc = SpectrumDocument(self.library[k].spectrum, n_decimals=n_decimals)
            emb = calc_embedding(self.w2v_model, doc, intensity_power).astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            rows.append(emb)
        self._w2v_matrices[key] = np.vstack(rows).astype(np.float32)
        self._w2v_matrix_keys = lib_keys
        self._w2v_matrix_key_to_idx = {k: i for i, k in enumerate(lib_keys)}
```

- [ ] **Step 4: Update `load_w2v` to call `_build_w2v_matrix` after loading**

In `load_w2v`, find the line `self.w2v_model = load_model(path)` and add below it:

```python
        if self.library is not None:
            # Detect n_decimals from model vocabulary
            n_decimals = 2
            vocab = self.w2v_model.wv.key_to_index
            if vocab:
                sample = next(iter(vocab))
                if "peak@" in sample:
                    parts = sample.split("peak@")
                    if len(parts) > 1 and "." in parts[1]:
                        n_decimals = len(parts[1].split(".")[1])
                    else:
                        n_decimals = 0
            self._build_w2v_matrix(n_decimals=n_decimals)
```

- [ ] **Step 5: Replace the body of `search_w2v` with the fast path**

```python
    def search_w2v(self, query_input, top_n=10, intensity_power=0.6, top_k_clusters=1, n_decimals=2):
        if self.w2v_model is None:
            raise RuntimeError("Word2Vec model must be loaded first")
        if self.preselector is None:
            raise RuntimeError("Preselector must be loaded first")

        # Build query spectrum and vector
        if isinstance(query_input, np.ndarray):
            if len(query_input) != (self.max_mz + 1):
                query_spectrum = vector_to_spectrum(query_input, shift=self.mz_shift)
                query_vector = spectrum_to_vector(query_spectrum, max_mz=self.max_mz)
            else:
                query_vector = query_input
                query_spectrum = vector_to_spectrum(query_input, shift=self.mz_shift)
        else:
            query_spectrum = [(mz + self.mz_shift, intensity) for mz, intensity in query_input]
            query_vector = spectrum_to_vector(query_spectrum, max_mz=self.max_mz)

        # Preselector
        if isinstance(self.preselector, ClusterPreselector):
            selected_keys = self.preselector.select(
                query_vector, list(self.library.keys()), top_k_clusters=top_k_clusters)
        elif isinstance(self.preselector, GMMPreselector):
            selected_keys = self.preselector.select(
                query_vector, list(self.library.keys()), top_k_components=top_k_clusters)
        else:
            selected_keys = self.preselector.select(query_vector, list(self.library.keys()))

        from .models import SpectrumDocument
        from .w2v import calc_embedding

        # Build or retrieve w2v matrix
        matrix_key = (n_decimals, intensity_power)
        if matrix_key not in self._w2v_matrices:
            self._build_w2v_matrix(n_decimals=n_decimals, intensity_power=intensity_power)

        w2v_mat = self._w2v_matrices[matrix_key]

        # Compute query embedding
        query_doc = SpectrumDocument(query_spectrum, n_decimals=n_decimals)
        query_emb = calc_embedding(self.w2v_model, query_doc, intensity_power).astype(np.float32)

        # Auto-detect n_decimals if query embedding is all zeros
        if np.all(query_emb == 0):
            vocab = self.w2v_model.wv.key_to_index
            if vocab:
                sample = next(iter(vocab))
                if "peak@" in sample:
                    parts = sample.split("peak@")
                    detected = len(parts[1].split(".")[1]) if "." in parts[1] else 0
                    if detected != n_decimals:
                        n_decimals = detected
                        matrix_key = (n_decimals, intensity_power)
                        if matrix_key not in self._w2v_matrices:
                            self._build_w2v_matrix(n_decimals=n_decimals,
                                                   intensity_power=intensity_power)
                        w2v_mat = self._w2v_matrices[matrix_key]
                        query_doc = SpectrumDocument(query_spectrum, n_decimals=n_decimals)
                        query_emb = calc_embedding(
                            self.w2v_model, query_doc, intensity_power).astype(np.float32)

        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return []
        query_emb /= query_norm

        # Subset matrix by preselector keys
        valid_keys = [k for k in selected_keys if k in self._w2v_matrix_key_to_idx]
        indices = [self._w2v_matrix_key_to_idx[k] for k in valid_keys]
        sub_matrix = w2v_mat[indices]  # (n_selected, vector_size)

        scores = sub_matrix @ query_emb
        results = [
            (self.library[k].name, float(scores[i]))
            for i, k in enumerate(valid_keys)
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
```

- [ ] **Step 6: Run all tests**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py -v
```

Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
git add ms_toolkit/api.py tests/test_search.py
git commit -m "perf: precompute w2v embedding matrix and use batched matmul in search_w2v"
```

---

## Task 7: Full test suite run and manual smoke test

- [ ] **Step 1: Run the full test suite**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
pytest tests/test_search.py -v
```

Expected: all tests pass, 0 failures

- [ ] **Step 2: Manual timing smoke test**

Run this snippet in a Python REPL or script to verify the speedup is real:

```python
import time
from ms_toolkit.api import MSToolkit

# Load library (use your actual path)
tk = MSToolkit()
tk.load_library(json_path='cache/your_library.json')
tk.load_preselector('cache/your_preselector.pkl')

query = [(43, 1000), (58, 800), (15, 600), (42, 400), (27, 200)]

# Warm up (builds matrix on first call)
tk.search_vector(query, top_n=10, composite=True, top_k_clusters=1)

# Time 20 searches
t0 = time.perf_counter()
for _ in range(20):
    tk.search_vector(query, top_n=10, composite=True, top_k_clusters=1)
elapsed = time.perf_counter() - t0
print(f"20 searches in {elapsed:.2f}s → {elapsed/20*1000:.1f}ms per search")
```

Expected: significant speedup over the old path on second+ calls.

- [ ] **Step 3: Commit final state if any cleanup was done**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/ms-toolkit"
git status  # confirm clean or commit any remaining changes
```
