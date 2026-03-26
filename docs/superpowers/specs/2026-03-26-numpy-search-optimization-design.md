# Design: Numpy Search Optimization

**Date:** 2026-03-26
**Status:** Approved
**Scope:** ms_toolkit — `preprocessing.py`, `similarity.py`, `api.py`

---

## Problem

The current `compare_spectra` loop re-preprocesses every library spectrum on every query via Python list comprehensions on tuple lists, dict construction, and per-compound scalar numpy calls. On a 300k-compound NIST EI library with 100 queries per GC-MS file, this produces an enormous amount of redundant Python work. Library spectra do not change between queries, so all preprocessing should happen once at load time.

---

## Goal

Close the majority of the performance gap with NIST software (estimated ~50x slower currently) by restructuring the search path to use precomputed, normalized library matrices and batched linear algebra. No new compiled dependencies. Public API unchanged.

---

## Architecture

The core change is moving library preprocessing from query time to load time.

**Current flow:**
```
search_vector(query):
  preselector.select() → selected_keys
  for each compound in selected_keys:
    preprocess_spectrum(compound.spectrum)   # Python list ops, N times per query
    align_spectra(proc_query, proc_lib)      # dict construction + list comps, N times
    np.dot / np.linalg.norm                  # fast but called N times individually
```

**New flow:**
```
_get_search_matrix(weighting_scheme):       # lazy, cached per scheme
  for each compound in library (once, at first search):
    preprocess_to_vector(compound.spectrum) # preprocessing done here, not at query time

search_vector(query):
  preprocess_to_vector(query)               # once per query
  preselector.select() → selected_keys
  sub_matrix = search_matrix[selected_indices]
  scores = sub_matrix @ query_vec           # single matmul
```

---

## Data Structures

Four new attributes on `MSToolkit`, all `None` until populated:

```python
self._search_matrices: dict[str, np.ndarray]   # weighting_scheme → (n, 1001) float32, rows pre-normalized
self._search_matrix_keys: list[str]             # compound keys in row order
self._search_matrix_key_to_idx: dict[str, int]  # reverse lookup for preselector subsetting

self._w2v_matrix: np.ndarray | None             # (n, vector_size) float32, rows pre-normalized
self._w2v_matrix_keys: list[str]                # compound keys in row order
self._w2v_matrix_key_to_idx: dict[str, int]     # reverse lookup
```

`_search_matrices` is a dict keyed by weighting scheme so multiple schemes can be cached independently across a session. Only schemes that are actually used get built (lazy).

**Memory:** 300k × 1001 × 4 bytes ≈ 1.2 GB per cached weighting scheme. `self.vectors` already occupies the same scale, so this is not new territory.

---

## Code Changes

### `preprocessing.py`

Add one new function:

```python
def preprocess_to_vector(spectrum, weighting_scheme="None", max_mz=1000) -> np.ndarray:
    """
    Preprocess spectrum and return a normalized dense float32 vector.
    Combines filter → tic_scale → mass_weight → vectorize → L2-normalize in one pass.
    Used for both library matrix construction and query preprocessing at search time.
    """
```

All existing functions remain unchanged.

### `similarity.py`

Add three new functions, keep all existing functions as fallback:

```python
def batch_cosine_similarity(query_vec, library_matrix) -> np.ndarray:
    """Single matmul. Both inputs pre-normalized — result is cosine similarity for each row."""

def batch_overlap(query_bool, library_bool_matrix) -> np.ndarray:
    """Count of common non-zero m/z peaks per library compound. Matmul on boolean matrices."""

def composite_ratio_factors(query_vec, lib_vec) -> float:
    """
    Compute avg_ratio_factor for one query/library pair using dense numpy arrays.
    Replaces the dict-construction + set-intersection + Python inner loop in the
    current composite_weighted_cosine_similarity.
    """
```

`compare_spectra` and `composite_weighted_cosine_similarity` remain untouched as fallback paths.

### `api.py`

New private methods:

```python
def _build_search_matrix(self, weighting_scheme: str) -> None:
    """
    Build and cache a preprocessed, normalized float32 matrix for the given weighting scheme.
    Loops over self.library once, calls preprocess_to_vector per compound, stacks with np.vstack.
    Populates self._search_matrices[weighting_scheme], self._search_matrix_keys,
    self._search_matrix_key_to_idx.
    """

def _get_search_matrix(self, weighting_scheme: str) -> np.ndarray:
    """Return cached matrix, building it first if not yet cached."""

def _build_w2v_matrix(self, n_decimals: int = 2, intensity_power: float = 0.6) -> None:
    """
    Precompute all library embeddings into a normalized float32 matrix.
    Called at load_w2v() time (eagerly, since the w2v model is fixed).
    """
```

Updated search methods:

- `search_vector`: calls `_get_search_matrix(weighting_scheme)`, maps preselector keys to row indices via `_search_matrix_key_to_idx`, extracts sub-matrix, runs `batch_cosine_similarity` (weighted_cosine path) or the partial-vectorization composite path (see below). Returns top-N.
- `search_w2v`: uses `self._w2v_matrix` and `self._w2v_matrix_key_to_idx` for subset extraction, single matmul for similarities.
- `load_w2v`: calls `_build_w2v_matrix` after loading the model.

---

## Composite Similarity: Partial Vectorization

The composite score has three components:

| Component | Approach |
|---|---|
| `cosine_sim` per compound | Batched matmul (one call for all candidates) |
| `overlap` per compound | Batched matmul on boolean matrices (one call for all candidates) |
| `avg_ratio_factor` per compound | Per-compound Python loop, but over numpy arrays |

The ratio factor loop cannot be batched because each query/library pair has a variable-length set of common peak indices. However, all dict construction, tuple iteration, set intersection, and list comprehensions are eliminated. Each loop iteration becomes:

```python
common_idx = np.where((lib_vec > 0) & (query_vec > 0))[0]
r_q = query_vec[common_idx[1:]] / query_vec[common_idx[:-1]]
r_l = lib_vec[common_idx[1:]] / lib_vec[common_idx[:-1]]
avg_rf = np.mean(np.minimum(r_q, r_l) / np.maximum(r_q, r_l))
```

This is ~4 numpy operations per compound instead of the current ~100+ Python operations (dict builds, set ops, list comps, inner loop).

**Fast path activates when:** `similarity_measure="composite"` or `similarity_measure="weighted_cosine"`, `unmatched_method="keep_all"` (default).
**Fallback to old path when:** `unmatched_method != "keep_all"` — the non-default alignment modes don't map to fixed-size dense vectors.

---

## Error Handling

- `_get_search_matrix` raises `RuntimeError("Library must be loaded before building search matrix")` if `self.library is None`.
- `_build_w2v_matrix` raises `RuntimeError("Word2Vec model must be loaded before building embedding matrix")` if `self.w2v_model is None`.
- Degenerate spectra (zero vector after preprocessing) store a zero row in the matrix, scoring 0 in matmul — same behavior as current path.

---

## Testing

Three things to verify in `tests/test_search.py`:

1. **Score equivalence** — for a sample of query/library pairs, results from the new path match the old path to within float tolerance (`np.allclose`, atol=1e-5).
2. **Cache behavior** — two consecutive searches with the same weighting scheme build the matrix only once (`len(self._search_matrices) == 1`).
3. **Fallback still works** — `unmatched_method="remove_all"` returns results via the old path without error.

---

## Out of Scope

- Numba JIT or Rust extension (deferred to a future optimization pass if needed after measuring this improvement)
- Caching search matrices to disk between sessions
- Supporting `unmatched_method != "keep_all"` on the fast path
- Vectorizing the composite ratio factor loop (variable-length common indices make this impractical without masking complexity that isn't worth it)
