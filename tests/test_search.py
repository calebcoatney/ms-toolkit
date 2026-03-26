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


# ---------------------------------------------------------------------------
# Task 4: search_vector weighted_cosine fast path
# ---------------------------------------------------------------------------

def test_search_vector_weighted_cosine_matches_old_path(toolkit):
    """Fast path (weighted_cosine) returns same top compound as old compare_spectra."""
    from ms_toolkit.similarity import compare_spectra
    from ms_toolkit.preprocessing import preprocess_spectrum

    query = QUERY
    query_spectrum = [(mz + toolkit.mz_shift, i) for mz, i in query]

    # Old path: compare against all 5 compounds directly
    old_results = compare_spectra(
        query_spectrum,
        toolkit.library,
        max_mz=toolkit.max_mz,
        weighting_scheme="None",
        similarity_measure="weighted_cosine",
        unmatched_method="keep_all",
    )

    # New path via search_vector (top_k_clusters=2 selects all compounds with 2-cluster preselector)
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


# ---------------------------------------------------------------------------
# Task 6: w2v matrix (structure test only — no model file required)
# ---------------------------------------------------------------------------

def test_w2v_matrix_raises_without_model(toolkit):
    """_build_w2v_matrix raises if w2v_model is None."""
    with pytest.raises(RuntimeError, match="Word2Vec model must be loaded"):
        toolkit._build_w2v_matrix()
