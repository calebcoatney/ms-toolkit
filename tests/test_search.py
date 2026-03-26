import numpy as np
import pytest
from ms_toolkit.models import Compound
from ms_toolkit.preprocessing import (
    preprocess_spectrum, preprocess_to_vector, align_spectra
)
from ms_toolkit.similarity import dot_product_similarity, compare_spectra

# Import functions that may not exist yet (for later tasks)
try:
    from ms_toolkit.similarity import (
        batch_cosine_similarity, batch_overlap, composite_ratio_factors,
        composite_weighted_cosine_similarity,
    )
except ImportError:
    pass

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
