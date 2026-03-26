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
