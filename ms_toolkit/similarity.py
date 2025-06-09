from .preprocessing import align_spectra, preprocess_spectrum
import numpy as np

def dot_product_similarity(spectrum1, spectrum2, max_mz=1000, unmatched_method="keep_all"):
    vec1, vec2 = align_spectra(spectrum1, spectrum2, max_mz, unmatched_method)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

# --- Composite Weighted Cosine Similarity ---
def composite_weighted_cosine_similarity(query_spectrum, lib_spectrum, max_mz=1000,
                                         unmatched_method="keep_all", weighting_scheme="None"):
    proc_query = preprocess_spectrum(query_spectrum, weighting_scheme=weighting_scheme)
    proc_lib = preprocess_spectrum(lib_spectrum, weighting_scheme=weighting_scheme)
    
    cosine_sim = dot_product_similarity(proc_query, proc_lib, max_mz, unmatched_method)
    
    dict_query = {mz: intensity for mz, intensity in proc_query if mz <= max_mz}
    dict_lib = {mz: intensity for mz, intensity in proc_lib if mz <= max_mz}
    
    common_keys = sorted(set(dict_query.keys()) & set(dict_lib.keys()))
    overlap = len(common_keys)
    
    ratio_factors = []
    for i in range(1, len(common_keys)):
        mz_prev, mz_curr = common_keys[i-1], common_keys[i]
        if dict_query.get(mz_prev, 0) and dict_lib.get(mz_prev, 0):
            r_query = dict_query[mz_curr] / dict_query[mz_prev]
            r_lib = dict_lib[mz_curr] / dict_lib[mz_prev]
            rf = min(r_query, r_lib) / max(r_query, r_lib)
            ratio_factors.append(rf)
    avg_ratio_factor = np.mean(ratio_factors) if ratio_factors else 1

    N = len([mz for mz, intensity in proc_query if intensity > 0])
    
    composite_sim = (N * cosine_sim + overlap * avg_ratio_factor) / (N + overlap) if (N + overlap) > 0 else 0
    return composite_sim

# --- Comparing Spectra ---
def compare_spectra(query_spectrum, library_spectra, max_mz=1000, unmatched_method="keep_all",
                   weighting_scheme="None", similarity_measure="weighted_cosine", mz_shift=0):
    """
    Compare query spectrum with library spectra.
    
    Args:
        query_spectrum: List of (mz, intensity) tuples
        library_spectra: Dictionary of compound names to Compound objects
        max_mz: Maximum m/z value to consider
        unmatched_method: How to handle unmatched peaks
        weighting_scheme: Scheme for weighting m/z and intensity values
        similarity_measure: Which similarity function to use
        mz_shift: m/z shift to apply to query spectrum, NOT library spectra
        
    Returns:
        List of (compound_name, similarity) tuples sorted by similarity
    """
    # Apply mz_shift to query spectrum if needed
    if mz_shift:
        query_spectrum = [(mz + mz_shift, intensity) for mz, intensity in query_spectrum]
    
    # Preprocess the query spectrum
    proc_query = preprocess_spectrum(query_spectrum, weighting_scheme=weighting_scheme)

    scores = {}
    for dict_key, compound in library_spectra.items():
        if similarity_measure == "weighted_cosine":
            proc_lib = preprocess_spectrum(compound.spectrum, weighting_scheme=weighting_scheme)
            similarity = dot_product_similarity(proc_query, proc_lib, max_mz, unmatched_method)
        elif similarity_measure == "composite":
            similarity = composite_weighted_cosine_similarity(query_spectrum, compound.spectrum,
                                                              max_mz, unmatched_method, weighting_scheme)
        else:
            raise ValueError("Unknown similarity_measure option.")
        
        # Use compound.name instead of dictionary key to return original name
        scores[compound.name] = similarity
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)