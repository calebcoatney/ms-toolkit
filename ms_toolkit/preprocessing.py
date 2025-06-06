import numpy as np

# --- Weighting Schemes ---
# Note: The weighting schemes apply exponents to m/z values (m/z weighting)
# and to intensities (intensity weighting) separately. For example, "NIST_GC" uses m/z^3 * I^0.6.
weighting_schemes = {
    "None": (0, 1),          # m/z^0 * I^1 (only intensity)
    "SQRT": (0, 0.5),        # weighting only by intensities (square root)
    "MassBank": (2, 0.5),    # m/z^2 * I^0.5
    "NIST11": (1.3, 0.53),   # m/z^1.3 * I^0.53 (LC)
    "NIST_GC": (3, 0.6)      # m/z^3 * I^0.6 (GC)
}

# --- Preprocessing Functions ---
def spectrum_to_vector(spectrum, max_mz=1000, bin_width=1.0):
    """
    Convert a spectrum (list of tuples) to a vector.
    
    Args:
        spectrum: List of (mz, intensity) tuples
        max_mz: Maximum m/z value to include
        bin_width: Width of m/z bins (default=1.0 for unit mass resolution)
        
    Returns:
        numpy.ndarray: Vector representation of the spectrum
    """
    # Calculate vector size based on bin width
    vector_size = int(max_mz / bin_width) + 1
    vector = np.zeros(vector_size)
    
    for mz, intensity in spectrum:
        if mz <= max_mz:
            # Convert m/z to appropriate bin index
            bin_idx = int(mz / bin_width)
            # Sum intensities that fall in the same bin
            vector[bin_idx] += intensity
    
    return vector

def vector_to_spectrum(mass_spectrum, shift=0):
    """
    Convert a vector to a spectrum (list of tuples).
    
    Args:
        mass_spectrum: numpy.ndarray vector representation
        shift: m/z shift to apply (useful for data from different sources)
        
    Returns:
        List of (mz, intensity) tuples for non-zero intensities
    """
    return [(index + 1 + shift, intensity) for index, intensity in enumerate(mass_spectrum) if intensity > 0]

def tic_scaling(spectrum):
    intensities = np.array([intensity for _, intensity in spectrum])
    total_ion_current = np.sum(intensities)
    return [(mz, intensity / total_ion_current) for mz, intensity in spectrum]

def mass_weighting(spectrum, a=0, b=1):
    """Apply weighting: each intensity is modified by m/z^a * I^b."""
    return [(mz, (mz ** a) * (intensity ** b)) for mz, intensity in spectrum]

def filter_low_intensity_peaks(spectrum, threshold_fraction=0.01):
    """
    Filters out peaks with intensity less than a specified fraction of the highest peak's intensity.
    
    Parameters:
        spectrum (List[Tuple[int, float]]): The input spectrum as a list of (m/z, intensity) tuples.
        threshold_fraction (float): The fraction threshold for filtering (default is 0.01, i.e., 1%).
    
    Returns:
        List[Tuple[int, float]]: The filtered spectrum.
    """
    max_intensity = max(intensity for _, intensity in spectrum)
    threshold = threshold_fraction * max_intensity
    return [(mz, intensity) for mz, intensity in spectrum if intensity >= threshold]

def preprocess_spectrum(spectrum, apply_tic_scaling=True, apply_mass_weighting=True,
                        weighting_scheme="None", intensity_threshold=0.01):
    """
    Preprocesses the spectrum by filtering low-intensity peaks, applying TIC scaling, 
    and optionally applying mass weighting.
    
    Parameters:
        spectrum (List[Tuple[int, float]]): The input spectrum as a list of (m/z, intensity) tuples.
        apply_tic_scaling (bool): Whether to apply TIC scaling (default is True).
        apply_mass_weighting (bool): Whether to apply mass weighting (default is True).
        weighting_scheme (str): The weighting scheme to use (default is "None").
        intensity_threshold (float): The fraction threshold for filtering low-intensity peaks (default is 0.01).
    
    Returns:
        List[Tuple[int, float]]: The preprocessed spectrum.
    """
    # Filter out low-intensity peaks
    spectrum = filter_low_intensity_peaks(spectrum, intensity_threshold)
    
    if apply_tic_scaling:
        spectrum = tic_scaling(spectrum)
    if apply_mass_weighting:
        a, b = weighting_schemes.get(weighting_scheme, (0, 1))
        spectrum = mass_weighting(spectrum, a, b)
    return spectrum

# --- Alignment for Handling Unmatched Signals ---
def align_spectra(spectrum1, spectrum2, max_mz=1000, unmatched_method="keep_all"):
    dict1 = {mz: intensity for mz, intensity in spectrum1 if mz <= max_mz}
    dict2 = {mz: intensity for mz, intensity in spectrum2 if mz <= max_mz}
    
    if unmatched_method == "keep_all":
        keys = list(range(max_mz + 1))
        vec1 = np.array([dict1.get(mz, 0) for mz in keys])
        vec2 = np.array([dict2.get(mz, 0) for mz in keys])
    elif unmatched_method == "remove_all":
        keys = sorted(set(dict1.keys()) & set(dict2.keys()))
        vec1 = np.array([dict1[k] for k in keys])
        vec2 = np.array([dict2[k] for k in keys])
    elif unmatched_method == "keep_library":
        keys = sorted(dict2.keys())
        vec1 = np.array([dict1.get(k, 0) for k in keys])
        vec2 = np.array([dict2[k] for k in keys])
    elif unmatched_method == "keep_experimental":
        keys = sorted(dict1.keys())
        vec1 = np.array([dict1[k] for k in keys])
        vec2 = np.array([dict2.get(k, 0) for k in keys])
    else:
        raise ValueError("Unknown unmatched_method option.")
    return vec1, vec2