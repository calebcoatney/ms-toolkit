import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from .preprocessing import spectrum_to_vector, preprocess_spectrum

# --- Optimization Utilities ---
def estimate_optimal_pca_components(vectors, variance_thresholds=[0.90, 0.95, 0.99]):
    """
    Estimate the number of PCA components needed to retain different levels of variance.
    
    Args:
        vectors (np.ndarray): Library vectors
        variance_thresholds (list): List of variance retention thresholds to test
        
    Returns:
        dict: Mapping of threshold to number of components needed
    """
    print(f"Analyzing PCA requirements for {vectors.shape[0]} vectors with {vectors.shape[1]} features...")
    
    # Fit PCA with maximum components
    pca = PCA()
    pca.fit(vectors)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    results = {}
    for threshold in variance_thresholds:
        n_components = np.argmax(cumsum >= threshold) + 1
        results[threshold] = n_components
        print(f"For {threshold*100:.0f}% variance: {n_components} components "
              f"(reduction: {vectors.shape[1]} → {n_components}, "
              f"{n_components/vectors.shape[1]*100:.1f}% of original)")
    
    return results

def estimate_optimal_gmm_components(vectors, n_samples=10000, k_range=[50, 100, 200, 500, 1000]):
    """
    Estimate optimal number of GMM components using BIC/AIC on a sample.
    
    Args:
        vectors (np.ndarray): Library vectors
        n_samples (int): Number of samples to use for estimation
        k_range (list): Range of component numbers to test
        
    Returns:
        dict: BIC and AIC scores for different component numbers
    """
    print(f"Estimating optimal GMM components using {n_samples} samples...")
    
    # Sample subset for faster estimation
    if len(vectors) > n_samples:
        sample_indices = np.random.choice(len(vectors), n_samples, replace=False)
        sample_vectors = vectors[sample_indices]
    else:
        sample_vectors = vectors
    
    results = {}
    
    for k in k_range:
        if k > len(sample_vectors):
            continue
            
        print(f"Testing {k} components...")
        
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='diag',
                max_iter=100,  # Fewer iterations for estimation
                random_state=42
            )
            gmm.fit(sample_vectors)
            
            bic = gmm.bic(sample_vectors)
            aic = gmm.aic(sample_vectors)
            
            results[k] = {'bic': bic, 'aic': aic}
            print(f"  BIC: {bic:.2f}, AIC: {aic:.2f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[k] = {'bic': np.inf, 'aic': np.inf}
    
    # Find optimal
    valid_results = {k: v for k, v in results.items() if v['bic'] != np.inf}
    if valid_results:
        best_bic = min(valid_results.keys(), key=lambda k: valid_results[k]['bic'])
        best_aic = min(valid_results.keys(), key=lambda k: valid_results[k]['aic'])
        
        print(f"\nOptimal by BIC: {best_bic} components")
        print(f"Optimal by AIC: {best_aic} components")
    
    return results

def analyze_dataset_characteristics(vectors):
    """
    Analyze characteristics of the vector dataset to suggest optimization strategies.
    
    Args:
        vectors (np.ndarray): Library vectors
        
    Returns:
        dict: Analysis results with recommendations
    """
    print(f"Analyzing dataset characteristics...")
    print(f"Dataset shape: {vectors.shape}")
    
    # Basic statistics
    sparsity = np.mean(vectors == 0)
    mean_norm = np.mean(np.linalg.norm(vectors, axis=1))
    std_norm = np.std(np.linalg.norm(vectors, axis=1))
    
    # Memory estimation
    memory_mb = vectors.nbytes / (1024 * 1024)
    
    print(f"Sparsity: {sparsity:.3f} ({sparsity*100:.1f}% zeros)")
    print(f"Vector norms: mean={mean_norm:.2f}, std={std_norm:.2f}")
    print(f"Memory usage: {memory_mb:.1f} MB")
    
    # Recommendations
    recommendations = []
    
    if vectors.shape[1] > 500:
        recommendations.append("Use PCA: High dimensionality detected")
    
    if vectors.shape[0] > 100000:
        recommendations.append("Use k-means initialization: Large dataset detected")
        
    if sparsity > 0.8:
        recommendations.append("Consider sparse representations or feature selection")
        
    if memory_mb > 1000:
        recommendations.append("Consider batch processing or data streaming")
    
    # Suggested parameters
    suggested_pca_components = min(200, vectors.shape[1] // 5)
    suggested_gmm_components = min(500, vectors.shape[0] // 1000)
    
    results = {
        'shape': vectors.shape,
        'sparsity': sparsity,
        'memory_mb': memory_mb,
        'recommendations': recommendations,
        'suggested_pca_components': suggested_pca_components,
        'suggested_gmm_components': suggested_gmm_components
    }
    
    print(f"\nSuggested parameters:")
    print(f"  PCA components: {suggested_pca_components}")
    print(f"  GMM components: {suggested_gmm_components}")
    print(f"\nRecommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    
    return results

# --- Clustering Preselection ---
class ClusterPreselector:
    def __init__(self, library_vectors, library_keys=None, n_clusters=100, random_state=42):
        """
        Initialize clustering-based preselector.
        
        Args:
            library_vectors: Spectral vectors for clustering.
            library_keys: List of keys corresponding to library_vectors. If None, integers will be used.
            n_clusters: Number of clusters.
            random_state: Random seed for KMeans.
        """
        self.n_clusters = n_clusters
        self.library_keys = library_keys
        self.cluster_model = KMeans(
            n_clusters=self.n_clusters, random_state=random_state)
        self.labels = self.cluster_model.fit_predict(library_vectors)

    def select(self, query_input, library, top_k_clusters=1, max_mz=1000, mz_shift=0):
        """
        Find clusters closest to the query and return their members.
        
        Args:
            query_input: Either a spectrum (list of tuples) or a vector (numpy array)
            library: List of keys or dictionary of library compounds
            top_k_clusters: Number of closest clusters to consider
            max_mz: Maximum m/z value (determines vector length)
            mz_shift: m/z shift to apply when converting between vectors and spectra
            
        Returns:
            List of selected library keys
        """
        # Check if input is already a vector
        if isinstance(query_input, np.ndarray):
            query_vector = query_input
        else:
            # Apply mz_shift to the query spectrum
            processed_spectrum = [(mz + mz_shift, intensity) for mz, intensity in query_input]
            processed_spectrum = preprocess_spectrum(processed_spectrum)
            query_vector = spectrum_to_vector(processed_spectrum, max_mz=max_mz)
        
        # Get distances to all cluster centers
        distances = self.cluster_model.transform([query_vector])[0]
        
        # Get top-k closest clusters
        top_clusters = np.argsort(distances)[:top_k_clusters]
        
        # Get keys (either from library or stored keys)
        keys = (
            list(library.keys())
            if isinstance(library, dict)
            else (library if len(library) == len(self.labels) else self.library_keys)
        )
        
        # Get all library entries belonging to the top clusters
        selected = [
            key for idx, key in enumerate(keys)
            if idx < len(self.labels) and self.labels[idx] in top_clusters
        ]
        
        return selected
    

class GMMPreselector:
    """
    Pre-select candidate spectra using a Gaussian Mixture Model (GMM).

    This model fits a GMM to your library of spectral vectors; at query time
    it computes the posterior responsibility of each mixture component for
    the query, takes the top `top_k_components`, and returns all library
    entries assigned to those components.

    Optimizations for large datasets:
    - Optional PCA dimensionality reduction
    - K-means initialization for GMM
    - Batch processing for very large datasets

    Attributes:
        gmm (GaussianMixture): The fitted GMM.
        labels (np.ndarray): Hard assignment of each library vector to a GMM component.
        library_keys (List[str]): Ordered keys/IDs corresponding to each vector.
        pca (PCA, optional): PCA transformer if dimensionality reduction was used.
        use_pca (bool): Whether PCA was applied.
    """

    def __init__(
        self,
        library_vectors: np.ndarray,
        library_keys: list,
        n_components: int = 200,
        covariance_type: str = "diag",
        max_iter: int = 200,
        random_state: int = 42,
        use_pca: bool = True,
        n_pca_components: int = None,
        pca_variance_threshold: float = 0.95,
        kmeans_init: bool = True,
        verbose: bool = True
    ):
        """
        Fit the GMM on your library vectors with optional optimizations.

        Args:
            library_vectors (np.ndarray): Array of shape (n_library, n_features);
                the vectorized spectra of your library.
            library_keys (List[str]): List of length n_library; the identifiers
                or keys of each library entry.
            n_components (int): Number of Gaussian components to learn.
            covariance_type (str): One of {'full', 'tied', 'diag', 'spherical'}.
            max_iter (int): Maximum EM iterations.
            random_state (int): Seed for reproducibility.
            use_pca (bool): Whether to apply PCA for dimensionality reduction.
            n_pca_components (int, optional): Number of PCA components. If None, 
                determined by variance_threshold.
            pca_variance_threshold (float): Fraction of variance to retain with PCA.
            kmeans_init (bool): Whether to use k-means for GMM initialization.
            verbose (bool): Whether to print progress information.
        """
        self.library_keys = library_keys
        self.use_pca = use_pca
        self.pca = None
        self.verbose = verbose
        
        if self.verbose:
            print(f"Training GMM preselector on {library_vectors.shape[0]} spectra...")
            print(f"Original feature dimension: {library_vectors.shape[1]}")
        
        # Apply PCA if requested
        processed_vectors = library_vectors
        if use_pca:
            if self.verbose:
                print("Applying PCA for dimensionality reduction...")
            
            self.pca = PCA(
                n_components=n_pca_components,
                random_state=random_state
            )
            processed_vectors = self.pca.fit_transform(library_vectors)
            
            # Determine actual number of components based on variance threshold
            if n_pca_components is None:
                cumsum = np.cumsum(self.pca.explained_variance_ratio_)
                n_components_actual = np.argmax(cumsum >= pca_variance_threshold) + 1
                self.pca.n_components = n_components_actual
                processed_vectors = processed_vectors[:, :n_components_actual]
            
            if self.verbose:
                print(f"PCA reduced dimension to: {processed_vectors.shape[1]}")
                if hasattr(self.pca, 'explained_variance_ratio_'):
                    variance_retained = np.sum(self.pca.explained_variance_ratio_[:processed_vectors.shape[1]])
                    print(f"Variance retained: {variance_retained:.3f}")
        
        # Initialize GMM
        gmm_kwargs = {
            'n_components': n_components,
            'covariance_type': covariance_type,
            'max_iter': max_iter,
            'random_state': random_state,
            'verbose': 1 if verbose else 0
        }
        
        # Use k-means initialization if requested
        if kmeans_init:
            if self.verbose:
                print("Using k-means initialization for GMM...")
            gmm_kwargs['init_params'] = 'k-means++'
        
        self.gmm = GaussianMixture(**gmm_kwargs)
        
        # Fit the GMM
        if self.verbose:
            print("Fitting GMM...")
        self.gmm.fit(processed_vectors)
        
        # Hard assign each library vector to its most likely component
        self.labels = self.gmm.predict(processed_vectors)
        
        if self.verbose:
            print("GMM training completed!")
            # Print some statistics
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            print(f"Component usage: min={counts.min()}, max={counts.max()}, "
                  f"mean={counts.mean():.1f}, std={counts.std():.1f}")

    def select(
        self,
        query_input,
        library,
        top_k_components: int = 3,
        max_mz: int = 1000,
        mz_shift: float = 0.0
    ) -> list:
        """
        Select a subset of library keys based on the query spectrum.

        Args:
            query_input (Union[np.ndarray, List[Tuple[float, float]]]):
                - If numpy array: assumed already vectorized (shape (n_features,)).
                - Else: list of (m/z, intensity) tuples to be preprocessed.
            library (Union[Dict[str, Any], List[str]]):
                Same indexing as `library_keys`; can be a dict or list.
                This is used only to determine ordering if library_keys
                were not passed explicitly.
            top_k_components (int): Number of GMM components to gather.
            max_mz (int): Maximum m/z for vectorization (must match training).
            mz_shift (float): m/z shift to apply before preprocessing.

        Returns:
            List[str]: The keys/IDs of all library spectra whose GMM-component
            label is among the top_k_components for this query.
        """
        # 1) Convert input → vector
        if isinstance(query_input, np.ndarray):
            q_vec = query_input
        else:
            # shift, preprocess, then vectorize
            shifted = [(mz + mz_shift, inten) for mz, inten in query_input]
            proc = preprocess_spectrum(shifted)
            q_vec = spectrum_to_vector(proc, max_mz=max_mz)

        # 2) Apply PCA transformation if it was used during training
        if self.use_pca and self.pca is not None:
            q_vec_pca = self.pca.transform(q_vec.reshape(1, -1))
            # Use only the number of components that were actually used during training
            if hasattr(self.pca, 'n_components') and self.pca.n_components is not None:
                q_vec = q_vec_pca[:, :self.pca.n_components].ravel()
            else:
                q_vec = q_vec_pca.ravel()

        # 3) Compute per-component responsibilities (posterior probabilities)
        #    predict_proba returns shape (1, n_components)
        resp = self.gmm.predict_proba(q_vec.reshape(1, -1)).ravel()

        # 4) Find top-K components
        top_comps = np.argsort(resp)[::-1][:top_k_components]

        # 5) Gather library entries whose hard label ∈ top_comps
        #    If user passed a dict, respect its key order; else use library_keys
        keys = (
            list(library.keys())
            if isinstance(library, dict)
            else (library if len(library) == len(self.labels) else self.library_keys)
        )
        selected = [
            key for idx, key in enumerate(keys)
            if idx < len(self.labels) and self.labels[idx] in top_comps
        ]
        return selected