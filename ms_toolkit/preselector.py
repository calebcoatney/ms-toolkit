import numpy as np
from sklearn.cluster import KMeans
from .preprocessing import spectrum_to_vector, preprocess_spectrum
from sklearn.mixture import GaussianMixture

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

    Attributes:
        gmm (GaussianMixture): The fitted GMM.
        labels (np.ndarray): Hard assignment of each library vector to a GMM component.
        library_keys (List[str]): Ordered keys/IDs corresponding to each vector.
    """

    def __init__(
        self,
        library_vectors: np.ndarray,
        library_keys: list,
        n_components: int = 200,
        covariance_type: str = "diag",
        max_iter: int = 200,
        random_state: int = 42
    ):
        """
        Fit the GMM on your library vectors.

        Args:
            library_vectors (np.ndarray): Array of shape (n_library, n_features);
                the vectorized spectra of your library.
            library_keys (List[str]): List of length n_library; the identifiers
                or keys of each library entry.
            n_components (int): Number of Gaussian components to learn.
            covariance_type (str): One of {'full', 'tied', 'diag', 'spherical'}.
            max_iter (int): Maximum EM iterations.
            random_state (int): Seed for reproducibility.
        """
        self.library_keys = library_keys
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state
        ).fit(library_vectors)
        # hard assign each library vector to its most likely component
        self.labels = self.gmm.predict(library_vectors)

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

        # 2) Compute per-component responsibilities (posterior probabilities)
        #    predict_proba returns shape (1, n_components)
        resp = self.gmm.predict_proba(q_vec.reshape(1, -1)).ravel()

        # 3) Find top-K components
        top_comps = np.argsort(resp)[::-1][:top_k_components]

        # 4) Gather library entries whose hard label ∈ top_comps
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