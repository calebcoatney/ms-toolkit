'''
ms_toolkit/
├── ms_toolkit/
│   ├── __init__.py
│   ├── utils.py            # CAS formatting, general helpers
│   ├── models.py           # Compound, SpectrumDocument
│   ├── preprocessing.py    # filter, scaling, vectorization, alignment
│   ├── preselector.py      # ClusterPreselector
│   ├── w2v.py              # train_model, embed_spectrum, search_w2v
│   └── api.py              # high-level Pipeline / facade
'''

import os
import joblib
import numpy as np

from .io import parse
from .preprocessing import spectrum_to_vector, vector_to_spectrum
from .w2v import train_model, load_model, calc_embedding
from .preselector import ClusterPreselector, GMMPreselector
from .similarity import compare_spectra
from .models import SpectrumDocument

class MSToolkit:
    """
    Facade for:
      - library loading & caching
      - vectorization
      - word2vec training/loading
      - k-means preselector
      - high-level search (vector or embedding)
    """
    def __init__(
        self,
        library_txt: str = None,
        cache_json: str = None,
        w2v_path: str = None,
        preselector_path: str = None,
        vector_max_mz: int = 1000,
        n_clusters: int = 100,
        mz_shift: int = 0,  # Add this parameter
        show_ui: bool = True,  # New parameter for UI control
        ui_framework: str = 'pyside6',  # Default to PySide6 for better compatibility
        progress_callback: callable = None  # Allow custom progress tracking
    ):
        """
        Initialize MSToolkit with paths and parameters.
        
        Args:
            library_txt (str, optional): Path to library file. Defaults to None.
            cache_json (str, optional): Path to cache file. Defaults to "library.json".
            w2v_path (str, optional): Default path for Word2Vec model. Defaults to "w2v.model".
            preselector_path (str, optional): Default path for preselector. Defaults to "preselector.pkl".
            vector_max_mz (int, optional): Maximum m/z for vectorization. Defaults to 1000.
            n_clusters (int, optional): Default number of clusters. Defaults to 100.
            mz_shift (int, optional): m/z shift to apply when converting between vectors and spectra. Defaults to 0.
            show_ui (bool, optional): Whether to show a progress UI when loading libraries. Defaults to True.
            ui_framework (str, optional): UI framework to use ('ctk' or 'pyside6'). Defaults to 'pyside6'.
            progress_callback (callable, optional): Custom progress callback function. Defaults to None.
        """
        self.library_txt = library_txt
        self.cache_json = cache_json or "library.json"
        self.w2v_path = w2v_path or "w2v.model"
        self.preselector_path = preselector_path or "preselector.pkl"
        self.max_mz = vector_max_mz
        self.n_clusters = n_clusters
        self.mz_shift = mz_shift  # Store the m/z shift
        
        # UI-related options
        self.show_ui = show_ui
        self.ui_framework = ui_framework
        self.progress_callback = progress_callback

        self.library = None
        self.vectors = None
        self.w2v_model = None
        self.preselector = None

    def load_library(
        self, 
        file_path: str = None, 
        json_path: str = None, 
        subset: str = None,
        show_ui: bool = None,
        ui_framework: str = None,
        progress_callback: callable = None,
        save_path: str = None
    ):
        """
        Load library directly from JSON cache or parse from text file.
        
        Args:
            file_path (str, optional): Path to library text file. Defaults to self.library_txt.
            json_path (str, optional): Path to JSON cache file. Defaults to self.cache_json.
            subset (str, optional): Subset of elements to include. Defaults to None.
            show_ui (bool, optional): Whether to show a progress UI. Defaults to value set in __init__.
            ui_framework (str, optional): UI framework to use ('ctk' or 'pyside6'). Defaults to value set in __init__.
            progress_callback (callable, optional): Custom progress callback function. Defaults to value set in __init__.
            save_path (str, optional): Path to save filtered subset for faster future loading. Defaults to None.
            
        Returns:
            The loaded library
            
        Raises:
            ValueError: If both file_path and json_path are None and no default values exist
            FileNotFoundError: If specified files don't exist
        """
        # Use parameters from method call if provided, else fall back to instance variables
        show_ui = show_ui if show_ui is not None else self.show_ui
        ui_framework = ui_framework or self.ui_framework
        progress_callback = progress_callback or self.progress_callback
        
        if self.library is None:
            text_path = file_path or self.library_txt
            cache_path = json_path or self.cache_json
            
            # Check if we have at least one valid path
            if not (text_path or cache_path):
                raise ValueError("Either a library text file path or a JSON cache path must be provided")
            
            # Try to load from JSON first if a cache path exists
            if cache_path and os.path.exists(cache_path):
                try:
                    # Set file_path to None to indicate we just want to load from cache
                    self.library = parse(
                        file_path=None,
                        load_cache=True,
                        cache_file=cache_path,
                        subset=subset,
                        show_ui=show_ui,
                        ui_framework=ui_framework,
                        progress_callback=progress_callback,
                        save_path=save_path
                    )
                    return self.library
                except Exception as e:
                    if not text_path:
                        raise ValueError(f"JSON cache loading failed: {str(e)}. Please provide a valid text file path.") from e
                    # If JSON loading fails but we have a text path, continue to text loading
            
            # If we reach here, either the JSON file doesn't exist, loading failed, or json_path wasn't provided
            # So try loading from text file if we have a path
            if text_path:
                if not os.path.exists(text_path):
                    raise FileNotFoundError(f"Library text file not found: {text_path}")
                
                self.library = parse(
                    file_path=text_path,
                    load_cache=True,
                    cache_file=cache_path,
                    subset=subset,
                    show_ui=show_ui,
                    ui_framework=ui_framework,
                    progress_callback=progress_callback,
                    save_path=save_path
                )
            else:
                # No text file path but JSON path was provided and failed to load
                raise ValueError("JSON cache loading failed and no text file path was provided")
        
        return self.library

    def vectorize_library(self):
        """
        Create full-spectrum vectors for clustering/search.
        """
        if self.library is None:
            raise RuntimeError("Library must be loaded first")
        self.vectors = {
            name: spectrum_to_vector(comp.spectrum, max_mz=self.max_mz)
            for name, comp in self.library.items()
        }
        return self.vectors

    def load_w2v(self, file_path=None, save_path=None):
        """
        Load Word2Vec model from file with option to save a copy elsewhere.
        
        Args:
            file_path (str, optional): Path to model file. Defaults to self.w2v_path.
            save_path (str, optional): Path to save a copy of the loaded model. Defaults to None.
            
        Returns:
            The loaded Word2Vec model
        """
        path = file_path or self.w2v_path
        if not path:
            raise ValueError("Model file path must be provided either during initialization or in the load_w2v call")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.w2v_model = load_model(path)
        
        # If save_path is provided, save a copy of the model
        if save_path and save_path != path:
            self.w2v_model.save(save_path)
        
        return self.w2v_model

    def train_w2v(self, save_path=None, vector_size=300, window=500, epochs=5, workers=16):
        """
        Train a new Word2Vec model on the library.
        
        Args:
            save_path (str, optional): Path to save the trained model. Defaults to self.w2v_path.
            vector_size (int): Dimensionality of the embedding vectors.
            window (int): Maximum distance between peaks for consideration.
            epochs (int): Number of training epochs.
            workers (int): Number of worker threads.
        
        Returns:
            The trained Word2Vec model
        """
        if self.library is None:
            raise RuntimeError("Library must be loaded first")
            
        path_to_save = save_path or self.w2v_path
        
        # Train new model
        self.w2v_model = train_model(
            library=self.library,
            file_path=path_to_save,
            vector_size=vector_size,
            window=window,
            epochs=epochs,
            workers=workers
        )
        return self.w2v_model

    def load_preselector(self, file_path=None, save_path=None):
        """
        Load preselector model from file with option to save a copy elsewhere.
        
        Args:
            file_path (str, optional): Path to model file. Defaults to self.preselector_path.
            save_path (str, optional): Path to save a copy of the loaded model. Defaults to None.
            
        Returns:
            The loaded preselector model
        """
        path = file_path or self.preselector_path
        if not path:
            raise ValueError("Preselector file path must be provided either during initialization or in the load_preselector call")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preselector file not found: {path}")
            
        with open(path, 'rb') as f:
            self.preselector = joblib.load(f)
        
        # If save_path is provided, save a copy of the model
        if save_path and save_path != path:
            with open(save_path, 'wb') as f:
                joblib.dump(self.preselector, f)
        
        return self.preselector

    def train_preselector(
        self, 
        save_path=None, 
        preselector_type="kmeans", 
        n_clusters=None, 
        n_components=None,
        covariance_type="diag", 
        max_iter=200, 
        random_state=42
    ):
        """
        Train a new preselector model on the library vectors.
        
        Args:
            save_path (str, optional): Path to save the trained model. Defaults to self.preselector_path.
            preselector_type (str): Type of preselector to train: "kmeans" or "gmm". Defaults to "kmeans".
            n_clusters (int, optional): Number of clusters for KMeans. Defaults to self.n_clusters.
            n_components (int, optional): Number of components for GMM. Defaults to 200.
            covariance_type (str): Covariance type for GMM. Defaults to "diag".
            max_iter (int): Maximum iterations for GMM. Defaults to 200.
            random_state (int): Random seed. Defaults to 42.
            
        Returns:
            The trained preselector model
        """
        if self.vectors is None:
            raise RuntimeError("Library must be vectorized first")
            
        path_to_save = save_path or self.preselector_path
        
        # Stack vectors into a 2D array for clustering
        mat = np.vstack(list(self.vectors.values()))
        library_keys = list(self.vectors.keys())
        
        # Train new model based on type
        if preselector_type.lower() == "kmeans":
            clusters = n_clusters or self.n_clusters
            self.preselector = ClusterPreselector(
                library_vectors=mat, 
                library_keys=library_keys,
                n_clusters=clusters,
                random_state=random_state
            )
        elif preselector_type.lower() == "gmm":
            components = n_components or 200
            self.preselector = GMMPreselector(
                library_vectors=mat,
                library_keys=library_keys,
                n_components=components,
                covariance_type=covariance_type,
                max_iter=max_iter,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown preselector type: {preselector_type}. Use 'kmeans' or 'gmm'.")
        
        # Save model
        with open(path_to_save, 'wb') as f:
            joblib.dump(self.preselector, f)
        
        return self.preselector

    def search_vector(
        self, 
        query_input, 
        top_n=10, 
        weighting_scheme="None", 
        composite=False, 
        unmatched_method="keep_all",
        top_k_clusters=1  # Add parameter to control number of clusters/components
    ):
        """
        Preselector + (composite_)cosine similarity in vector space.
        
        Args:
            query_input: Either a spectrum (list of tuples) or a vector (numpy array)
            top_n: Number of top results to return
            weighting_scheme: Weighting scheme to use for similarity calculations
            composite: Whether to use composite similarity measure
            unmatched_method: How to handle unmatched peaks during spectral alignment.
                              Options: "keep_all", "remove_all", "keep_library", "keep_experimental"
            top_k_clusters: Number of clusters/components to consider (for KMeans/GMM)
            
        Returns:
            List of (compound_name, similarity_score) tuples
        """
        if self.preselector is None:
            raise RuntimeError("Preselector must be loaded first")
            
        # Convert input to appropriate formats based on type
        if isinstance(query_input, np.ndarray):
            # If input is a vector, ensure it has the right dimensions
            if len(query_input) != (self.max_mz + 1):
                # Convert to spectrum (applying mz_shift) and back to vector (with correct dimensions)
                query_spectrum = vector_to_spectrum(query_input, shift=self.mz_shift)
                query_vector = spectrum_to_vector(query_spectrum, max_mz=self.max_mz)
            else:
                query_vector = query_input
                # If vector has correct dimensions, still need spectrum for later
                query_spectrum = vector_to_spectrum(query_input, shift=self.mz_shift) 
        else:
            # If input is a spectrum, apply mz_shift and convert to vector
            query_spectrum = [(mz + self.mz_shift, intensity) for mz, intensity in query_input]
            query_vector = spectrum_to_vector(query_spectrum, max_mz=self.max_mz)
        
        # Now we're guaranteed to have both a vector with correct dimensions and a spectrum
        
        # Handle different preselector types
        if isinstance(self.preselector, ClusterPreselector):
            selected_keys = self.preselector.select(
                query_vector, 
                list(self.library.keys()),
                top_k_clusters=top_k_clusters
            )
        elif isinstance(self.preselector, GMMPreselector):
            selected_keys = self.preselector.select(
                query_vector, 
                list(self.library.keys()),
                top_k_components=top_k_clusters
            )
        else:
            # Backward compatibility with older models
            selected_keys = self.preselector.select(query_vector, list(self.library.keys()))
        
        subset = {k: self.library[k] for k in selected_keys if k in self.library}

        similarity_measure = "composite" if composite else "weighted_cosine"
        results = compare_spectra(
            query_spectrum, 
            subset, 
            max_mz=self.max_mz,
            weighting_scheme=weighting_scheme, 
            similarity_measure=similarity_measure,
            unmatched_method=unmatched_method
        )
        return results[:top_n]

    def search_w2v(self, query_input, top_n=10, intensity_power=0.6, top_k_clusters=1):
        """
        Preselector + Word2Vec embedding + cosine similarity.

        Args:
            query_input: Either a spectrum (list of tuples) or a vector (numpy array)
            top_n: Number of top results to return
            intensity_power: Exponent for intensity weighting
            top_k_clusters: Number of clusters/components to consider (for KMeans/GMM)
            
        Returns:
            List of (compound_name, similarity_score) tuples
        """
        if self.w2v_model is None:
            raise RuntimeError("Word2Vec model must be loaded first")
        if self.preselector is None:
            raise RuntimeError("Preselector must be loaded first")
        
        # Convert input to appropriate formats based on type
        if isinstance(query_input, np.ndarray):
            # If input is a vector, ensure it has the right dimensions
            if len(query_input) != (self.max_mz + 1):
                # Convert to spectrum (applying mz_shift) and back to vector (with correct dimensions)
                query_spectrum = vector_to_spectrum(query_input, shift=self.mz_shift)
                query_vector = spectrum_to_vector(query_spectrum, max_mz=self.max_mz)
            else:
                query_vector = query_input
                # If vector has correct dimensions, still need spectrum for later
                query_spectrum = vector_to_spectrum(query_input, shift=self.mz_shift) 
        else:
            # If input is a spectrum, apply mz_shift and convert to vector
            query_spectrum = [(mz + self.mz_shift, intensity) for mz, intensity in query_input]
            query_vector = spectrum_to_vector(query_spectrum, max_mz=self.max_mz)
        
        # Now we're guaranteed to have both a vector with correct dimensions and a spectrum
        
        # Handle different preselector types
        if isinstance(self.preselector, ClusterPreselector):
            selected_keys = self.preselector.select(
                query_vector, 
                list(self.library.keys()),
                top_k_clusters=top_k_clusters
            )
        elif isinstance(self.preselector, GMMPreselector):
            selected_keys = self.preselector.select(
                query_vector, 
                list(self.library.keys()),
                top_k_components=top_k_clusters
            )
        else:
            # Backward compatibility with older models
            selected_keys = self.preselector.select(query_vector, list(self.library.keys()))
        
        query_doc = SpectrumDocument(query_spectrum, n_decimals=2)
        query_embedding = calc_embedding(self.w2v_model, query_doc, intensity_power)

        embeddings = {name: calc_embedding(self.w2v_model, SpectrumDocument(self.library[name].spectrum, n_decimals=2), intensity_power)
                      for name in selected_keys if name in self.library}

        similarities = {name: float(np.dot(query_embedding, vec) / (np.linalg.norm(query_embedding) * np.linalg.norm(vec)))
                        for name, vec in embeddings.items()}
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
