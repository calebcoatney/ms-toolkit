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
        ui_framework: str = 'tqdm',
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
        self.cache_json = cache_json
        self.w2v_path = w2v_path
        self.preselector_path = preselector_path
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
        save_path: str = None,
        quiet: bool = None  # Add quiet parameter
    ):
        """
        Load library directly from JSON cache or parse from text file,
        then automatically vectorize the library.
        
        Args:
            file_path (str, optional): Path to library text file. Defaults to self.library_txt.
            json_path (str, optional): Path to JSON cache file. Defaults to self.cache_json.
            subset (str, optional): Subset of elements to include. Defaults to None.
            show_ui (bool, optional): Whether to show a progress UI. Defaults to value set in __init__.
            ui_framework (str, optional): UI framework to use ('tqdm', 'ctk' or 'pyside6'). Defaults to value set in __init__.
            progress_callback (callable, optional): Custom progress callback function. Defaults to value set in __init__.
            save_path (str, optional): Path to save filtered subset for faster future loading. Defaults to None.
            quiet (bool, optional): Suppress informational messages. Defaults to True when using tqdm.
            
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
        
        # Default quiet to True when using tqdm to prevent print messages from breaking the progress bar
        if quiet is None:
            quiet = (ui_framework == 'tqdm')
        
        # If show_ui is True but no progress_callback is defined, we need to create a tqdm progress bar here
        progress_bar = None
        if show_ui and not progress_callback and ui_framework == 'tqdm':
            try:
                from tqdm import tqdm
                # Create a cleaner progress bar without timing stats
                progress_bar = tqdm(total=100, desc="Loading library", unit="%", 
                               bar_format="{desc}: {percentage:3.0f}%|{bar}|")
                
                # Create a progress callback that updates the tqdm bar
                def tqdm_callback(value):
                    current = int(value * 100)
                    last = progress_bar.n
                    if current > last:
                        progress_bar.update(current - last)
                    
                    # Change description at 90% to indicate vectorization stage
                    if value >= 0.9 and progress_bar.desc == "Loading library":
                        progress_bar.set_description("Vectorizing library")
                    
                    if value >= 1.0:
                        progress_bar.close()
                        print("Library successfully loaded and vectorized.")
                        
                progress_callback = tqdm_callback
                self.progress_callback = tqdm_callback  # Store for future use
                
            except ImportError:
                print("Warning: tqdm not found. Install with 'pip install tqdm' to see progress bars.")
                show_ui = False
    
        # Create a scaled callback to report only 90% for library loading
        original_callback = progress_callback
        def scaled_progress_callback(value):
            if original_callback:
                # Scale value from 0-1 to 0-0.9 (90% of total)
                scaled_value = value * 0.9
                original_callback(scaled_value)
        
        # Flag to track if we need to vectorize (only if library was loaded)
        library_was_loaded = False
        
        if self.library is None:
            library_was_loaded = True
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
                        show_ui=False,  # Don't show UI in parse since we have our own progress bar
                        ui_framework=ui_framework,
                        progress_callback=scaled_progress_callback,
                        save_path=save_path,
                        quiet=quiet  # Pass quiet flag to suppress print messages
                    )
                    # Now at 90% complete
                except Exception as e:
                    if not text_path:
                        raise ValueError(f"JSON cache loading failed: {str(e)}. Please provide a valid text file path.") from e
                    # If JSON loading fails but we have a text path, continue to text loading
        
        # If we reach here, either the JSON file doesn't exist, loading failed, or json_path wasn't provided
        # So try loading from text file if we have a path
        if not self.library and text_path:
            if not os.path.exists(text_path):
                raise FileNotFoundError(f"Library text file not found: {text_path}")
            
            self.library = parse(
                file_path=text_path,
                load_cache=True,
                cache_file=cache_path,
                subset=subset,
                show_ui=False,  # Don't show UI in parse since we have our own progress bar
                ui_framework=ui_framework,
                progress_callback=scaled_progress_callback,
                save_path=save_path,
                quiet=quiet  # Pass quiet flag to suppress print messages
            )
            # Now at 90% complete
        elif not self.library:
            # No text file path but JSON path was provided and failed to load
            raise ValueError("JSON cache loading failed and no text file path was provided")
    
        # Automatically vectorize the library if it was just loaded
        if library_was_loaded:
            if original_callback:
                # No need to print "Vectorizing library..." since our progress bar already shows this
                self._vectorize_library_with_progress(progress_callback=original_callback, start_progress=0.9)
            else:
                # If no callback, just vectorize without progress reporting
                print("Vectorizing library...")
                self.vectorize_library()
                print("Library successfully loaded and vectorized.")
        
        return self.library

    def _vectorize_library_with_progress(self, bin_width=1.0, progress_callback=None, start_progress=0.9):
        """
        Create full-spectrum vectors for clustering/search with progress reporting.
        
        Args:
            bin_width: Width of m/z bins (default=1.0 for unit mass resolution)
            progress_callback: Function to call with progress updates
            start_progress: Starting progress value (default=0.9 or 90%)
        """
        if self.library is None:
            raise RuntimeError("Library must be loaded first")
            
        # Calculate how much progress each item represents
        total_items = len(self.library)
        progress_increment = 0.1 / total_items  # Final 10% divided by number of items
        
        self.vectors = {}
        
        # Process each compound with progress updates
        for i, (name, comp) in enumerate(self.library.items()):
            self.vectors[name] = spectrum_to_vector(comp.spectrum, max_mz=self.max_mz, bin_width=bin_width)
            
            # Report progress if callback provided
            if progress_callback and i % max(1, total_items // 100) == 0:  # Update progress ~100 times
                current_progress = start_progress + (i + 1) * progress_increment
                # Ensure we don't exceed 100%
                current_progress = min(1.0, current_progress)
                progress_callback(current_progress)
        
        # Ensure we report 100% when done
        if progress_callback:
            progress_callback(1.0)
            
        return self.vectors

    def vectorize_library(self, bin_width=1.0):
        """
        Create full-spectrum vectors for clustering/search.
        
        Args:
            bin_width: Width of m/z bins (default=1.0 for unit mass resolution)
        """
        if self.library is None:
            raise RuntimeError("Library must be loaded first")
        self.vectors = {
            name: spectrum_to_vector(comp.spectrum, max_mz=self.max_mz, bin_width=bin_width)
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

    def train_w2v(self, save_path=None, vector_size=300, window=500, epochs=5, workers=16, n_decimals=2, show_progress=True):
        """
        Train a new Word2Vec model on the library.
        
        Args:
            save_path (str, optional): Path to save the trained model. Defaults to self.w2v_path.
            vector_size (int): Dimensionality of the embedding vectors.
            window (int): Maximum distance between peaks for consideration.
            epochs (int): Number of training epochs.
            workers (int): Number of worker threads.
            n_decimals (int): Number of decimals to use for m/z values. Defaults to 2.
            show_progress (bool): Whether to display a progress bar during training. Defaults to True.

        Returns:
            The trained Word2Vec model
        """
        if self.library is None:
            raise RuntimeError("Library must be loaded first")
        
        # Use provided path, then self.w2v_path, or generate a temporary path
        path_to_save = save_path or self.w2v_path
        
        # If we still don't have a path, create a unique temporary file
        if not path_to_save:
            # Create a unique identifier based on time, library size, and parameters
            import time
            import hashlib
            import random
            
            # Create directory if it doesn't exist
            import os
            os.makedirs("cache", exist_ok=True)
            
            # Create a unique identifier
            timestamp = int(time.time())
            lib_hash = hashlib.md5(f"{len(self.library)}_{vector_size}_{window}_{epochs}".encode()).hexdigest()[:8]
            random_suffix = ''.join(random.choices('0123456789abcdef', k=4))
            
            # Format: cache/w2v_temp_<timestamp>_<libhash>_<randomsuffix>.model
            path_to_save = f"cache/w2v_temp_{timestamp}_{lib_hash}_{random_suffix}.model"
            
            print(f"\n⚠️ No save path provided for Word2Vec model!\n")
            print(f"Model will be temporarily saved to: {path_to_save}")
            print(f"To use this model in the future, either:")
            print(f"  1. Load it directly with toolkit.load_w2v('{path_to_save}')")
            print(f"  2. Rename it to a permanent location: toolkit.load_w2v('{path_to_save}', save_path='your/permanent/path.model')")
            print(f"  3. Train a new model with a specified path: toolkit.train_w2v(save_path='your/model/path.model')\n")
        
        print(f"Training Word2Vec model, will save to {path_to_save}")
        
        # Train new model
        self.w2v_model = train_model(
            library=self.library,
            file_path=path_to_save,
            vector_size=vector_size,
            window=window,
            epochs=epochs,
            workers=workers,
            n_decimals=n_decimals,
            show_progress=show_progress
        )
        
        # Update the instance variable for future reference
        self.w2v_path = path_to_save
        
        # If we used a temporary path, remind the user
        if "w2v_temp_" in path_to_save:
            print(f"\n✅ Model training complete!")
            print(f"Remember, this is a temporary file. If you want to keep it, rename or move it to a permanent location.")
        
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
        top_k_clusters=1):
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
        
        # The compare_spectra function will now return results with original compound names
        return results[:top_n]

    def search_w2v(self, query_input, top_n=10, intensity_power=0.6, top_k_clusters=1, n_decimals=2):
        """
        Preselector + Word2Vec embedding + cosine similarity.

        Args:
            query_input: Either a spectrum (list of tuples) or a vector (numpy array)
            top_n: Number of top results to return
            intensity_power: Exponent for intensity weighting
            top_k_clusters: Number of clusters/components to consider (for KMeans/GMM)
            n_decimals: Number of decimals to use for m/z values. Defaults to 2.
        
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
                query_spectrum = vector_to_spectrum(query_input, shift=self.mz_shift)
                query_vector = spectrum_to_vector(query_spectrum, max_mz=self.max_mz)
            else:
                query_vector = query_input
                query_spectrum = vector_to_spectrum(query_input, shift=self.mz_shift) 
        else:
            query_spectrum = [(mz + self.mz_shift, intensity) for mz, intensity in query_input]
            query_vector = spectrum_to_vector(query_spectrum, max_mz=self.max_mz)
        
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
            selected_keys = self.preselector.select(query_vector, list(self.library.keys()))
        
        # Create initial document with provided n_decimals
        query_doc = SpectrumDocument(query_spectrum, n_decimals=n_decimals)
        
        # First try with the specified n_decimals
        try:
            query_embedding = calc_embedding(self.w2v_model, query_doc, intensity_power)
            
            # If the embedding is all zeros, the vocabulary might be mismatched
            if np.all(query_embedding == 0):
                raise ValueError("No matching words found in model vocabulary - possible decimal precision mismatch")
                
        except (ValueError, IndexError) as e:
            # Try to detect the correct decimal precision from the model's vocabulary
            if len(self.w2v_model.wv.key_to_index) > 0:
                # Get the first key from the model's vocabulary
                sample_word = list(self.w2v_model.wv.key_to_index.keys())[0]
                
                # Parse the decimal precision from the first word (e.g., "peak@41.00" → n_decimals=2)
                if "peak@" in sample_word:
                    peak_value = sample_word.split("peak@")[1]
                    if "." in peak_value:
                        detected_decimals = len(peak_value.split(".")[1])
                    else:
                        detected_decimals = 0
                    
                    if detected_decimals != n_decimals:
                        print(f"Warning: Decimal precision mismatch. Model uses {detected_decimals} decimals, but search used {n_decimals}.")
                        print(f"Automatically adjusting to {detected_decimals} decimals.")
                        
                        # Recreate document with the correct decimal precision
                        query_doc = SpectrumDocument(query_spectrum, n_decimals=detected_decimals)
                        query_embedding = calc_embedding(self.w2v_model, query_doc, intensity_power)
                        
                        # Update n_decimals for library spectra embeddings too
                        n_decimals = detected_decimals
        
        # Calculate embeddings for library spectra with the (potentially adjusted) n_decimals
        embeddings = {name: calc_embedding(self.w2v_model, SpectrumDocument(self.library[name].spectrum, n_decimals=n_decimals), intensity_power)
                     for name in selected_keys if name in self.library}

        # Calculate similarities - avoid division by zero
        similarities = {}
        for dict_key, vec in embeddings.items():
            query_norm = np.linalg.norm(query_embedding)
            vec_norm = np.linalg.norm(vec)
            
            if query_norm > 0 and vec_norm > 0:
                # Use the original compound name, not the dictionary key with suffix
                compound_name = self.library[dict_key].name
                similarities[compound_name] = float(np.dot(query_embedding, vec) / (query_norm * vec_norm))
            else:
                compound_name = self.library[dict_key].name
                similarities[compound_name] = 0.0
                
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def download_library(self, version="2025.05.1", output_dir="cache", force=False, subset=None):
        """
        Download the MassBank library file and automatically load it.
        
        Args:
            version (str): MassBank version to download (default: "2025.05.1")
            output_dir (str): Directory to save the file (default: "cache")
            force (bool): Whether to download even if file exists (default: False)
            subset (str): Subset of elements to include when loading (default: None)
            
        Returns:
            dict: The loaded library
        """
        import os
        import requests
        from tqdm import tqdm
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        url = f"https://github.com/MassBank/MassBank-data/releases/download/{version}/MassBank.msp_NIST"
        output_path = os.path.join(output_dir, "MassBank.msp_NIST")
        json_path = os.path.join(output_dir, f"massbank_{version}.json")
        
        # Check if file already exists
        if os.path.exists(output_path) and not force:
            print(f"MassBank library already exists at {output_path}")
        else:
            print(f"Downloading MassBank library v{version}...")
            
            # Download file with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, 
                desc="MassBank"
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            print(f"Downloaded MassBank library to {output_path}")
        
        # Load the library automatically
        return self.load_library(
            file_path=output_path, 
            json_path=json_path, 
            subset=subset,
            save_path=json_path
        )
