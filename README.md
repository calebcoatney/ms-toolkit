# ms-toolkit

Tools for mass spectrometry (MS) library searching and model training.

This library provides a pipeline for vectorizing spectra, training Word2Vec
models, preselecting candidates using clustering/GMM, and searching using
weighted cosine or embedding similarity. Portions of the code are adapted from
the [Spec2Vec](https://github.com/iomega/spec2vec) project.

## Features

- Parse MS library text files with optional progress UI
- Create `SpectrumDocument` objects for Word2Vec training
- Train and load Word2Vec models (`w2v.py`)
- Vectorize spectra and perform similarity search (`preprocessing.py`, `similarity.py`)
- Preselect candidates using KMeans or Gaussian Mixture Models (`preselector.py`)
- High-level `MSToolkit` facade wrapping the workflow (`api.py`)

## Installation

Install with pip using the included `setup.py`:

```bash
pip install .
```

Dependencies include `numpy`, `joblib`, `gensim`, and `scikit-learn`. Optional UI
features require `customtkinter` or `PySide6`.

## Quick Start: Open MassBank Workflow

Pretrained models for MassBank are provided for immediate use.

> **Note:** [MassBank](https://massbank.eu) is an open source mass spectral library for small molecule identification. It is freely available and can be downloaded directly through the ms-toolkit API.

```python
from ms_toolkit.api import MSToolkit

# Initialize toolkit (defaults to open MassBank workflow)
toolkit = MSToolkit()

# Download and load the MassBank library (first run will download and cache)
toolkit.download_library()

# Load pretrained MassBank Word2Vec and preselector models
toolkit.load_w2v('models/massbank_25epochs.model')
toolkit.load_preselector('models/massbank_kmeans.pkl')

# Search using a query spectrum (list of (m/z, intensity) tuples)
query = [
    (27.0, 0.09),
    (39.0, 0.04),
    (41.0, 0.09),
    (43.0, 0.33),
    (71.0, 0.28),
    (114.0, 0.05),
]
results = toolkit.search_w2v(query)
for compound, score in results:
    print(f'{compound}: {score:.3f}')
```

## Advanced Usage

You can also train your own models or use other libraries (e.g., NIST) if available:

```python
# Example: Train your own Word2Vec model
toolkit = MSToolkit(library_txt="your_library.txt")
toolkit.load_library()
toolkit.vectorize_library()
toolkit.train_preselector(save_path="my_kmeans.pkl")
toolkit.train_w2v(save_path="my_w2v.model")
```

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
The `NOTICE` file explains that some code derives from Spec2Vec, which is also
Apache 2.0 licensed.

