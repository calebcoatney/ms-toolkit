# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file are adapted from the Spec2Vec project
# (https://github.com/iomega/spec2vec) under the Apache License 2.0.

from .models import SpectrumDocument
from gensim.models import Word2Vec
import numpy as np

def train_model(library: dict, file_path: str, vector_size: int = 300, window: int = 500, workers: int = 16, epochs: int = 5):   

    library_documents = [SpectrumDocument(spectrum) for spectrum in library.values()]
    model = Word2Vec(library_documents, vector_size=vector_size, window=window, min_count=1, workers=workers, compute_loss=True, epochs=epochs)
    model.save(file_path)

def load_model(file_path: str):
    return Word2Vec.load(file_path)

# Adapted from the Spec2Vec project.
# Calculates a weighted spectrum embedding.

def calc_embedding(model, document, intensity_power):

    idx_not_in_model = [i for i, x in enumerate(document.words) if x not in model.wv.key_to_index]
    words_in_model = [x for i, x in enumerate(document.words) if i not in idx_not_in_model]
    weights_in_model = np.asarray([x for i, x in enumerate(document.weights)
                                   if i not in idx_not_in_model]).reshape(len(words_in_model), 1)

    word_vectors = model.wv[words_in_model]
    weights_raised = np.power(weights_in_model, intensity_power)

    weights_raised_tiled = np.tile(weights_raised, (1, model.wv.vector_size))
    return np.sum(word_vectors * weights_raised_tiled, 0)