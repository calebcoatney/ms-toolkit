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

def train_model(library: dict, file_path: str, vector_size: int = 300, window: int = 500, workers: int = 16, epochs: int = 5, n_decimals: int = 2, show_progress: bool = True):   
    from tqdm.auto import tqdm
    
    library_documents = [SpectrumDocument(compound.spectrum, n_decimals=n_decimals) for compound in library.values()]
    
    # Set up callbacks for progress reporting
    class TqdmCallback(object):
        def __init__(self, total_epochs):
            self.pbar = tqdm(total=total_epochs, desc="Training Word2Vec model", unit="epoch")
            self.epoch = 0
            self.prev_loss = 0
            
        def on_epoch_begin(self, model):
            pass
            
        def on_epoch_end(self, model):
            self.epoch += 1
            if model.get_latest_training_loss():
                current_loss = model.get_latest_training_loss()
                loss_diff = current_loss - self.prev_loss
                self.prev_loss = current_loss
                self.pbar.set_postfix(loss=f"{loss_diff:.4f}")
            self.pbar.update(1)
            
        def on_train_begin(self, model):
            self.prev_loss = model.get_latest_training_loss() or 0
            
        def on_train_end(self, model):
            self.pbar.close()
    
    # Initialize and train the model with progress tracking
    callbacks = [TqdmCallback(epochs)] if show_progress else []
    
    try:
        # For Gensim 4.x
        model = Word2Vec(
            library_documents, 
            vector_size=vector_size, 
            window=window, 
            min_count=1, 
            workers=workers, 
            compute_loss=True, 
            epochs=epochs,
            callbacks=callbacks
        )
    except TypeError:
        # Fallback for Gensim 3.x which doesn't support callbacks parameter
        if show_progress:
            print(f"Training Word2Vec model for {epochs} epochs...")
            model = Word2Vec(
                library_documents, 
                vector_size=vector_size, 
                window=window, 
                min_count=1, 
                workers=workers, 
                compute_loss=True, 
                epochs=1  # Start with 1 epoch
            )
            pbar = tqdm(total=epochs, desc="Training Word2Vec model", unit="epoch")
            pbar.update(1)  # First epoch already done
            
            # Train the remaining epochs manually with progress updates
            for _ in range(epochs - 1):
                model.train(
                    library_documents,
                    total_examples=model.corpus_count,
                    epochs=1
                )
                pbar.update(1)
            pbar.close()
        else:
            model = Word2Vec(
                library_documents, 
                vector_size=vector_size, 
                window=window, 
                min_count=1, 
                workers=workers, 
                compute_loss=True, 
                epochs=epochs
            )
    
    model.save(file_path)
    return model

def load_model(file_path: str):
    return Word2Vec.load(file_path)

# Adapted from the Spec2Vec project.
# Calculates a weighted spectrum embedding.

def calc_embedding(model, document, intensity_power):
    # Check if there are any matching words in the model's vocabulary
    idx_not_in_model = [i for i, x in enumerate(document.words) if x not in model.wv.key_to_index]
    words_in_model = [x for i, x in enumerate(document.words) if i not in idx_not_in_model]
    
    # Return zero vector if no words match the model's vocabulary
    if not words_in_model:
        return np.zeros(model.wv.vector_size)
    
    weights_in_model = np.asarray([x for i, x in enumerate(document.weights)
                                  if i not in idx_not_in_model]).reshape(len(words_in_model), 1)

    word_vectors = model.wv[words_in_model]
    weights_raised = np.power(weights_in_model, intensity_power)

    weights_raised_tiled = np.tile(weights_raised, (1, model.wv.vector_size))
    return np.sum(word_vectors * weights_raised_tiled, 0)