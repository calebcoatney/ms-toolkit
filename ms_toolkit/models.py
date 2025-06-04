from dataclasses import asdict, dataclass
import json
from typing import List, Tuple


@dataclass
class Compound:
    name: str = None
    formula: str = None
    mw: float = None
    casno: str = None
    id_: int = None
    comment: str = None
    num_peaks: int = None
    spectrum: List[Tuple[int, float]] = None

    def to_json(self):
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(data):
        if isinstance(data, str):
            return Compound(**json.loads(data))
        else:
            return Compound(**data)

    def __repr__(self):
        repr_str = f"Name: {self.name}\n"
        repr_str += f"Formula: {self.formula}\n"
        repr_str += f"MW: {self.mw}\n"
        repr_str += f"CASNO: {self.casno}\n"
        repr_str += f"ID: {self.id_}\n"
        repr_str += f"Comment: {self.comment}\n"
        repr_str += f"Num peaks: {self.num_peaks}\n"
        if self.spectrum:
            for peak, intensity in self.spectrum:
                repr_str += f"{peak} {intensity:.2f}\n"
        return repr_str.strip()

# --- Spectrum Document (spec2vec-inspired) ---
class SpectrumDocument:
    """
    Create documents from spectra inspired by spec2vec.
    
    Each peak's m/z value is converted into a word (e.g., "peak@100.0").
    The list of words (self.words) is used for Word2Vec training.
    Peak intensities are stored separately in self.weights.
    """
    def __init__(self, spectrum, n_decimals: int = 0):
        self.spectrum = spectrum
        self.n_decimals = n_decimals
        self.words = self._make_words()
        self.weights = self._add_weights()
        self._index = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.words)

    def __next__(self):
        """gensim.models.Word2Vec() wants its corpus elements to be iterable"""
        if self._index < len(self.words):
            word = self.words[self._index]
            self._index += 1
            return word
        self._index = 0
        raise StopIteration

    def __str__(self):
        return self.words.__str__()

    def _make_words(self) -> List[str]:
        # Convert each peak m/z to a word (losses omitted for GC-MS)
        peak_words = [f"peak@{mz:.{self.n_decimals}f}" for mz, _ in self.spectrum]
        return peak_words

    def _add_weights(self) -> List[float]:
        # Store the normalized intensities as weights.
        intensities = [intensity for _, intensity in self.spectrum]
        return intensities