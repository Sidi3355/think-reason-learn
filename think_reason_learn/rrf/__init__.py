"""Random Rule Forest: YES/NO questions written by a language model, then weighed."""

from ._decision import answers_text
from ._rrf import RRF, Combiner

__all__ = ["RRF", "Combiner", "answers_text"]
