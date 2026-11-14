"""The decision combiner's text: a sample's answers as lines a decision model reads.

With ``combiner="decision"``, RRF does not weigh the answers itself. It writes each
sample's answers as text, one line per rule in rule order, and asks a decision model
(Jev, Kev, or any model behind TypeSafe's System One API) one yes/no question about
that text: is the sample positive? The model's probability is the score.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from think_reason_learn.llm import YES_AT


def answers_text(questions: Sequence[str], yes_probabilities: Sequence[float]) -> str:
    """A sample's answers as the decision model reads them.

    One line per question, in the order given: the question, then ``yes`` when its
    probability of yes is at least 0.5, ``no`` below that, and ``unknown`` where it
    was not answered (NaN). For example::

        Has the founder founded a company before? yes
        Did the founder study at a top-50 university? no

    Fine-tuning a decision model on these texts, then using it as the combiner,
    needs the texts built exactly this way: use this function for both.

    Args:
        questions: The rules, in order.
        yes_probabilities: Each rule's probability of yes for the sample.
    """
    if len(questions) != len(yes_probabilities):
        raise ValueError(
            f"{len(questions)} questions but {len(yes_probabilities)} answers."
        )
    lines: list[str] = []
    for question, probability in zip(questions, yes_probabilities, strict=True):
        p = float(probability)
        word = "unknown" if math.isnan(p) else "yes" if p >= YES_AT else "no"
        lines.append(f"{question.strip()} {word}")
    return "\n".join(lines)


def default_question(task: str, positive: str) -> str:
    """What the decision model is asked when ``combiner_question`` is not set."""
    return (
        f"{task.strip()} Given the answers to these yes/no questions about the "
        f'sample, is it "{positive}"?'
    )
