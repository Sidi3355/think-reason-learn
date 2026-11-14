"""RRF's decision combiner: a decision model reads each sample's answers and decides."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from think_reason_learn.llm import Question, config_context
from think_reason_learn.rrf import RRF, answers_text
from think_reason_learn.testing import FakeLLM, GenerateRequest, fake_llms

QUESTIONS = (
    "Has the founder exited a company?",
    "Is the founder a chef?",
    "Does the founder code?",
)
X = [
    f"Founder {i}: {'has exited a company' if i % 2 else 'never exited'}, "
    f"{'codes' if i % 3 else 'does not code'}."
    for i in range(30)
]
Y = ["good" if i % 2 else "bad" for i in range(30)]
ASKED = "Will this founder's company succeed?"


def write(request: GenerateRequest) -> str | dict[str, object]:
    if request.schema is None:
        return "Look at the founder's track record."
    return {"questions": list(QUESTIONS)}


def answer(sample: str, question: Question) -> float:
    """The answerer: reads the founder text."""
    if question.text == QUESTIONS[0]:
        return 0.9 if "has exited" in sample else 0.2
    if question.text == QUESTIONS[2]:
        return 0.8 if "does not code" not in sample else 0.1
    return 0.05


class Decider:
    """A decision model that records what it reads.

    Positive if the exit question's line says yes.
    """

    def __init__(self) -> None:
        self.read: list[tuple[str, str]] = []

    def __call__(self, state: str, question: Question) -> float:
        self.read.append((state, question.text))
        return 0.85 if f"{QUESTIONS[0]} yes" in state else 0.15


@pytest.fixture
def models() -> Iterator[tuple[FakeLLM, FakeLLM, Decider]]:
    decider = Decider()
    writer = FakeLLM(generate=write)
    judge = FakeLLM(answer=answer, kind="decision")
    with fake_llms(
        writer=writer, judge=judge, decider=FakeLLM(answer=decider, kind="decision")
    ):
        yield writer, judge, decider


def rrf(**settings: Any) -> RRF:
    chosen: dict[str, Any] = {
        "task": "Predict whether the founder's company succeeds.",
        "generator": "fake:writer",
        "answerer": "fake:judge",
        "n_prompts": 1,
        "questions_per_prompt": 3,
        "combiner": "decision",
        "combiner_model": "fake:decider",
    }
    return RRF(**{**chosen, **settings})


def test_answers_text_is_one_line_per_rule_in_order() -> None:
    text = answers_text(["Q1?", " Q2? ", "Q3?", "Q4?"], [0.5, 0.49, float("nan"), 1.0])
    assert text == "Q1? yes\nQ2? no\nQ3? unknown\nQ4? yes"
    with pytest.raises(ValueError, match="3 questions but 2 answers"):
        answers_text(["a", "b", "c"], [0.1, 0.2])


def test_the_decision_model_reads_the_answers_and_decides(
    models: tuple[FakeLLM, FakeLLM, Decider],
) -> None:
    _, _, decider = models
    model = rrf(combiner_question=ASKED).fit(X, Y)

    # Every question goes in, best by F-beta first; the exit question predicts.
    assert model.rules_[0] == QUESTIONS[0]
    assert set(model.rules_) == set(QUESTIONS)
    assert model.questions_["used"].all()
    assert model.combiner_ is None and model.decision_question_ == ASKED
    # It read the answers as text, one line per rule, and was asked our question.
    state, asked = decider.read[0]
    assert asked == ASKED
    assert [line.rsplit(" ", 1)[0] for line in state.splitlines()] == list(model.rules_)
    # Its probabilities are the scores; the threshold separates them.
    proba = model.predict_proba(X)
    assert set(np.round(proba[:, 1], 2)) == {0.15, 0.85}
    assert 0.15 < model.threshold_ <= 0.85
    assert list(model.predict(X)) == Y


def test_prediction_asks_the_answerer_then_the_decision_model(
    models: tuple[FakeLLM, FakeLLM, Decider],
) -> None:
    _, judge, decider = models
    model = rrf().fit(X, Y)
    judge_calls, decider_reads = judge.n_calls, len(decider.read)
    with config_context(cache=False):  # its answers match a fitted sample's
        model.predict_proba(["Founder: has exited a company, codes."])
    assert judge.n_calls > judge_calls
    [(state, _)] = decider.read[decider_reads:]
    assert f"{QUESTIONS[0]} yes" in state


def test_the_default_question_names_the_task_and_positive_class(
    models: tuple[FakeLLM, FakeLLM, Decider],
) -> None:
    _, _, decider = models
    model = rrf().fit(X, Y)
    assert model.decision_question_ is not None
    assert "Predict whether the founder's company succeeds." in model.decision_question_
    assert '"good"' in model.decision_question_
    assert decider.read[0][1] == model.decision_question_


def test_combiner_model_is_needed_before_any_call(
    models: tuple[FakeLLM, FakeLLM, Decider],
) -> None:
    writer, judge, _ = models
    with pytest.raises(ValueError, match="needs combiner_model"):
        rrf(combiner_model=None).fit(X, Y)
    with pytest.raises(ValueError, match="is not one of"):
        rrf(combiner="average").fit(X, Y)
    assert writer.n_calls == judge.n_calls == 0


def test_other_combiners_ignore_combiner_model(
    models: tuple[FakeLLM, FakeLLM, Decider],
) -> None:
    _, _, decider = models
    model = rrf(combiner="elasticnet").fit(X, Y)
    assert model.decision_question_ is None and model.combiner_ is not None
    model.predict_proba(X[:2])
    assert decider.read == []


def test_estimate_cost_counts_the_decision_model(
    models: tuple[FakeLLM, FakeLLM, Decider],
) -> None:
    fit_plan = rrf().estimate_cost(X, Y).by("role")
    assert fit_plan.set_index("role").loc["combiner", "calls"] == len(X)
    model = rrf().fit(X, Y)
    predict_plan = model.estimate_cost(X[:4]).by("role").set_index("role")
    assert predict_plan.loc["combiner", "calls"] == 4


def test_save_and_load_keep_the_decision_combiner(
    models: tuple[FakeLLM, FakeLLM, Decider], tmp_path: object
) -> None:
    from pathlib import Path

    model = rrf(combiner_question=ASKED).fit(X, Y)
    folder = model.save(Path(str(tmp_path)) / "m")
    loaded = RRF.load(folder)
    assert loaded.decision_question_ == ASKED
    assert loaded.rules_ == model.rules_
    assert math.isclose(loaded.threshold_, model.threshold_)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
