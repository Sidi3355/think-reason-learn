"""Random Rule Forest (RRF): YES/NO questions written by a language model, weighed.

Griffin, Yin, Vidaurre, Koyluoglu, Ternasky, Alican and Ihlamur, "Random Rule
Forest: Interpretable Ensembles of LLM-Generated Questions for Predicting
Startup Success" (2025), https://arxiv.org/abs/2505.24622.

The generator writes questions from small class-balanced batches of training
samples, and the answerer answers every question for them. As in 0.2, an elastic
net weighs every question's answers (Jev's probabilities by default), and its
threshold is tuned for F1 by cross-validation. The paper's vote is
``combiner="vote"``: questions are ranked by F-beta, weak and redundant ones
dropped, and the prediction is positive when at least T of the top N questions
say YES, with (N, T) tuned by cross-validation; the paper also writes questions
from a held-out generation set (``generation_size=500``) and uses ``beta=0.5``.
``combiner="decision"`` hands the answers to a decision model instead: each sample's
answers, as text, go to ``combiner_model`` (Jev, or Kev on your own server), whose
probability that the sample is positive is the score, with an F-beta threshold.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel

from think_reason_learn.core._answers import yes_probabilities
from think_reason_learn.core._classifier import BinaryTRLClassifier, Labels, Proba
from think_reason_learn.core._combine import (
    LinearCombiner,
    best_threshold,
    fit_combiner,
)
from think_reason_learn.core._data import (
    Classes,
    ClassRatio,
    TextInput,
    balanced_subset,
    class_shares,
    count,
    example_batches,
    take,
)
from think_reason_learn.core._prompts import examples_block, most_written
from think_reason_learn.exceptions import DataError, NoOutputError
from think_reason_learn.llm import (
    QUESTION_TOKENS,
    Endpoint,
    LLMSession,
    Models,
    UsageRecord,
    YesNoQuestion,
    estimate_answers,
    estimate_generations,
    mean_tokens,
)

from ._decision import answers_text, default_question
from ._prompts import INSTRUCTIONS_PURPOSE, QUESTION_INSTRUCTIONS, question_prompt
from ._selection import as_yes, question_stats, rank, tune_vote

type Combiner = Literal["vote", "elasticnet", "decision"]

_COMBINERS: tuple[Combiner, ...] = ("elasticnet", "vote", "decision")


class RRF(BinaryTRLClassifier):
    """Random Rule Forest: YES/NO questions, weighed by elastic net or put to a vote.

    For two classes. The positive class is ``pos_label``, by default the second
    of the sorted labels (``"YES"`` of ``"NO"``/``"YES"``). The defaults are
    those tuned in practice in 0.2; each setting names the paper's value where
    it differs.

    Args:
        task: What to predict, in plain words.
        generator: Model(s) that write the instructions and the questions.
        answerer: Model(s) that answer the questions. Defaults to Jev.
        instructions: Instructions for writing questions; written from ``task``
            during ``fit`` when not given (see ``write_instructions``).
        expert_questions: Questions of your own, scored like the written ones.
        n_prompts: Question-writing prompts; the paper uses 25. 0 writes none,
            scoring only ``expert_questions``, and needs no generator.
        questions_per_prompt: Questions per prompt; the paper uses 10.
        n_examples: Labelled samples shown per prompt; the paper uses 20.
        example_class_ratio: Class shares among them; the paper balances them.
        generation_size: ``None`` writes questions from the samples that score
            them. Otherwise, as the paper does, that many samples (int) or that
            fraction of the training samples (float) are set aside for writing
            and not used to score, so selection is not tuned on samples the
            generator saw; each class keeps at least half its samples for
            scoring.
        beta: F-beta used to rank questions and tune the threshold; the paper
            uses 0.5.
        drop_below_base_rate: The vote drops questions whose precision is at
            most the positive rate.
        hamming_threshold: The vote drops a question whose answers differ from a
            better question's on fewer than this share of samples.
        combiner: ``"elasticnet"``, a logistic regression on the answers'
            probabilities that weighs every question (C of 1, L1 ratio of 0.1
            or 0.5 by inner cross-validation); ``"vote"``, the paper's; or
            ``"decision"``, a decision model reading each sample's answers as
            text (see ``answers_text``), with an F-beta threshold on its
            probabilities. Nothing is fitted for it here: the model decides as
            it is, so fine-tune it elsewhere on ``answers_text`` of training
            samples if it should learn the task. Its threshold is tuned on the
            samples ``fit`` scores, so it is optimistic if the model was trained
            on them.
        combiner_model: Decision model(s) for ``combiner="decision"``, e.g.
            ``"jev"`` or ``LLM("kev", base_url=".../v1/systemone",
            kind="decision")``. Ignored by the other combiners.
        combiner_question: The question ``combiner_model`` answers about each
            sample's answers. ``None`` asks whether the sample is the positive
            class, given ``task``.
        max_rules: Most questions the combiner may use, the best by F-beta.
        cv: Folds for tuning the elastic net or the vote.
        screening_size: Score every question on this many samples (int) or
            share of samples (float) first, and answer only the best
            ``screened_questions`` for the rest. ``None`` answers all.
        screened_questions: Questions kept after screening.
        pos_label: The positive class.
        random_state: Seed for examples, splits and folds.
        verbose: Show progress when above 0.
    """

    _instructions_purpose = INSTRUCTIONS_PURPOSE

    def __init__(
        self,
        task: str | None = None,
        generator: Models | None = None,
        answerer: Models = "jev",
        *,
        instructions: str | None = None,
        expert_questions: tuple[str, ...] = (),
        n_prompts: int = 10,
        questions_per_prompt: int = 10,
        n_examples: int = 30,
        example_class_ratio: ClassRatio = "balanced",
        generation_size: int | float | None = None,
        beta: float = 1.0,
        drop_below_base_rate: bool = True,
        hamming_threshold: float = 0.15,
        combiner: Combiner = "elasticnet",
        combiner_model: Models | None = None,
        combiner_question: str | None = None,
        max_rules: int | None = None,
        cv: int = 3,
        screening_size: int | float | None = None,
        screened_questions: int = 20,
        pos_label: str | int | None = None,
        random_state: int | None = 42,
        verbose: int = 0,
    ) -> None:
        self.task = task
        self.generator = generator
        self.answerer = answerer
        self.instructions = instructions
        self.expert_questions = expert_questions
        self.n_prompts = n_prompts
        self.questions_per_prompt = questions_per_prompt
        self.n_examples = n_examples
        self.example_class_ratio = example_class_ratio
        self.generation_size = generation_size
        self.beta = beta
        self.drop_below_base_rate = drop_below_base_rate
        self.hamming_threshold = hamming_threshold
        self.combiner = combiner
        self.combiner_model = combiner_model
        self.combiner_question = combiner_question
        self.max_rules = max_rules
        self.cv = cv
        self.screening_size = screening_size
        self.screened_questions = screened_questions
        self.pos_label = pos_label
        self.random_state = random_state
        self.verbose = verbose

    example_class_ratio: ClassRatio
    combiner: Combiner
    combiner_model: Models | None
    combiner_question: str | None
    instructions_: str
    questions_: pd.DataFrame
    rules_: tuple[str, ...]
    min_votes_: int
    combiner_: LinearCombiner | None
    decision_question_: str | None
    """What the decision model is asked, when ``combiner="decision"`` was fitted."""
    threshold_: float

    def answers(self, X: TextInput) -> pd.DataFrame:
        """Probability of YES for each rule (columns) and sample (rows)."""
        matrix = self._run(
            X,
            "answering",
            lambda texts, session: self._answer(texts, self.rules_, session),
        )
        return pd.DataFrame(matrix, columns=pd.Index(self.rules_))

    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        positive = self._positive(y)
        if not (self.n_prompts or self.expert_questions):
            raise ValueError("n_prompts=0 needs expert_questions=(...) to score.")
        writer = self._writer()
        self._models("answerer", "answer")  # fail before paying for questions
        self._decider()
        rng = np.random.default_rng(self.random_state)
        generation, scoring = self._split_generation(y, len(self.classes_), rng)
        self.instructions_ = (
            await self._instructions(session, writer)
            if writer
            else self.instructions or ""
        )
        questions = await self._write(texts, y, generation, writer, session, rng)
        if not questions:
            raise NoOutputError("The generator wrote no usable questions.")
        questions, answers = await self._score(
            texts, scoring, positive[scoring], questions, session, rng
        )
        await self._select(questions, answers, positive[scoring], session)

    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        answers = await self._answer(texts, self.rules_, session)
        question = getattr(self, "decision_question_", None)
        if question is not None:
            scores = await self._decision_scores(
                self.rules_, answers, question, session
            )
            return self._binary_proba(scores)
        return self._binary_proba(self._positive_scores(answers))

    def _positive_scores(
        self, answers: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if self.combiner_ is not None:
            return self.combiner_.scores(np.nan_to_num(answers, nan=0.0))
        if not self.rules_:
            return np.zeros(len(answers))
        return as_yes(answers).mean(axis=1)

    def _fit_plan(
        self, texts: list[str], y: Labels, classes: Classes
    ) -> list[UsageRecord]:
        writer = self._writer()
        answerer = self._models("answerer", "answer")
        rng = np.random.default_rng(self.random_state)
        _, scoring = self._split_generation(y, len(classes), rng)
        rows = take(texts, scoring)
        written = self.n_prompts * self.questions_per_prompt
        n_questions = written + len(self.expert_questions)
        prompt = self.n_examples * mean_tokens(texts) + self._instructions_tokens()
        planned = (
            [
                *self._instructions_plan(),
                estimate_generations(writer, prompt, calls=self.n_prompts),
            ]
            if writer
            else []
        )
        if self.screening_size is None or n_questions <= self.screened_questions:
            answering = [
                estimate_answers(answerer, rows, [QUESTION_TOKENS] * n_questions)
            ]
            kept = n_questions
        else:
            screen = count(self.screening_size, len(rows))
            answering = [
                estimate_answers(
                    answerer, rows[:screen], [QUESTION_TOKENS] * n_questions
                ),
                estimate_answers(
                    answerer, rows[screen:], [QUESTION_TOKENS] * self.screened_questions
                ),
            ]
            kept = self.screened_questions
        return [*planned, *answering, *self._decision_plan(len(rows), kept)]

    def _predict_plan(self, texts: list[str]) -> list[UsageRecord]:
        answerer = self._models("answerer", "answer")
        plan = [estimate_answers(answerer, texts, list(self.rules_))]
        if getattr(self, "decision_question_", None) is not None:
            state = answers_text(self.rules_, [0.0] * len(self.rules_))
            plan.append(
                estimate_answers(
                    self._models("combiner_model", "answer"),
                    [state] * len(texts),
                    [self.decision_question_ or ""],
                    role="combiner",
                )
            )
        return plan

    def _decision_plan(self, n_samples: int, n_questions: int) -> list[UsageRecord]:
        """The decision model's calls during ``fit``: one per scored sample."""
        decider = self._decider()
        if not decider:
            return []
        n_rules = min(n_questions, self.max_rules or n_questions)
        # A rule's line is about a question's length; four characters a token.
        state = "x" * (4 * n_rules * (QUESTION_TOKENS + 1))
        question = self.combiner_question or default_question(self._task(), "yes")
        return [
            estimate_answers(decider, [state] * n_samples, [question], role="combiner")
        ]

    def _decider(self) -> tuple[Endpoint, ...]:
        """The decision combiner's model(s); none for the other combiners.

        Raises:
            ValueError: ``combiner`` is unknown, or ``"decision"`` without a
                ``combiner_model``.
        """
        if self.combiner not in _COMBINERS:
            raise ValueError(
                f"combiner={self.combiner!r} is not one of {list(_COMBINERS)}."
            )
        if self.combiner != "decision":
            return ()
        if self.combiner_model is None:
            raise ValueError(
                'combiner="decision" needs combiner_model=..., a decision model '
                'such as "jev".'
            )
        return self._models("combiner_model", "answer")

    def _writer(self) -> tuple[Endpoint, ...]:
        """The generator, if questions are written: ``n_prompts=0`` needs none."""
        return self._models("generator", "generate") if self.n_prompts else ()

    def _check_classes(self, classes: Classes) -> None:
        super()._check_classes(classes)
        class_shares(classes, self.example_class_ratio)  # a bad ratio fails now

    def _split_generation(
        self, y: Labels, n_classes: int, rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """Samples for writing questions, and those for scoring them."""
        if self.generation_size is None:
            every = np.arange(len(y), dtype=np.intp)
            return every, every
        size = count(self.generation_size, len(y))
        generation = balanced_subset(y, n_classes, size, rng, max_share=0.5)
        if not len(generation):
            raise DataError(
                "No samples to write questions from: each class sets aside at most "
                "half its samples. Give more samples, or a larger generation_size."
            )
        scoring = np.setdiff1d(np.arange(len(y)), generation).astype(np.intp)
        return generation, scoring

    async def _write(
        self,
        texts: list[str],
        y: Labels,
        generation: npt.NDArray[np.intp],
        writer: tuple[Endpoint, ...],
        session: LLMSession,
        rng: np.random.Generator,
    ) -> list[str]:
        """Questions from every prompt, deduplicated, then the expert's."""
        prompts = self._prompts(texts, y, generation, rng)
        written = await most_written(
            [
                session.generate(
                    writer,
                    p,
                    role="generator",
                    schema=_Questions,
                    instructions=QUESTION_INSTRUCTIONS,
                )
                for p in prompts
            ],
            "prompts",
        )
        candidates = [
            q for batch in written for q in batch.questions[: self.questions_per_prompt]
        ]
        return _unique([*candidates, *self.expert_questions])

    def _prompts(
        self,
        texts: list[str],
        y: Labels,
        generation: npt.NDArray[np.intp],
        rng: np.random.Generator,
    ) -> list[str]:
        """One prompt per batch of class-balanced examples from ``generation``."""
        batches = example_batches(
            y[generation],
            self.classes_,
            n_batches=self.n_prompts,
            batch_size=min(self.n_examples, len(generation)),
            class_ratio=self.example_class_ratio,
            rng=rng,
        )
        return [
            question_prompt(
                self._task(),
                self.instructions_,
                self.questions_per_prompt,
                examples_block(texts, y, self.classes_, generation[batch]),
            )
            for batch in batches
        ]

    async def _score(
        self,
        texts: list[str],
        scoring: npt.NDArray[np.intp],
        positive: npt.NDArray[np.bool_],
        questions: list[str],
        session: LLMSession,
        rng: np.random.Generator,
    ) -> tuple[list[str], npt.NDArray[np.float64]]:
        """Answers of the scoring samples, after screening if it is on."""
        rows = take(texts, scoring)
        if self.screening_size is None or len(questions) <= self.screened_questions:
            return questions, await self._answer(rows, questions, session)
        screen = rng.permutation(len(rows))[: count(self.screening_size, len(rows))]
        screened = await self._answer(take(rows, screen), questions, session)
        _, _, scores = question_stats(as_yes(screened), positive[screen], self.beta)
        best = np.argsort(-scores, kind="stable")[: self.screened_questions]
        kept = take(questions, best)
        rest = np.setdiff1d(np.arange(len(rows)), screen).astype(np.intp)
        answers = np.empty((len(rows), len(kept)))
        answers[screen] = screened[:, best]
        answers[rest] = await self._answer(take(rows, rest), kept, session)
        return kept, answers

    async def _answer(
        self, texts: Sequence[str], questions: Sequence[str], session: LLMSession
    ) -> npt.NDArray[np.float64]:
        """P(YES) per sample (rows) and question (columns); NaN where unanswered."""
        return await yes_probabilities(
            session,
            self._models("answerer", "answer"),
            texts,
            questions,
            missing=np.nan,
        )

    async def _select(
        self,
        questions: list[str],
        answers: npt.NDArray[np.float64],
        positive: npt.NDArray[np.bool_],
        session: LLMSession,
    ) -> None:
        yes = as_yes(answers)
        precision, recall, scores = question_stats(yes, positive, self.beta)
        self.questions_ = pd.DataFrame(
            {
                "question": questions,
                "precision": precision,
                "recall": recall,
                f"f{self.beta:g}": scores,
            }
        )
        self.combiner_, self.min_votes_ = None, 0
        self.decision_question_ = None
        if self.combiner == "vote":
            # Tuning the vote is CPU work: off the shared event loop.
            used = await asyncio.to_thread(self._fit_vote, yes, positive)
        elif self.combiner == "decision":
            used = await self._fit_decision(questions, yes, answers, positive, session)
        else:
            used = await self._fit_elasticnet(yes, answers, positive)
        self.rules_ = tuple(questions[j] for j in used)
        self.questions_["used"] = [j in used for j in range(len(questions))]

    def _fit_vote(
        self, yes: npt.NDArray[np.bool_], positive: npt.NDArray[np.bool_]
    ) -> list[int]:
        """The rules of the tuned vote; sets its minimum votes and threshold."""
        ranked, n_rules, min_votes = tune_vote(
            yes,
            positive,
            beta=self.beta,
            drop_below_base_rate=self.drop_below_base_rate,
            hamming_threshold=self.hamming_threshold,
            max_rules=self.max_rules,
            cv=self.cv,
            random_state=self.random_state,
        )
        self.min_votes_ = min_votes
        self.threshold_ = min_votes / n_rules if n_rules else math.inf
        return ranked[:n_rules]

    async def _fit_elasticnet(
        self,
        yes: npt.NDArray[np.bool_],
        answers: npt.NDArray[np.float64],
        positive: npt.NDArray[np.bool_],
    ) -> list[int]:
        """Rules weighted by elastic net; sets the combiner and its threshold.

        Every question goes in (the best ``max_rules`` if set): the L1 penalty
        drops the weak and redundant ones itself.
        """
        ranked = rank(
            yes,
            positive,
            beta=self.beta,
            drop_below_base_rate=False,
            hamming_threshold=0,
        )[: self.max_rules]
        if not ranked:
            self.threshold_ = math.inf
            return []
        self.combiner_, kept = await fit_combiner(
            np.nan_to_num(answers[:, ranked], nan=0.0),
            positive,
            l1_ratios=(0.1, 0.5),
            cv=self.cv,
            beta=self.beta,
            class_weight=None,
            random_state=self.random_state,
            cs=(1.0,),
            drop_zero=True,
        )
        self.threshold_ = self.combiner_.threshold
        return [ranked[j] for j in kept]

    async def _fit_decision(
        self,
        questions: list[str],
        yes: npt.NDArray[np.bool_],
        answers: npt.NDArray[np.float64],
        positive: npt.NDArray[np.bool_],
        session: LLMSession,
    ) -> list[int]:
        """Rules for the decision model, best by F-beta first; sets its threshold.

        Every question goes in (the best ``max_rules`` if set), so the model
        reads them all; the threshold maximises F-beta on its probabilities for
        the scored samples.
        """
        ranked = rank(
            yes,
            positive,
            beta=self.beta,
            drop_below_base_rate=False,
            hamming_threshold=0,
        )[: self.max_rules]
        self.decision_question_ = self.combiner_question or default_question(
            self._task(), self._positive_label()
        )
        if not ranked:
            self.threshold_ = math.inf
            return []
        rules = [questions[j] for j in ranked]
        scores = await self._decision_scores(
            rules, answers[:, ranked], self.decision_question_, session
        )
        self.threshold_ = best_threshold(scores, positive, self.beta)
        return ranked

    async def _decision_scores(
        self,
        rules: Sequence[str],
        answers: npt.NDArray[np.float64],
        question: str,
        session: LLMSession,
    ) -> npt.NDArray[np.float64]:
        """The decision model's probability that each sample is positive.

        An unanswered sample scores 0, read as negative; the session warns
        about every missing answer.
        """
        if not len(answers):
            return np.zeros(0)
        states = [answers_text(rules, row) for row in answers]
        replies = await session.answer(
            self._models("combiner_model", "answer"),
            states,
            {"positive": YesNoQuestion(question)},
            role="combiner",
        )
        return np.array(
            [
                a.probability() if (a := row["positive"]) is not None else 0.0
                for row in replies
            ]
        )

    def _summary(self) -> Mapping[str, object]:
        return {
            "instructions": self.instructions_,
            "rules": list(self.rules_),
            "min_votes": self.min_votes_,
            "decision_question": getattr(self, "decision_question_", None),
            "threshold": self.threshold_,
            "questions": self.questions_.to_dict("records"),
        }


def _unique(questions: Sequence[str]) -> list[str]:
    """Questions in order, without blanks or repeats (ignoring case)."""
    seen: set[str] = set()
    unique: list[str] = []
    for question in (q.strip() for q in questions):
        if question and question.lower() not in seen:
            seen.add(question.lower())
            unique.append(question)
    return unique


class _Questions(BaseModel):
    questions: list[str]
