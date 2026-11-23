"""Prompt pieces every model shares: labelled examples and task instructions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from think_reason_learn.core._data import Classes
from think_reason_learn.exceptions import NoOutputError
from think_reason_learn.llm import Endpoint, LLMSession

logger = logging.getLogger(__name__)

_INSTRUCTIONS_WRITER: Final = """\
You write the instructions another model will follow to write {purpose} for a
classification task. Turn the task description into clear, specific guidance:
what the classes mean, which signals in a sample are likely to matter, and what
to avoid (anything that restates the label, or that cannot be judged from a
sample's text alone). Do not invent facts about the data. Reply with the
instructions only."""

_YES_NO: Final = (("yes", "no"), ("true", "false"), ("1", "0"))
"""Pairs of labels, positive first, that read as a judge's YES and NO as they are."""


async def most_written[T](calls: Sequence[Awaitable[T]], what: str) -> list[T]:
    """The replies of ``calls`` run at once, skipping those that wrote nothing.

    One refusal or unusable reply among many (prompts, mined samples) is
    logged and skipped; any other error, such as a bad key or the budget,
    cancels the calls still running and stops the run, and so does every call
    writing nothing.
    """

    async def kept(call: Awaitable[T]) -> T | NoOutputError:
        try:
            return await call
        except NoOutputError as error:
            return error

    try:
        async with asyncio.TaskGroup() as group:  # the first error cancels the rest
            tasks = [group.create_task(kept(call)) for call in calls]
    except BaseExceptionGroup as errors:
        raise errors.exceptions[0] from None  # tasks do not nest groups
    replies = [task.result() for task in tasks]
    failed = [r for r in replies if isinstance(r, NoOutputError)]
    written = [r for r in replies if not isinstance(r, NoOutputError)]
    if failed and not written:
        raise failed[0]
    if failed:
        logger.warning(
            f"{len(failed)} of {len(replies)} {what} wrote nothing: {failed[0]}"
        )
    return written


def examples_block(
    texts: Sequence[str],
    codes: npt.NDArray[np.intp],
    classes: Classes,
    indices: npt.NDArray[np.intp],
) -> str:
    """The samples at ``indices``, each with its label, ready for a prompt."""
    return "\n\n".join(
        f"Example {n} (label: {classes[codes[i]]}):\n{texts[i]}"
        for n, i in enumerate(indices, 1)
    )


def class_note(classes: Classes, pos: int) -> str:
    """The sentence that ends a judge's question by saying which class YES means.

    "Is the answer to the task YES?" does not say that the positive class is
    ``'miss'`` of ``'hit'``/``'miss'``, so the note names both classes. It is
    empty when the positive class reads as yes and the other as no: ``yes`` and
    ``no``, ``true`` and ``false`` or ``1`` and ``0`` as text, in any case. Such
    questions, and their cached answers, stay as they were.

    Args:
        classes: The two classes.
        pos: Index of the positive class.
    """
    positive, negative = str(classes[pos]), str(classes[1 - pos])
    if (positive.strip().casefold(), negative.strip().casefold()) in _YES_NO:
        return ""
    return (
        f" Answer YES if the sample belongs to the class '{positive}', and NO if "
        f"it belongs to the class '{negative}'."
    )


async def write_instructions(
    session: LLMSession, models: Sequence[Endpoint], task: str, purpose: str
) -> str:
    """Instructions refined from ``task`` by the generator.

    Args:
        session: The run's session.
        models: The generator's models.
        task: The user's description of the task.
        purpose: What the instructions are for, e.g. ``"questions that split a
            decision tree"``.
    """
    return await session.generate(
        models,
        f"Task description:\n{task}",
        role="generator",
        instructions=_INSTRUCTIONS_WRITER.format(purpose=purpose),
    )
