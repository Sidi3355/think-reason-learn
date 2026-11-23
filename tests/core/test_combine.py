"""The shared combiner: F-beta thresholds and the logistic fit."""

from __future__ import annotations

import numpy as np

from think_reason_learn.core._combine import best_threshold, fit_logistic


class TestCombine:
    def test_best_threshold(self) -> None:
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        threshold = best_threshold(scores, np.array([False, False, True, True]), 1.0)
        assert threshold == 0.5  # midway between 0.4 and 0.6

    def test_beta_moves_the_threshold(self) -> None:
        scores = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.9])
        positive = np.array([False, True, False, True, False, True])
        # A cut at 0.9 lets no negative in and finds one positive of three; a cut
        # at 0.3 finds all three and lets two negatives in.
        precise = best_threshold(scores, positive, 0.5)
        thorough = best_threshold(scores, positive, 2.0)
        assert np.allclose([precise, thorough], [0.75, 0.2])

    def test_of_equally_good_cuts_the_lowest_wins(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        positive = np.array([False, True, False, False, True])
        # F1 is 2/3 at 0.2 (both positives, two negatives) and at 0.9 (one
        # positive, no negative).
        threshold = best_threshold(scores, positive, 1.0)
        assert np.isclose(threshold, 0.15)  # midway between 0.1 and 0.2

    def test_logistic_learns_a_separable_feature(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.random(80) > 0.5
        features = np.column_stack([y.astype(float), rng.random(80)])
        fitted = fit_logistic(
            features,
            y,
            l1_ratios=(0.5,),
            cs=4,
            cv=3,
            beta=1.0,
            class_weight=None,
            random_state=0,
        )
        assert fitted.coef[0] > abs(fitted.coef[1])
        assert ((fitted.scores(features) >= fitted.threshold) == y).mean() == 1.0

    def test_the_threshold_is_tuned_on_out_of_fold_scores(self) -> None:
        # 18 rows share a trait and are positive; 20 lack it and are negative. Two
        # more positives lack it too, and each has a column of its own that only
        # a model fitted on that row can use.
        trait = np.arange(40) < 18
        marked = np.arange(40) >= 38
        y = trait | marked
        features = np.column_stack([trait, np.eye(40)[:, marked]]).astype(float)
        # L2 with C = 1 keeps the weight of a one-row column under 1, so the
        # fitted model lifts the two marked rows only part of the way up.
        fitted = fit_logistic(
            features,
            y,
            l1_ratios=(0.0,),
            cs=(1.0,),
            cv=4,
            beta=1.0,
            class_weight=None,
            random_state=0,
        )
        own = fitted.scores(features)
        # Tuned on the fitted rows' own scores, the cut would fall just under the
        # two rows the model memorised and find every positive.
        assert best_threshold(own, y, 1.0) < own[marked].min()
        # Out of fold they score like any row without the trait, so the cut
        # stays between the two groups, above them.
        assert own[marked].max() < fitted.threshold <= own[trait].min()


# A local model's answers to five rules for 24 loan applicants (7 defaulted): the
# first rule is right on 6 of the 7. One 2-fold split made every weight look
# useless, so Policy Induction weighed no rule and flagged everyone.
RULE_ANSWERS = np.array(
    [
        [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1], [1, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1],
    ],
    dtype=float,
)  # fmt: skip
DEFAULTED = np.array(
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    dtype=bool,
)


def test_a_small_dataset_keeps_its_one_good_rule() -> None:
    combiner = fit_logistic(
        RULE_ANSWERS,
        DEFAULTED,
        l1_ratios=(1.0,),
        cv=2,
        beta=0.5,
        class_weight=None,
        random_state=42,
    )
    assert combiner.coef[0] > 1
