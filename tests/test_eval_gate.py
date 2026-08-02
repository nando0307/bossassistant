"""The gate's statistics, which are the part that must not be hand-wavy."""
from __future__ import annotations

from scripts.eval_gate import mcnemar_exact_p, newcombe_difference, wilson_interval


def test_wilson_interval_stays_in_bounds_at_the_extremes() -> None:
    """The normal approximation gives impossible intervals near 0 and 1.

    A pass rate of 75/75 is exactly where a portfolio eval sits, so the
    interval has to behave there.
    """
    low, high = wilson_interval(75, 75)
    assert 0.0 <= low <= 1.0 and high == 1.0
    assert low > 0.9

    low, high = wilson_interval(0, 75)
    assert low == 0.0 and 0.0 <= high <= 1.0


def test_wilson_interval_narrows_with_sample_size() -> None:
    narrow = wilson_interval(900, 1000)
    wide = wilson_interval(9, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_newcombe_difference_brackets_zero_for_identical_rates() -> None:
    low, high = newcombe_difference(60, 75, 60, 75)
    assert low < 0 < high


def test_newcombe_difference_is_negative_for_a_large_drop() -> None:
    low, high = newcombe_difference(30, 75, 70, 75)
    assert high < 0, "a 40-case drop should exclude zero"


def test_mcnemar_ignores_cases_that_did_not_change() -> None:
    """The whole point of pairing: concordant cases carry no information.

    A two-proportion test sees only the aggregate rate and cannot distinguish
    "ten cases regressed" from "ten regressed and ten improved".
    """
    assert mcnemar_exact_p(0, 0) == 1.0
    # Same aggregate rate, very different stories.
    one_sided = mcnemar_exact_p(10, 0)
    balanced = mcnemar_exact_p(10, 10)
    assert one_sided < 0.01
    assert balanced > 0.3


def test_mcnemar_does_not_fire_on_small_wobbles() -> None:
    """Provider flake has cost this project 3-5 cases on identical runs.

    Those must not trip the gate, or it gets switched off.
    """
    assert mcnemar_exact_p(3, 1) > 0.05
    assert mcnemar_exact_p(5, 2) > 0.05


def test_mcnemar_fires_on_a_consistent_one_way_regression() -> None:
    assert mcnemar_exact_p(8, 0) < 0.05
    assert mcnemar_exact_p(12, 2) < 0.05


def test_mcnemar_is_one_sided() -> None:
    """Improvements must never fail the gate."""
    assert mcnemar_exact_p(0, 12) == 1.0
