import numpy as np
import pytest

from src.evaluate_models import bootstrap_ci, token_f1


def test_token_f1_identical_is_one():
    assert token_f1("Amoxicillin 500 mg", "amoxicillin 500 mg") == 1.0


def test_token_f1_no_overlap_is_zero():
    assert token_f1("ceftriaxone", "amoxicillin") == 0.0


def test_token_f1_partial_overlap():
    # 2 shared tokens; precision 2/2, recall 2/4 -> F1 = 2/3
    assert token_f1("amoxicillin 500", "amoxicillin 500 mg daily") == pytest.approx(2 / 3)


def test_token_f1_handles_empty_prediction():
    assert token_f1("", "amoxicillin") == 0.0


def test_bootstrap_ci_contains_mean_and_ignores_nan():
    mean, lo, hi = bootstrap_ci([0, 1, 1, 1, np.nan] * 20)
    assert mean == pytest.approx(0.75)
    assert lo <= mean <= hi
