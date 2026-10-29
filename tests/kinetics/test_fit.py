"""Üstel bozunma kinetik uydurma testleri."""

import numpy as np
import pytest

from bozunma.kinetics.fit import fit_first_order


def test_fit_first_order_exact_synthetic() -> None:
    # Model: C(t) = c0 * exp(-k * t)
    # c0 = 100.0, k = 0.05
    c0_true = 100.0
    k_true = 0.05

    t = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 60.0])
    c = c0_true * np.exp(-k_true * t)

    res = fit_first_order(t, c)

    assert "c0" in res
    assert "k" in res
    assert "k_err" in res
    assert res["c0"] == pytest.approx(c0_true, rel=1e-3)
    assert res["k"] == pytest.approx(k_true, rel=1e-3)
    assert res["k_err"] >= 0.0


def test_fit_first_order_noisy() -> None:
    np.random.seed(42)
    c0_true = 50.0
    k_true = 0.02
    t = np.linspace(0, 100, 20)
    c = c0_true * np.exp(-k_true * t) + np.random.normal(0, 0.5, len(t))

    res = fit_first_order(t, c)
    assert res["c0"] == pytest.approx(c0_true, rel=0.1)
    assert res["k"] == pytest.approx(k_true, rel=0.1)
