
import numpy as np
import pandas as pd
import pytest

from dcf.financials import load_financials
from dcf.market import blume_adjust, regression_beta
from dcf.model import (Assumptions, default_assumptions, run_dcf, scenarios,
                       wacc_growth_grid)

M = 1e6


@pytest.fixture
def fin(client):
    return load_financials(client, "TEST")


def test_dcf_matches_hand_calculation(fin, market):
    """Two-year, end-of-year DCF checked against an explicit calculation."""
    a = Assumptions(projection_years=2, revenue_growth=[0.10, 0.05], terminal_growth=0.02,
                    ebit_margin=0.20, tax_rate=0.25, da_pct=0.04, capex_pct=0.05,
                    nwc_pct=0.10, wacc_override=0.09, mid_year=False)
    res = run_dcf(fin, market, a, use_ttm=False)

    r0 = 1400 * M
    r1, r2 = r0 * 1.10, r0 * 1.10 * 1.05
    f1 = r1 * 0.20 * 0.75 + r1 * 0.04 - r1 * 0.05 - (r1 - r0) * 0.10
    f2 = r2 * 0.20 * 0.75 + r2 * 0.04 - r2 * 0.05 - (r2 - r1) * 0.10
    tv = f2 * 1.02 / (0.09 - 0.02)
    ev = f1 / 1.09 + f2 / 1.09 ** 2 + tv / 1.09 ** 2
    equity = ev - 200 * M + 250 * M          # debt 200, cash 200 + STI 50
    assert res.enterprise_value == pytest.approx(ev)
    assert res.equity_value == pytest.approx(equity)
    assert res.value_per_share == pytest.approx(equity / (98 * M))
    assert res.upside == pytest.approx(res.value_per_share / market.price - 1)


def test_wacc_components(fin, market):
    a = default_assumptions(fin, market, use_ttm=False, erp=0.05)
    res = run_dcf(fin, market, a, use_ttm=False)
    w = res.wacc
    assert w["cost_of_equity"] == pytest.approx(0.04 + 1.1 * 0.05)
    assert w["weight_equity"] + w["weight_debt"] == pytest.approx(1)
    expected = (w["weight_equity"] * w["cost_of_equity"]
                + w["weight_debt"] * a.pre_tax_cost_of_debt * (1 - a.tax_rate))
    assert w["wacc"] == pytest.approx(expected)


def test_default_assumptions_from_history(fin, market):
    a = default_assumptions(fin, market)
    assert a.ebit_margin == pytest.approx(0.20, abs=0.002)
    assert a.tax_rate == pytest.approx(0.20)
    assert a.capex_pct == pytest.approx(0.05, abs=0.001)
    assert -0.05 <= a.initial_growth <= 0.25
    assert a.sources["beta"].startswith("5-yr")
    a2 = default_assumptions(fin, market, terminal_growth=0.03, erp=a.erp)
    assert a2.sources["erp"].startswith("Default")
    assert a2.terminal_growth == 0.03 and a2.sources["terminal_growth"] == "manual override"


def test_growth_fades_to_terminal():
    a = Assumptions(projection_years=5, initial_growth=0.12, terminal_growth=0.02)
    path = a.growth_path()
    assert path[0] == pytest.approx(0.12)
    assert np.all(np.diff(path) < 0) and path[-1] > 0.02


def test_wacc_must_exceed_growth(fin, market):
    a = Assumptions(wacc_override=0.02, terminal_growth=0.03)
    with pytest.raises(ValueError):
        run_dcf(fin, market, a)


def test_sensitivity_monotonic(fin, market):
    a = default_assumptions(fin, market)
    res = run_dcf(fin, market, a)
    grid = wacc_growth_grid(fin, market, a, res)
    assert grid.shape == (5, 5)
    assert (grid.diff(axis=0).dropna() < 0).all().all()   # higher WACC -> lower value
    assert (grid.diff(axis=1).iloc[:, 1:] > 0).all().all() # higher g -> higher value
    centre = grid.iloc[2, 2]
    assert centre == pytest.approx(res.value_per_share, rel=1e-3)


def test_scenarios_ordered(fin, market):
    a = default_assumptions(fin, market)
    s = scenarios(fin, market, a)
    assert s.loc["Bear", "Value / share"] < s.loc["Base", "Value / share"] < s.loc["Bull", "Value / share"]


def test_regression_beta_recovers_true_beta():
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0.01, 0.04, 60))
    stock = 1.3 * mkt + rng.normal(0, 0.005, 60)
    beta, r2, n = regression_beta(stock, mkt)
    assert beta == pytest.approx(1.3, abs=0.05) and r2 > 0.9 and n == 60
    assert blume_adjust(1.3) == pytest.approx(0.67 * 1.3 + 0.33)
