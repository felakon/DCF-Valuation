from dataclasses import replace

import numpy as np
import pytest

from dcf.classify import ai_score, classify, html_to_text
from dcf.financials import load_financials
from dcf.growth import (GrowthAssumptions, growth_assumptions, growth_margin_grid, implied_growth,
                        run_growth_dcf, trading_multiples, weighted_scenarios)
from dcf.model import default_assumptions, run_dcf
from dcf.report import assumptions_table, export_excel, projection_table

M = 1e6
AI_TEXT = ("Our AI platforms and generative AI accelerators power large language models. " * 50
           + "We sell products. " * 200)


@pytest.fixture
def fin(client):
    return load_financials(client, "TEST")


# ---------------------------------------------------------------- classification
def test_ai_score_counts_terms_not_substrings():
    score, hits, words = ai_score("AI said the chair. Artificial intelligence and machine learning. ai")
    assert hits == 3 and words == 10
    assert score == pytest.approx(3000)


def test_html_to_text():
    raw = "<html><style>x{}</style><p>Revenue&nbsp;grew</p><ix:header>hidden</ix:header></html>"
    assert html_to_text(raw).strip() == "Revenue grew"


def test_classify_routes(fin):
    # SIC 3571 (computers), ~8% growth, no 10-K text -> mature tech -> standard DCF
    c = classify(fin)
    assert c.is_tech and c.profile == "Mature tech" and c.model == "standard"
    # Heavy AI language in a tech company -> growth model
    c = classify(fin, AI_TEXT)
    assert c.profile == "AI / hyper-growth tech" and c.model == "growth" and c.ai_score > 20
    # Same text, but a beverage company -> stays on the standard model
    fin.sic, fin.sic_description = "2080", "Beverages"
    c = classify(fin, AI_TEXT)
    assert c.profile == "Traditional" and c.model == "standard"
    assert any("not a tech business" in r for r in c.reasons)


# ---------------------------------------------------------------- growth model
def test_growth_path_two_stages():
    g = GrowthAssumptions(projection_years=10, stage1_years=5, initial_growth=0.40, terminal_growth=0.02)
    path = g.growth_path()
    assert path[0] == pytest.approx(0.40)
    assert path[4] == pytest.approx(0.02 + 0.5 * 0.38)   # half the excess gone by year 5
    assert np.all(np.diff(path) < 0) and path[-1] > 0.02


def test_growth_dcf_hand_calculation(fin, market):
    g = GrowthAssumptions(projection_years=3, stage1_years=2, initial_growth=0.30, terminal_growth=0.02,
                          ebit_margin=0.20, tax_rate=0.25, sales_to_capital=2.0, terminal_roic=0.20,
                          risk_free=0.04, beta=1.5, mature_beta=1.0, erp=0.05,
                          pre_tax_cost_of_debt=0.06, mid_year=False)
    res = run_growth_dcf(fin, market, g, use_ttm=False)

    we = 20 * 98 * M / (20 * 98 * M + 200 * M)
    kd = 0.06 * 0.75
    w0 = we * (0.04 + 1.5 * 0.05) + (1 - we) * kd
    wm = we * (0.04 + 1.0 * 0.05) + (1 - we) * kd
    rates = [w0, w0, wm]                               # stage 1 = 2 years, converged by year 3
    growth = g.growth_path()                           # [0.30, 0.16, 0.09]
    assert growth == pytest.approx([0.30, 0.16, 0.09])
    rev, prev, cum, ev = 1400 * M, 1400 * M, 1.0, 0.0
    for gr, r in zip(growth, rates):
        rev = prev * (1 + gr)
        fcff = rev * 0.20 * 0.75 - (rev - prev) / 2.0
        cum *= 1 + r
        ev += fcff / cum
        prev = rev
    fcff_next = rev * 1.02 * 0.20 * 0.75 * (1 - 0.02 / 0.20)
    ev += fcff_next / (wm - 0.02) / cum
    assert res.enterprise_value == pytest.approx(ev)
    assert res.wacc["terminal_wacc"] == pytest.approx(wm)
    assert res.projection.loc["WACC"].iloc[0] == pytest.approx(w0)
    assert res.value_per_share == pytest.approx((ev - 200 * M + 250 * M) / (98 * M))


def test_growth_defaults_and_overrides(fin, market):
    cls = classify(fin, AI_TEXT)
    g = growth_assumptions(fin, market, cls, consensus_growth=0.35, consensus_note="consensus")
    assert g.projection_years == 10 and g.stage1_years == 5
    assert g.initial_growth == pytest.approx(0.35)
    assert g.sources["initial_growth"].startswith("consensus")
    assert 0.8 <= g.sales_to_capital <= 5.0
    g2 = growth_assumptions(fin, market, cls, terminal_roic=0.30)
    assert g2.terminal_roic == 0.30 and g2.sources["terminal_roic"] == "manual override"
    # no consensus -> history blend, capped
    assert -0.05 <= g2.initial_growth <= 0.60


def test_growth_model_values_more_than_standard_for_fast_grower(fin, market):
    std = run_dcf(fin, market, replace(default_assumptions(fin, market), initial_growth=0.35))
    gro = run_growth_dcf(fin, market, replace(growth_assumptions(fin, market), initial_growth=0.35))
    assert gro.value_per_share > std.value_per_share


def test_reverse_dcf_hits_market_price(fin, market):
    """At the implied growth rate, the model value equals the market price.
    (The price also sets the WACC weights, so it is solved jointly.)"""
    for price in (40.0, 90.0):
        mkt = replace(market, price=price)
        g = growth_assumptions(fin, mkt)
        g_star = implied_growth(fin, mkt, g)
        assert run_growth_dcf(fin, mkt, replace(g, initial_growth=g_star)).value_per_share == \
            pytest.approx(price, rel=1e-3)
        a = default_assumptions(fin, mkt)
        a_star = implied_growth(fin, mkt, a)
        assert run_dcf(fin, mkt, replace(a, initial_growth=a_star)).value_per_share == \
            pytest.approx(price, rel=1e-3)
    # A price no plausible growth rate can justify returns None instead of a wrong number.
    assert implied_growth(fin, replace(market, price=1e6), growth_assumptions(fin, market)) is None
    assert implied_growth(fin, replace(market, price=15.0), growth_assumptions(fin, market)) is None


def test_scenarios_grid_multiples_and_reports(fin, market, tmp_path):
    g = growth_assumptions(fin, market)
    sc = weighted_scenarios(fin, market, g)
    v = sc["Value / share"]
    assert v["Bear"] < v["Base"] < v["Bull"]
    assert v["Probability-weighted"] == pytest.approx(0.25 * v["Bear"] + 0.5 * v["Base"] + 0.25 * v["Bull"])

    grid = growth_margin_grid(fin, market, g)
    assert (grid.diff(axis=0).dropna() > 0).all().all()     # more growth -> more value
    assert (grid.diff(axis=1).iloc[:, 1:] > 0).all().all()  # more margin -> more value

    res = run_growth_dcf(fin, market, g)
    mult = trading_multiples(fin, market, {"Growth DCF": res})
    ev_mkt = 20 * 98 * M + 140 * M - 250 * M               # TTM base: debt 140, cash 250
    assert mult.loc["EV / Revenue", "Market today"] == pytest.approx(ev_mkt / (1480 * M))

    assert "Terminal ROIC" in assumptions_table(res).index
    assert projection_table(res).loc["WACC"].iloc[0].endswith("%")
    path = export_excel(fin, res, grid, str(tmp_path / "x.xlsx"), extra_sheets={"Scenarios": sc})
    assert (tmp_path / "x.xlsx").exists() and path.endswith("x.xlsx")
