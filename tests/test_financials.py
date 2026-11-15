import numpy as np
import pytest

from dcf.financials import load_financials

M = 1e6


@pytest.fixture
def fin(client):
    return load_financials(client, "TEST", years=5)


def test_fiscal_years_and_revenue(fin):
    assert list(fin.annual.columns) == ["FY2020", "FY2021", "FY2022", "FY2023", "FY2024"]
    rev = fin.annual.loc["revenue"] / M
    # FY2020 only exists under the old tag -> tag merge works.
    assert rev["FY2020"] == 1000
    # FY2023 was restated in the latest 10-K -> latest filing wins.
    assert rev["FY2023"] == 1305
    # The Q4 (3-month) fact must not replace the annual number.
    assert rev["FY2024"] == 1400


def test_derived_items(fin):
    a = fin.annual / M
    assert a.loc["gross_profit", "FY2024"] == pytest.approx(1400 * 0.4)
    assert a.loc["ebitda", "FY2024"] == pytest.approx(1400 * 0.24)
    # debt = current LTD 20 + commercial paper 30 + non-current 150
    assert a.loc["total_debt", "FY2024"] == pytest.approx(200)
    # NWC = (600 - 200 cash - 50 STI) - (300 - 50 current debt) = 100
    assert a.loc["nwc", "FY2024"] == pytest.approx(100)
    assert a.loc["capex", "FY2024"] > 0
    assert fin.annual.loc["tax_rate", "FY2024"] == pytest.approx(0.20)


def test_ttm(fin):
    assert fin.ttm_end == "2025-09-30"
    assert fin.ttm["revenue"] / M == pytest.approx(1400 + 1110 - 1030)
    assert fin.ttm["ebit"] / M == pytest.approx(280 + 222 - 206)
    assert fin.ttm["cash"] / M == pytest.approx(250)
    # A weighted share count is carried, not summed.
    assert fin.ttm["diluted_shares"] / M == pytest.approx(100)


def test_shares_from_latest_cover_page(fin):
    assert fin.shares_outstanding == 98 * M
    assert fin.shares_date == "2025-10-20"


def test_statement_formatting(fin):
    inc = fin.statement("Income Statement")
    assert "Revenue" in inc.index
    assert inc.loc["Revenue", "FY2024"] == pytest.approx(1400)
    assert inc.columns[-1].startswith("TTM")
    assert not fin.is_financial


def test_no_ttm_option(client):
    fin = load_financials(client, "TEST", include_ttm=False)
    assert fin.ttm is None and np.isfinite(fin.annual.loc["revenue"]).all()
