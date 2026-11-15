"""Foreign filers (20-F / IFRS / EUR), Yahoo Finance fallback, and currency handling."""
import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import FakeClient, build_facts
from dcf.classify import classify
from dcf.financials import load_financials
from dcf.growth import growth_assumptions, run_growth_dcf
from dcf.market import MarketData, fx_rate, normalise_currency
from dcf.model import default_assumptions, money, run_dcf
from dcf.yahoo import financials_from_frames, has_exchange_suffix

M = 1e6
GAAP_TO_IFRS = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
    "SalesRevenueNet": "RevenueFromContractsWithCustomers",
    "CostOfRevenue": "CostOfSales",
    "OperatingIncomeLoss": "ProfitLossFromOperatingActivities",
    "InterestExpense": "FinanceCosts",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
        "ProfitLossBeforeTax",
    "IncomeTaxExpenseBenefit": "IncomeTaxExpenseContinuingOperations",
    "NetIncomeLoss": "ProfitLossAttributableToOwnersOfParent",
    "DepreciationDepletionAndAmortization": "DepreciationAndAmortisationExpense",
    "NetCashProvidedByUsedInOperatingActivities": "CashFlowsFromUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment": "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
    "CashAndCashEquivalentsAtCarryingValue": "CashAndCashEquivalents",
    "ShortTermInvestments": "CurrentInvestments",
    "AssetsCurrent": "CurrentAssets",
    "LiabilitiesCurrent": "CurrentLiabilities",
    "LongTermDebtCurrent": "CurrentPortionOfLongtermBorrowings",
    "CommercialPaper": "ShorttermBorrowings",
    "LongTermDebtNoncurrent": "NoncurrentPortionOfNoncurrentBorrowings",
    "StockholdersEquity": "EquityAttributableToOwnersOfParent",
    "Assets": "Assets",
    "EarningsPerShareDiluted": "DilutedEarningsLossPerShare",
    "WeightedAverageNumberOfDilutedSharesOutstanding": "AdjustedWeightedAverageShares",
}
UNIT = {"USD": "EUR", "USD/shares": "EUR/shares", "shares": "shares"}


def ifrs_facts():
    """The US test company re-expressed as a 20-F filer reporting IFRS in EUR (no 10-Qs)."""
    gaap = build_facts()
    ifrs = {}
    for tag, node in gaap["facts"]["us-gaap"].items():
        units = {}
        for unit, rows in node["units"].items():
            kept = [{**r, "form": "20-F"} for r in copy.deepcopy(rows) if r["form"] == "10-K"]
            units[UNIT[unit]] = kept
        ifrs[GAAP_TO_IFRS[tag]] = {"label": "x", "units": units}
    return {"cik": 937966, "entityName": "Lithoco N.V.",
            "facts": {"ifrs-full": ifrs, "dei": gaap["facts"]["dei"]}}


class IFRSClient(FakeClient):
    def company_facts(self, cik):
        return ifrs_facts()

    def submissions(self, cik):
        return {"sic": "3559", "sicDescription": "Special Industry Machinery"}


@pytest.fixture
def eur_market():
    return MarketData(price=20.0, beta=1.1, raw_beta=1.15, risk_free=0.03, currency="EUR",
                      price_currency="EUR")


def test_20f_ifrs_company_loads_in_eur():
    fin = load_financials(IFRSClient(), "LITH")
    assert fin.taxonomy == "ifrs-full" and fin.currency == "EUR"
    assert list(fin.annual.columns) == ["FY2020", "FY2021", "FY2022", "FY2023", "FY2024"]
    a = fin.annual / M
    assert a.loc["revenue", "FY2024"] == 1400 and a.loc["revenue", "FY2020"] == 1000
    assert a.loc["ebit", "FY2024"] == pytest.approx(280)
    assert a.loc["total_debt", "FY2024"] == pytest.approx(200)
    assert fin.ttm is None                      # foreign filers have no 10-Qs
    assert any("20-F" in w for w in fin.warnings)


def test_eur_company_runs_both_models(eur_market):
    fin = load_financials(IFRSClient(), "LITH")
    std = run_dcf(fin, eur_market, default_assumptions(fin, eur_market))
    gro = run_growth_dcf(fin, eur_market, growth_assumptions(fin, eur_market))
    assert std.currency == gro.currency == "EUR"
    assert std.summary()["Intrinsic value / share"].startswith("€")
    assert "Enterprise value (EURm)" in std.summary().index
    assert np.isfinite(std.value_per_share) and np.isfinite(gro.value_per_share)


# ---------------------------------------------------------------- Yahoo fallback
def yahoo_frames():
    dates = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"])
    inc = pd.DataFrame({
        "Total Revenue": [100, 120, 150, 180], "Cost Of Revenue": [50, 60, 72, 85],
        "Operating Income": [20, 26, 35, 45], "Pretax Income": [19, 25, 34, 44],
        "Tax Provision": [4, 5, 7, 9], "Net Income Common Stockholders": [15, 20, 27, 35],
        "Research And Development": [15, 18, 22, 27], "Diluted Average Shares": [10, 10, 10, 10],
        "Reconciled Depreciation": [3, 4, 5, 6], "Diluted EPS": [1.5, 2.0, 2.7, 3.5],
    }, index=dates).T.iloc[:, ::-1] * 1.0
    inc.loc[inc.index != "Diluted EPS"] *= M
    bal = pd.DataFrame({
        "Cash And Cash Equivalents": [30, 35, 40, 50], "Current Assets": [80, 90, 100, 120],
        "Current Liabilities": [40, 45, 50, 55], "Current Debt": [5, 5, 5, 5],
        "Long Term Debt": [20, 20, 20, 20], "Stockholders Equity": [60, 70, 85, 100],
        "Total Assets": [150, 170, 190, 220], "Ordinary Shares Number": [10, 10, 10, 10],
    }, index=dates).T.iloc[:, ::-1] * M
    cf = pd.DataFrame({"Operating Cash Flow": [22, 28, 36, 46], "Capital Expenditure": [-5, -6, -8, -9]},
                      index=dates).T.iloc[:, ::-1] * M
    info = {"longName": "Euro Software SE", "financialCurrency": "EUR", "currency": "EUR",
            "sector": "Technology", "industry": "Software - Application", "sharesOutstanding": 10 * M}
    return inc, bal, cf, info


def test_yahoo_frames_map_to_standard_items(eur_market):
    inc, bal, cf, info = yahoo_frames()
    fin = financials_from_frames("ESW.DE", inc, bal, cf, info)
    assert fin.source == "Yahoo Finance" and fin.currency == "EUR" and fin.sic is None
    assert list(fin.annual.columns) == ["FY2022", "FY2023", "FY2024", "FY2025"]
    a = fin.annual / M
    assert a.loc["revenue", "FY2025"] == 180
    assert a.loc["capex", "FY2025"] == 9                      # sign flipped
    assert a.loc["total_debt", "FY2025"] == 25
    assert a.loc["nwc", "FY2025"] == pytest.approx((120 - 50) - (55 - 5))
    # Classified from Yahoo's sector/industry, growth ~20% -> growth tech
    cls = classify(fin)
    assert cls.is_tech and cls.profile == "Growth tech" and cls.model == "growth"
    res = run_growth_dcf(fin, eur_market, growth_assumptions(fin, eur_market, cls))
    assert res.currency == "EUR" and np.isfinite(res.value_per_share)


def test_exchange_suffix_detection():
    assert has_exchange_suffix("ASML.AS") and has_exchange_suffix("MC.PA") and has_exchange_suffix("SHEL.L")
    assert not has_exchange_suffix("ASML") and not has_exchange_suffix("BRK-B")


# ---------------------------------------------------------------- currency
def test_currency_helpers():
    assert normalise_currency(2500.0, "GBp") == (25.0, "GBP")
    assert normalise_currency(10.0, "EUR") == (10.0, "EUR")
    assert fx_rate("EUR", "EUR") == 1.0
    quotes = {"USDEUR=X": 0.92}
    assert fx_rate("USD", "EUR", lookup=quotes.__getitem__) == 0.92
    assert money(1234.5, "EUR") == "€1,234.50" and money(3, "SEK", 0) == "SEK 3"


def test_market_conversion_with_manual_price(monkeypatch):
    """A USD-traded ADR of a EUR reporter: price is converted into EUR."""
    import dcf.market as m
    monkeypatch.setattr(m, "fx_rate", lambda frm, to, lookup=None: 0.9 if (frm, to) == ("USD", "EUR") else 1.0)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", None)   # force offline path
    mkt = m.fetch_market_data("LITH", price=100.0, beta=1.2, risk_free=0.03,
                              reporting_currency="EUR", price_currency="USD")
    assert mkt.price == pytest.approx(90.0) and mkt.currency == "EUR" and mkt.price_original == 100.0
    assert replace(mkt, price=1.0).fx_rate == 0.9
