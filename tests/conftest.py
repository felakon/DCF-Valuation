"""
Synthetic SEC 'companyfacts' payload that mimics the real EDGAR structure, including
the awkward parts the parser must handle:
  * prior-year comparatives repeated in every 10-K (and one restatement)
  * a revenue tag switch (SalesRevenueNet -> RevenueFromContract...)
  * 3-month Q4 facts inside the 10-K that must NOT be read as annual numbers
  * 10-Q year-to-date facts for a TTM calculation
"""
from __future__ import annotations

import pytest

from dcf.market import MarketData

YEARS = [2020, 2021, 2022, 2023, 2024]
REVENUE = {2020: 1000, 2021: 1100, 2022: 1210, 2023: 1300, 2024: 1400}
M = 1e6


def flow(val, year, form="10-K", filed=None, start=None, end=None):
    return {"start": start or f"{year}-01-01", "end": end or f"{year}-12-31", "val": val,
            "accn": f"0000-{filed or year + 1}", "fy": year, "fp": "FY", "form": form,
            "filed": filed or f"{year + 1}-02-15"}


def inst(val, year, form="10-K", filed=None, end=None):
    return {"end": end or f"{year}-12-31", "val": val, "accn": f"0000-{filed or year + 1}",
            "fy": year, "fp": "FY", "form": form, "filed": filed or f"{year + 1}-02-15"}


def annual_flow(per_year: dict, filing_years=YEARS, lookback=2):
    """Each 10-K reports the current and `lookback` prior years."""
    out = []
    for fy in filing_years:
        for y in range(fy - lookback, fy + 1):
            if y in per_year:
                out.append(flow(per_year[y] * M, y, filed=f"{fy + 1}-02-15"))
    return out


def annual_inst(per_year: dict, filing_years=YEARS):
    out = []
    for fy in filing_years:
        for y in (fy - 1, fy):
            if y in per_year:
                out.append(inst(per_year[y] * M, y, filed=f"{fy + 1}-02-15"))
    return out


def build_facts():
    ebit = {y: REVENUE[y] * 0.20 for y in YEARS}
    pretax = {y: ebit[y] - 10 for y in YEARS}
    tax = {y: pretax[y] * 0.20 for y in YEARS}
    ni = {y: pretax[y] - tax[y] for y in YEARS}

    # Revenue: old tag for FY2020-21 filings, new tag afterwards.
    old_rev = annual_flow(REVENUE, filing_years=[2020, 2021])
    new_rev = annual_flow(REVENUE, filing_years=[2023, 2024])
    # Restatement: FY2023 revenue restated to 1305 in the FY2024 10-K.
    for f in new_rev:
        if f["end"] == "2023-12-31" and f["filed"] == "2025-02-15":
            f["val"] = 1305 * M
    # Q4-only fact inside a 10-K (must be ignored for annual data).
    new_rev.append(flow(380 * M, 2024, start="2024-10-01", filed="2025-02-15"))
    # 10-Q YTD facts: 9M 2025 and 9M 2024 comparative, plus a 3-month quarter.
    new_rev += [
        flow(1110 * M, 2025, form="10-Q", start="2025-01-01", end="2025-09-30", filed="2025-10-30"),
        flow(1030 * M, 2025, form="10-Q", start="2024-01-01", end="2024-09-30", filed="2025-10-30"),
        flow(390 * M, 2025, form="10-Q", start="2025-07-01", end="2025-09-30", filed="2025-10-30"),
    ]

    def gaap(units_key, facts):
        return {"label": "x", "units": {units_key: facts}}

    def ytd(tag_val_25, tag_val_24):
        return [
            flow(tag_val_25 * M, 2025, form="10-Q", start="2025-01-01", end="2025-09-30", filed="2025-10-30"),
            flow(tag_val_24 * M, 2025, form="10-Q", start="2024-01-01", end="2024-09-30", filed="2025-10-30"),
        ]

    us_gaap = {
        "SalesRevenueNet": gaap("USD", old_rev),
        "RevenueFromContractWithCustomerExcludingAssessedTax": gaap("USD", new_rev),
        "CostOfRevenue": gaap("USD", annual_flow({y: REVENUE[y] * 0.6 for y in YEARS})),
        "OperatingIncomeLoss": gaap("USD", annual_flow(ebit) + ytd(222, 206)),
        "InterestExpense": gaap("USD", annual_flow({y: 10 for y in YEARS})),
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
            gaap("USD", annual_flow(pretax)),
        "IncomeTaxExpenseBenefit": gaap("USD", annual_flow(tax)),
        "NetIncomeLoss": gaap("USD", annual_flow(ni)),
        "DepreciationDepletionAndAmortization": gaap("USD", annual_flow({y: REVENUE[y] * 0.04 for y in YEARS})),
        "NetCashProvidedByUsedInOperatingActivities": gaap("USD", annual_flow({y: REVENUE[y] * 0.22 for y in YEARS})),
        "PaymentsToAcquirePropertyPlantAndEquipment": gaap("USD", annual_flow({y: REVENUE[y] * 0.05 for y in YEARS})),
        "CashAndCashEquivalentsAtCarryingValue": gaap("USD", annual_inst({y: 200 for y in YEARS})
                                                      + [inst(250 * M, 2025, "10-Q", "2025-10-30", "2025-09-30")]),
        "ShortTermInvestments": gaap("USD", annual_inst({y: 50 for y in YEARS})),
        "AssetsCurrent": gaap("USD", annual_inst({y: 600 for y in YEARS})),
        "LiabilitiesCurrent": gaap("USD", annual_inst({y: 300 for y in YEARS})),
        "LongTermDebtCurrent": gaap("USD", annual_inst({y: 20 for y in YEARS})),
        "CommercialPaper": gaap("USD", annual_inst({y: 30 for y in YEARS})),
        "LongTermDebtNoncurrent": gaap("USD", annual_inst({y: 150 for y in YEARS})
                                       + [inst(140 * M, 2025, "10-Q", "2025-10-30", "2025-09-30")]),
        "StockholdersEquity": gaap("USD", annual_inst({y: 800 for y in YEARS})),
        "Assets": gaap("USD", annual_inst({y: 1500 for y in YEARS})),
        "EarningsPerShareDiluted": gaap("USD/shares", [
            {**f, "val": f["val"] / M / 100} for f in annual_flow(ni)]),
        "WeightedAverageNumberOfDilutedSharesOutstanding": gaap("shares", [
            {**f, "val": 100 * M} for f in annual_flow(REVENUE)]),
    }
    dei = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2025-01-31", "val": 99 * M, "accn": "0000-2025-02-15", "fy": 2024, "fp": "FY",
         "form": "10-K", "filed": "2025-02-15"},
        {"end": "2025-10-20", "val": 98 * M, "accn": "0000-2025-10-30", "fy": 2025, "fp": "Q3",
         "form": "10-Q", "filed": "2025-10-30"},
    ]}}}
    return {"cik": 1234, "entityName": "Testco Inc.", "facts": {"us-gaap": us_gaap, "dei": dei}}


class FakeClient:
    def lookup(self, ticker):
        return {"ticker": ticker.upper(), "cik": 1234, "name": "Testco Inc."}

    def company_facts(self, cik):
        return build_facts()

    def submissions(self, cik):
        return {"sic": "3571", "sicDescription": "Electronic Computers"}

    def latest_filing(self, cik):
        return None


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def market():
    return MarketData(price=20.0, beta=1.1, raw_beta=1.15, risk_free=0.04)
