"""
Turn raw SEC XBRL "company facts" into clean, standardized financial statements.

Why this is non-trivial
-----------------------
* Every 10-K repeats prior years as comparatives, so the same fiscal year shows up
  in several filings (sometimes restated). We key every value on its *period end
  date* and keep the most recently filed number.
* Companies tag the same line item differently, and switch tags over time
  (e.g. ``SalesRevenueNet`` -> ``RevenueFromContractWithCustomerExcludingAssessedTax``).
  Each standardized line item has an ordered list of candidate US-GAAP tags,
  merged per period.
* 10-Ks also contain 3-month (Q4) facts. Flow items are restricted to ~12-month
  durations so quarters never leak into annual numbers.
* Trailing-twelve-months (TTM) = last fiscal year + current YTD - prior-year YTD,
  using the latest 10-Q, so the model can run on the freshest data available.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sec import Filing, SECClient

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A",      # US companies
                "20-F", "20-F/A", "40-F", "40-F/A"}       # foreign private issuers (e.g. ASML, SAP)
QUARTERLY_FORMS = {"10-Q", "10-Q/A"}

# ----------------------------------------------------------------------------
# Standardized line items -> candidate US-GAAP tags, in priority order.
# ----------------------------------------------------------------------------
FLOW_TAGS: dict[str, list[str]] = {
    # Income statement
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ],
    "cogs": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
    ],
    "gross_profit": ["GrossProfit"],
    "rnd": ["ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "operating_income": ["OperatingIncomeLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating",
                         "InterestExpenseDebt", "InterestAndDebtExpense"],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    # Cash flow statement
    "da": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndOther",
        "Depreciation",
    ],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements"],
    "dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
}

PER_SHARE_TAGS = {"eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"]}
SHARE_FLOW_TAGS = {"diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding",
                                      "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"]}

INSTANT_TAGS: dict[str, list[str]] = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "Cash"],
    "st_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent",
                       "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "inventory": ["InventoryNet"],
    "current_assets": ["AssetsCurrent"],
    "lt_investments": ["MarketableSecuritiesNoncurrent", "LongTermInvestments",
                       "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent"],
    "ppe_net": ["PropertyPlantAndEquipmentNet"],
    "total_assets": ["Assets"],
    "accounts_payable": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "debt_current_total": ["DebtCurrent"],
    "ltd_current": ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "st_borrowings": ["CommercialPaper", "ShortTermBorrowings", "OtherShortTermBorrowings"],
    "ltd_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
                       "LongTermNotesPayable"],
    "ltd_total": ["LongTermDebt"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "minority_interest": ["MinorityInterest"],
}

# ----------------------------------------------------------------------------
# IFRS equivalents (taxonomy "ifrs-full") for foreign companies filing a 20-F,
# e.g. ASML, SAP, Novo Nordisk, Shell, TotalEnergies. Values are in the company's
# reporting currency (usually EUR).
# ----------------------------------------------------------------------------
IFRS_FLOW_TAGS: dict[str, list[str]] = {
    "revenue": ["Revenue", "RevenueFromContractsWithCustomers", "RevenueFromSaleOfGoods"],
    "cogs": ["CostOfSales"],
    "gross_profit": ["GrossProfit"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "operating_income": ["ProfitLossFromOperatingActivities"],
    "interest_expense": ["FinanceCosts", "InterestExpense"],
    "pretax_income": ["ProfitLossBeforeTax"],
    "income_tax": ["IncomeTaxExpenseContinuingOperations"],
    "net_income": ["ProfitLossAttributableToOwnersOfParent", "ProfitLoss"],
    "da": ["DepreciationAndAmortisationExpense",
           "DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss",
           "AdjustmentsForDepreciationAndAmortisationExpense", "DepreciationExpense"],
    "sbc": ["AdjustmentsForSharebasedPayments", "ExpenseFromSharebasedPaymentTransactionsWithEmployees"],
    "cfo": ["CashFlowsFromUsedInOperatingActivities"],
    "capex": ["PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
              "PurchaseOfPropertyPlantAndEquipment"],
    "dividends": ["DividendsPaidClassifiedAsFinancingActivities", "DividendsPaid"],
    "buybacks": ["PaymentsToAcquireOrRedeemEntitysShares", "PurchaseOfTreasuryShares"],
}
IFRS_PER_SHARE_TAGS = {"eps_diluted": ["DilutedEarningsLossPerShare",
                                       "DilutedEarningsLossPerShareFromContinuingOperations"]}
IFRS_SHARE_FLOW_TAGS = {"diluted_shares": ["AdjustedWeightedAverageShares", "WeightedAverageShares"]}
IFRS_INSTANT_TAGS: dict[str, list[str]] = {
    "cash": ["CashAndCashEquivalents"],
    "st_investments": ["CurrentInvestments", "ShorttermDepositsNotClassifiedAsCashEquivalents",
                       "OtherCurrentFinancialAssets"],
    "receivables": ["TradeAndOtherCurrentReceivables", "CurrentTradeReceivables"],
    "inventory": ["Inventories"],
    "current_assets": ["CurrentAssets"],
    "lt_investments": ["NoncurrentInvestments", "OtherNoncurrentFinancialAssets"],
    "ppe_net": ["PropertyPlantAndEquipment"],
    "total_assets": ["Assets"],
    "accounts_payable": ["TradeAndOtherCurrentPayables", "TradeAndOtherCurrentPayablesToTradeSuppliers"],
    "current_liabilities": ["CurrentLiabilities"],
    "debt_current_total": ["CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings"],
    "ltd_current": ["CurrentPortionOfLongtermBorrowings"],
    "st_borrowings": ["ShorttermBorrowings"],
    "ltd_noncurrent": ["NoncurrentPortionOfNoncurrentBorrowings", "LongtermBorrowings"],
    "ltd_total": [],
    "total_liabilities": ["Liabilities"],
    "equity": ["EquityAttributableToOwnersOfParent", "Equity"],
    "minority_interest": ["NoncontrollingInterests"],
}

TAG_SETS = {
    "us-gaap": (FLOW_TAGS, PER_SHARE_TAGS, SHARE_FLOW_TAGS, INSTANT_TAGS),
    "ifrs-full": (IFRS_FLOW_TAGS, IFRS_PER_SHARE_TAGS, IFRS_SHARE_FLOW_TAGS, IFRS_INSTANT_TAGS),
}


# Rows shown in each statement (after derivation), with display labels.
STATEMENT_LAYOUT = {
    "Income Statement": {
        "revenue": "Revenue", "cogs": "Cost of revenue", "gross_profit": "Gross profit",
        "rnd": "R&D", "sga": "SG&A", "ebit": "Operating income (EBIT)",
        "da": "D&A", "ebitda": "EBITDA", "interest_expense": "Interest expense",
        "pretax_income": "Pre-tax income", "income_tax": "Income tax",
        "net_income": "Net income", "eps_diluted": "Diluted EPS",
        "diluted_shares": "Diluted shares",
    },
    "Balance Sheet": {
        "cash": "Cash & equivalents", "st_investments": "Short-term investments",
        "receivables": "Receivables", "inventory": "Inventory",
        "current_assets": "Total current assets",
        "lt_investments": "Long-term investments", "ppe_net": "PP&E (net)",
        "total_assets": "Total assets", "accounts_payable": "Accounts payable",
        "current_liabilities": "Total current liabilities", "total_debt": "Total debt",
        "total_liabilities": "Total liabilities", "equity": "Shareholders' equity",
        "minority_interest": "Minority interest", "nwc": "Operating working capital",
    },
    "Cash Flow": {
        "cfo": "Cash from operations", "capex": "Capital expenditures",
        "fcf": "Free cash flow (CFO - CapEx)", "sbc": "Stock-based comp",
        "dividends": "Dividends paid", "buybacks": "Share buybacks",
    },
}


# ----------------------------------------------------------------------------
# Low-level fact extraction
# ----------------------------------------------------------------------------
def _facts_frame(facts: dict, tag: str, unit_pref: tuple[str, ...]) -> pd.DataFrame:
    """All reported values for one us-gaap tag as a DataFrame (one row per fact)."""
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
    if not node:
        return pd.DataFrame()
    units = node.get("units", {})
    unit = next((u for u in unit_pref if u in units), None)
    if unit is None:
        return pd.DataFrame()
    df = pd.DataFrame(units[unit])
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["start"] = pd.to_datetime(df["start"]) if "start" in df else pd.NaT
    df["filed"] = pd.to_datetime(df["filed"])
    df["days"] = (df["end"] - df["start"]).dt.days
    return df


def _latest_per_end(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values("filed").drop_duplicates(subset="end", keep="last")
    return df.set_index("end")["val"].astype(float).sort_index()


def _annual_series(facts, tags, instant: bool, unit_pref) -> pd.Series:
    """Annual values keyed on period-end date, merging tags in priority order."""
    merged = pd.Series(dtype=float)
    for tag in tags:
        df = _facts_frame(facts, tag, unit_pref)
        if df.empty:
            continue
        df = df[df["form"].isin(ANNUAL_FORMS)]
        if instant:
            df = df[df["start"].isna()] if "start" in df else df
        else:
            df = df[df["days"].between(350, 380)]
        if df.empty:
            continue
        merged = merged.combine_first(_latest_per_end(df)) if len(merged) else _latest_per_end(df)
    return merged.sort_index()


def _ytd_pair(facts, tags, fy_end: pd.Timestamp, unit_pref, target_end=None):
    """(current YTD value, prior-year YTD value, current period end) from 10-Qs."""
    for tag in tags:
        df = _facts_frame(facts, tag, unit_pref)
        if df.empty or "start" not in df:
            continue
        df = df[df["form"].isin(QUARTERLY_FORMS) & df["start"].notna()]
        cur = df[(abs((df["start"] - (fy_end + pd.Timedelta(days=1))).dt.days) <= 7)
                 & (df["end"] > fy_end)]
        if target_end is not None:
            cur = cur[cur["end"] == target_end]
        if cur.empty:
            continue
        cur = cur.sort_values(["end", "filed"]).iloc[-1]
        prior_end = cur["end"] - pd.DateOffset(years=1)
        prior = df[(abs((df["end"] - prior_end).dt.days) <= 7)
                   & (abs(df["days"] - cur["days"]) <= 7)]
        if prior.empty:
            continue
        prior = prior.sort_values("filed").iloc[-1]
        return float(cur["val"]), float(prior["val"]), cur["end"]
    return None


def _latest_instant(facts, tags, end: pd.Timestamp, unit_pref):
    for tag in tags:
        df = _facts_frame(facts, tag, unit_pref)
        if df.empty:
            continue
        df = df[(df["end"] == end) & (df["start"].isna() if "start" in df else True)]
        if not df.empty:
            return float(df.sort_values("filed")["val"].iloc[-1])
    return np.nan


def _shares_outstanding(facts) -> tuple[float, str | None]:
    """Latest cover-page share count (sums share classes reported in one filing)."""
    node = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding")
    if not node:
        return np.nan, None
    df = pd.DataFrame(node["units"].get("shares", []))
    if df.empty:
        return np.nan, None
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    latest = df.sort_values(["filed", "end"]).iloc[-1]
    same = df[(df["accn"] == latest["accn"]) & (df["end"] == latest["end"])]
    return float(same["val"].drop_duplicates().sum()), latest["end"].date().isoformat()


# ----------------------------------------------------------------------------
# Derived metrics
# ----------------------------------------------------------------------------
def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived line items. Columns = periods, rows = line items."""
    def row(name):
        return df.loc[name] if name in df.index else pd.Series(np.nan, index=df.columns)

    out = df.copy()
    rev, cogs = row("revenue"), row("cogs")
    out.loc["gross_profit"] = row("gross_profit").fillna(rev - cogs)
    out.loc["cogs"] = cogs.fillna(rev - out.loc["gross_profit"])
    out.loc["ebit"] = row("operating_income").fillna(
        row("pretax_income") + row("interest_expense").fillna(0))
    out.loc["ebitda"] = out.loc["ebit"] + row("da")

    # Debt: DebtCurrent already includes short-term borrowings; otherwise add up.
    current_debt = row("debt_current_total").fillna(
        row("ltd_current").fillna(0) + row("st_borrowings").fillna(0))
    noncurrent = row("ltd_noncurrent")
    # If only LongTermDebt (total, incl. current portion) exists, avoid double counting.
    noncurrent = noncurrent.fillna(row("ltd_total") - row("ltd_current").fillna(0))
    out.loc["total_debt"] = current_debt.fillna(0) + noncurrent.fillna(0)

    # Operating (non-cash, non-debt) working capital.
    cash_like = row("cash").fillna(0) + row("st_investments").fillna(0)
    nwc_full = (row("current_assets") - cash_like) - (row("current_liabilities") - current_debt.fillna(0))
    nwc_simple = (row("receivables").fillna(0) + row("inventory").fillna(0)
                  - row("accounts_payable").fillna(0))
    nwc_simple = nwc_simple.where(row("receivables").notna() | row("accounts_payable").notna())
    out.loc["nwc"] = nwc_full.fillna(nwc_simple)

    out.loc["capex"] = row("capex").abs()
    out.loc["fcf"] = row("cfo") - out.loc["capex"]
    pretax = row("pretax_income")
    out.loc["tax_rate"] = (row("income_tax") / pretax).where(pretax > 0)
    return out


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
@dataclass
class Financials:
    ticker: str
    name: str
    cik: int
    sic: str | None
    sic_description: str | None
    annual: pd.DataFrame                  # rows = line items, cols = fiscal-year labels
    ttm: pd.Series | None                 # latest twelve months (if a newer 10-Q exists)
    ttm_end: str | None
    shares_outstanding: float
    shares_date: str | None
    latest_10k: Filing | None             # latest annual report (10-K, or 20-F for foreign filers)
    warnings: list[str] = field(default_factory=list)
    currency: str = "USD"                 # reporting currency of the statements
    source: str = "SEC EDGAR"
    taxonomy: str = "us-gaap"             # "us-gaap", "ifrs-full" or "yahoo"
    sector: str | None = None             # Yahoo sector/industry (used when there is no SIC code)
    industry: str | None = None

    @property
    def is_financial(self) -> bool:
        try:
            return 6000 <= int(self.sic) <= 6799
        except (TypeError, ValueError):
            return False

    def statement(self, name: str, include_ttm: bool = True) -> pd.DataFrame:
        """Formatted statement in millions of the reporting currency (EPS per share)."""
        layout = STATEMENT_LAYOUT[name]
        data = self.annual.copy()
        if include_ttm and self.ttm is not None:
            data[f"TTM {self.ttm_end}"] = self.ttm
        data = data.reindex(list(layout))
        scale = pd.Series(1e6, index=data.index)
        if "eps_diluted" in scale.index:
            scale["eps_diluted"] = 1.0
        table = data.div(scale, axis=0)
        table.index = [layout[k] for k in table.index]
        return table.dropna(how="all")


def _detect_taxonomy_and_currency(facts: dict) -> tuple[str, str]:
    """US-GAAP if present, else IFRS; currency = the unit most used by revenue/asset facts."""
    all_facts = facts.get("facts", {})
    for tax in ("us-gaap", "ifrs-full"):
        node = all_facts.get(tax)
        if not node:
            continue
        flows, _, _, instants = TAG_SETS[tax]
        counts: dict[str, int] = {}
        for tag in flows["revenue"] + instants["total_assets"] + flows["net_income"]:
            for unit, rows in node.get(tag, {}).get("units", {}).items():
                if "/" not in unit and unit != "shares":
                    counts[unit] = counts.get(unit, 0) + len(rows)
        if counts:
            return tax, max(counts, key=counts.get)
    raise ValueError("No US-GAAP or IFRS financial data in this company's SEC filings.")


def load_financials(client: SECClient, ticker: str, years: int = 5,
                    include_ttm: bool = True) -> Financials:
    info = client.lookup(ticker)
    cik = info["cik"]
    facts = client.company_facts(cik)
    try:
        subs = client.submissions(cik)
        sic, sic_desc = subs.get("sic"), subs.get("sicDescription")
        latest_10k = client.latest_filing(cik)
    except Exception:  # submissions endpoint is optional for the model
        sic = sic_desc = latest_10k = None

    taxonomy, ccy = _detect_taxonomy_and_currency(facts)
    flow_tags, ps_tags, sh_tags, inst_tags = TAG_SETS[taxonomy]
    # The helpers read the "us-gaap" slot; point it at whichever taxonomy the company uses.
    view = {"facts": {"us-gaap": facts["facts"][taxonomy], "dei": facts["facts"].get("dei", {})}}
    money, per_share = (ccy,), (f"{ccy}/shares",)

    warnings: list[str] = []
    if taxonomy == "ifrs-full":
        warnings.append(f"Foreign filer: IFRS statements from the annual report (20-F), in {ccy}.")
    series: dict[str, pd.Series] = {}
    for key, tags in flow_tags.items():
        series[key] = _annual_series(view, tags, False, money)
    for key, tags in ps_tags.items():
        series[key] = _annual_series(view, tags, False, per_share)
    for key, tags in sh_tags.items():
        series[key] = _annual_series(view, tags, False, ("shares",))
    for key, tags in inst_tags.items():
        series[key] = _annual_series(view, tags, True, money)

    # Fiscal year ends = periods where we have revenue (or net income as fallback).
    anchor = series["revenue"].dropna()
    if anchor.empty:
        anchor = series["net_income"].dropna()
        warnings.append("No revenue tag found - revenue-based projections will not work.")
    if anchor.empty:
        raise ValueError(f"No annual financial data found for {ticker} in SEC filings.")
    fy_ends = anchor.index[-years:]

    annual = pd.DataFrame({k: s.reindex(fy_ends) for k, s in series.items()}).T
    annual = _derive(annual)
    annual.columns = [f"FY{d.year}" for d in fy_ends]
    last_fy_end = fy_ends[-1]

    # ---- Trailing twelve months from the latest 10-Q (US filers only) -------
    ttm, ttm_end = None, None
    if include_ttm:
        rev_tags = flow_tags["revenue"] if not series["revenue"].dropna().empty else flow_tags["net_income"]
        anchor_pair = _ytd_pair(view, rev_tags, last_fy_end, money)
        if anchor_pair:
            end = anchor_pair[2]
            values = {}
            for group, unit in [(flow_tags, money), (sh_tags, ("shares",))]:
                for key, tags in group.items():
                    pair = _ytd_pair(view, tags, last_fy_end, unit, target_end=end)
                    fy_val = series[key].get(last_fy_end, np.nan)
                    if key == "diluted_shares":   # a weighted share count is not additive
                        values[key] = pair[0] if pair else fy_val
                    else:
                        values[key] = fy_val + pair[0] - pair[1] if pair else np.nan
            for key, tags in inst_tags.items():
                values[key] = _latest_instant(view, tags, end, money)
            ttm_df = _derive(pd.DataFrame({"ttm": values}))
            ttm = ttm_df["ttm"]
            ttm_end = end.date().isoformat()

    shares, shares_date = _shares_outstanding(view)
    diluted = annual.loc["diluted_shares"].dropna()
    if diluted.size:
        ref = diluted.iloc[-1]
        if not np.isfinite(shares) or abs(shares / ref - 1) > 0.30:
            warnings.append(
                "Cover-page share count missing or inconsistent with diluted shares; "
                "using latest diluted weighted-average shares instead.")
            shares, shares_date = ref, annual.columns[-1]

    fin = Financials(
        ticker=info["ticker"], name=facts.get("entityName", info["name"]), cik=cik,
        sic=sic, sic_description=sic_desc, annual=annual, ttm=ttm, ttm_end=ttm_end,
        shares_outstanding=shares, shares_date=shares_date, latest_10k=latest_10k,
        warnings=warnings, currency=ccy, source="SEC EDGAR", taxonomy=taxonomy,
    )
    if fin.is_financial:
        fin.warnings.append(
            f"SIC {sic} ({sic_desc}) is a financial company. An unlevered FCFF DCF is not "
            "meaningful for banks/insurers - treat results with caution.")
    return fin
