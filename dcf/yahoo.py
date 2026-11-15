"""
Fallback data source for companies that do not file with the SEC.

European companies listed only in Europe (e.g. LVMH ``MC.PA``, Siemens ``SIE.DE``,
Adyen ``ADYEN.AS``) never appear on EDGAR. For those, the standardized annual
statements published by Yahoo Finance are used instead (usually 4 fiscal years),
mapped onto exactly the same line items as the SEC parser so every model works unchanged.

``load_company`` is the single entry point: SEC first (US GAAP or IFRS), Yahoo otherwise.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .financials import (FLOW_TAGS, INSTANT_TAGS, PER_SHARE_TAGS, SHARE_FLOW_TAGS,
                         Financials, _derive, load_financials)
from .sec import SECClient, SECError

# Yahoo row labels -> standardized line items (first match wins).
YAHOO_MAP: dict[str, list[str]] = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cogs": ["Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "rnd": ["Research And Development"],
    "sga": ["Selling General And Administration"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "pretax_income": ["Pretax Income"],
    "income_tax": ["Tax Provision"],
    "net_income": ["Net Income Common Stockholders", "Net Income"],
    "da": ["Reconciled Depreciation", "Depreciation And Amortization",
           "Depreciation Amortization Depletion"],
    "sbc": ["Stock Based Compensation"],
    "cfo": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure"],
    "dividends": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "buybacks": ["Repurchase Of Capital Stock", "Common Stock Payments"],
    "eps_diluted": ["Diluted EPS"],
    "diluted_shares": ["Diluted Average Shares"],
    "cash": ["Cash And Cash Equivalents"],
    "st_investments": ["Other Short Term Investments"],
    "receivables": ["Accounts Receivable", "Receivables"],
    "inventory": ["Inventory"],
    "current_assets": ["Current Assets"],
    "lt_investments": ["Investments And Advances"],
    "ppe_net": ["Net PPE"],
    "total_assets": ["Total Assets"],
    "accounts_payable": ["Accounts Payable", "Payables"],
    "current_liabilities": ["Current Liabilities"],
    "debt_current_total": ["Current Debt", "Current Debt And Capital Lease Obligation"],
    "ltd_noncurrent": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "equity": ["Stockholders Equity", "Common Stock Equity"],
    "minority_interest": ["Minority Interest"],
}

# Exchange suffixes that never exist on EDGAR -> go straight to Yahoo.
EXCHANGE_SUFFIX = re.compile(
    r"\.(AS|PA|DE|F|MI|MC|L|IL|SW|ST|CO|HE|OL|BR|LS|VI|IR|WA|AT|PR|BD|TO|V|HK|T|AX|SI|KS|NS|BO)$", re.I)

ALL_ITEMS = (list(FLOW_TAGS) + list(PER_SHARE_TAGS) + list(SHARE_FLOW_TAGS) + list(INSTANT_TAGS))


def has_exchange_suffix(ticker: str) -> bool:
    return bool(EXCHANGE_SUFFIX.search(ticker.strip()))


def financials_from_frames(ticker: str, income: pd.DataFrame, balance: pd.DataFrame,
                           cashflow: pd.DataFrame, info: dict, years: int = 5) -> Financials:
    """Build a Financials object from Yahoo statement tables (rows = labels, cols = dates)."""
    frames = [f for f in (income, balance, cashflow) if f is not None and not f.empty]
    if not frames:
        raise ValueError(f"Yahoo Finance has no financial statements for {ticker}.")
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.columns = pd.to_datetime(combined.columns)
    combined = combined.reindex(sorted(combined.columns), axis=1)

    rows = {}
    for key in ALL_ITEMS:
        label = next((lbl for lbl in YAHOO_MAP.get(key, []) if lbl in combined.index), None)
        rows[key] = combined.loc[label].astype(float) if label else pd.Series(np.nan, index=combined.columns)
    annual = pd.DataFrame(rows).T
    annual = annual.loc[:, annual.loc["revenue"].notna()]          # drop empty years
    if annual.empty:
        raise ValueError(f"Yahoo Finance has no revenue history for {ticker}.")
    annual = annual.iloc[:, -years:]
    ends = annual.columns
    annual = _derive(annual)
    annual.columns = [f"FY{d.year}" for d in ends]

    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    shares_date = "Yahoo Finance (latest)"
    if not shares:
        ordinary = combined.loc["Ordinary Shares Number"].dropna() if "Ordinary Shares Number" in combined.index else []
        shares = float(ordinary.iloc[-1]) if len(ordinary) else annual.loc["diluted_shares"].dropna().iloc[-1]
        shares_date = annual.columns[-1]

    ccy = info.get("financialCurrency") or info.get("currency") or "USD"
    warnings = [f"Not an SEC filer: statements from Yahoo Finance (standardized annual report data, {ccy})."]
    return Financials(
        ticker=ticker.upper(), name=info.get("longName") or info.get("shortName") or ticker.upper(),
        cik=0, sic=None, sic_description=info.get("industry"), annual=annual, ttm=None, ttm_end=None,
        shares_outstanding=float(shares), shares_date=shares_date, latest_10k=None,
        warnings=warnings, currency=ccy, source="Yahoo Finance", taxonomy="yahoo",
        sector=info.get("sector"), industry=info.get("industry"),
    )


def load_financials_yahoo(ticker: str, years: int = 5) -> Financials:
    import yfinance as yf
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:  # noqa: BLE001
        info = {}
    return financials_from_frames(ticker, t.income_stmt, t.balance_sheet, t.cashflow, info, years)


def load_company(ticker: str, user_agent: str | None = None, years: int = 5,
                 include_ttm: bool = True, source: str = "auto") -> tuple[Financials, SECClient | None]:
    """SEC EDGAR first (10-K or 20-F), Yahoo Finance as fallback. Returns (financials, sec_client)."""
    ticker = ticker.strip()
    note = None
    if source in ("auto", "sec") and not has_exchange_suffix(ticker):
        try:
            client = SECClient(user_agent)
            return load_financials(client, ticker, years, include_ttm), client
        except (SECError, ValueError) as exc:
            if source == "sec":
                raise
            note = f"SEC data not available ({exc})"
    fin = load_financials_yahoo(ticker, years)
    if note:
        fin.warnings.insert(0, note)
    return fin, None
