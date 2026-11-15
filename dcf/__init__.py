"""SEC-powered DCF valuation: ticker in, intrinsic value per share out."""
from .financials import Financials, load_financials
from .market import MarketData, fetch_market_data
from .model import (Assumptions, DCFResult, default_assumptions, run_dcf, scenarios,
                    sensitivity, wacc_growth_grid)
from .sec import SECClient
from .classify import Classification, classify, fetch_10k_text
from .yahoo import load_company, load_financials_yahoo
from .growth import (GrowthAssumptions, growth_assumptions, growth_margin_grid, implied_growth,
                     rule_of_40, run_growth_dcf, trading_multiples, weighted_scenarios)

__version__ = "2.1.0"


def value_company(ticker: str, user_agent: str, use_ttm: bool = True,
                  market_overrides: dict | None = None, **assumption_overrides):
    """One-call pipeline: SEC filings -> market data -> assumptions -> DCF."""
    fin, _ = load_company(ticker, user_agent, include_ttm=use_ttm)
    mkt = fetch_market_data(fin.ticker, reporting_currency=fin.currency, **(market_overrides or {}))
    a = default_assumptions(fin, mkt, use_ttm=use_ttm, **assumption_overrides)
    return fin, mkt, run_dcf(fin, mkt, a, use_ttm=use_ttm)


def auto_value(ticker: str, user_agent: str, use_ttm: bool = True, mode: str = "auto",
               scan_10k: bool = True, **overrides):
    """Classify the company, then run the growth-stage DCF (AI/tech) or the standard DCF."""
    from .market import fetch_consensus_growth
    fin, client = load_company(ticker, user_agent, include_ttm=use_ttm)
    text = fetch_10k_text(client, fin) if scan_10k else None
    cls = classify(fin, text)
    mkt = fetch_market_data(fin.ticker, reporting_currency=fin.currency)
    model = cls.model if mode == "auto" else mode
    if model == "growth":
        cg, note = fetch_consensus_growth(fin.ticker)
        a = growth_assumptions(fin, mkt, cls, use_ttm, cg, note, **overrides)
        res = run_growth_dcf(fin, mkt, a, use_ttm)
    else:
        a = default_assumptions(fin, mkt, use_ttm=use_ttm, **overrides)
        res = run_dcf(fin, mkt, a, use_ttm)
    return fin, cls, mkt, res


__all__ = [
    "SECClient", "Financials", "load_financials", "MarketData", "fetch_market_data",
    "Assumptions", "DCFResult", "default_assumptions", "run_dcf", "sensitivity",
    "wacc_growth_grid", "scenarios", "value_company", "Classification", "classify",
    "fetch_10k_text", "GrowthAssumptions", "growth_assumptions", "run_growth_dcf",
    "weighted_scenarios", "growth_margin_grid", "implied_growth", "trading_multiples", "rule_of_40",
    "auto_value", "load_company", "load_financials_yahoo",
]
