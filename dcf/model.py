"""
Unlevered free-cash-flow (FCFF) DCF.

    FCFF  = EBIT x (1 - tax) + D&A - CapEx - change in NWC
    WACC  = E/(D+E) x Ke + D/(D+E) x Kd x (1 - tax),   Ke = rf + beta x ERP   (CAPM)
    TV    = FCFF_N x (1 + g) / (WACC - g)                                    (Gordon growth)
    EV    = sum PV(FCFF) + PV(TV)
    Equity = EV - debt + cash & short-term investments - minority interest

Default assumptions are derived from the company's own history (3-year averages,
clipped to sensible ranges) and every one of them can be overridden.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .financials import Financials
from .market import MarketData

DEFAULT_ERP = 0.045          # equity risk premium - update from Damodaran's monthly implied ERP
DEFAULT_TERMINAL_GROWTH = 0.025
MARGINAL_TAX = 0.21          # US federal statutory rate (fallback)
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "CHF ", "SEK": "SEK ",
                    "DKK": "DKK ", "NOK": "NOK ", "CAD": "C$", "AUD": "A$", "HKD": "HK$"}


def money(x: float, currency: str = "USD", decimals: int = 2) -> str:
    sym = CURRENCY_SYMBOLS.get(currency, currency + " ")
    return f"{sym}{x:,.{decimals}f}"


@dataclass
class Assumptions:
    projection_years: int = 5
    initial_growth: float = 0.05          # year-1 revenue growth, fades linearly to terminal_growth
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH
    revenue_growth: list[float] | None = None   # explicit path overrides the fade
    ebit_margin: float = 0.15             # base-year EBIT margin
    target_ebit_margin: float | None = None  # if set, margin moves linearly to this by the final year
    tax_rate: float = MARGINAL_TAX
    da_pct: float = 0.03                  # D&A as % of revenue
    capex_pct: float = 0.04               # CapEx as % of revenue
    nwc_pct: float = 0.05                 # operating working capital as % of revenue
    risk_free: float = 0.0425
    beta: float = 1.0
    erp: float = DEFAULT_ERP
    pre_tax_cost_of_debt: float = 0.055
    wacc_override: float | None = None
    mid_year: bool = True                 # mid-year discounting convention
    include_lt_investments: bool = False  # count long-term marketable securities as cash-like
    sources: dict = field(default_factory=dict, repr=False)   # where each default came from

    def growth_path(self) -> np.ndarray:
        n = self.projection_years
        if self.revenue_growth is not None:
            path = list(self.revenue_growth)[:n]
            return np.array(path + [path[-1]] * (n - len(path)))
        # n+1 evenly spaced points from year-1 growth to terminal growth; the last
        # point (= terminal growth) belongs to the perpetuity, not the forecast.
        return np.linspace(self.initial_growth, self.terminal_growth, n + 1)[:-1]

    def margin_path(self) -> np.ndarray:
        n = self.projection_years
        target = self.ebit_margin if self.target_ebit_margin is None else self.target_ebit_margin
        return np.linspace(self.ebit_margin, target, n + 1)[1:]


@dataclass
class DCFResult:
    company: str
    ticker: str
    base_label: str
    projection: pd.DataFrame
    wacc: dict
    enterprise_value: float
    pv_fcff: float
    pv_terminal: float
    terminal_value: float
    bridge: dict
    equity_value: float
    shares: float
    value_per_share: float
    price: float
    upside: float
    implied_exit_multiple: float
    assumptions: "Assumptions"             # or growth.GrowthAssumptions
    warnings: list[str] = field(default_factory=list)
    model: str = "standard"                # "standard" or "growth"
    currency: str = "USD"                  # reporting currency of values and price

    @property
    def terminal_share_of_ev(self) -> float:
        return self.pv_terminal / self.enterprise_value

    def summary(self) -> pd.Series:
        return pd.Series({
            "Model": ("Growth-stage DCF (10-yr, 2 stages)" if self.model == "growth"
                      else "Standard DCF (5-yr)"),
            "WACC": (f"{self.wacc['wacc']:.2%} -> {self.wacc['terminal_wacc']:.2%} (mature)"
                     if "terminal_wacc" in self.wacc else f"{self.wacc['wacc']:.2%}"),
            "Terminal growth": f"{self.assumptions.terminal_growth:.2%}",
            f"PV of FCFF ({self.currency}m)": f"{self.pv_fcff / 1e6:,.0f}",
            f"PV of terminal value ({self.currency}m)": f"{self.pv_terminal / 1e6:,.0f}",
            f"Enterprise value ({self.currency}m)": f"{self.enterprise_value / 1e6:,.0f}",
            f"Equity value ({self.currency}m)": f"{self.equity_value / 1e6:,.0f}",
            "Shares (m)": f"{self.shares / 1e6:,.1f}",
            "Intrinsic value / share": money(self.value_per_share, self.currency),
            "Current price": money(self.price, self.currency),
            "Upside / (downside)": f"{self.upside:+.1%}",
            "Terminal value % of EV": f"{self.terminal_share_of_ev:.0%}",
            "Implied exit EV/EBITDA": f"{self.implied_exit_multiple:.1f}x",
        }, name=self.ticker)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _clip(x, lo, hi, fallback):
    if x is None or not np.isfinite(x):
        return fallback
    return float(min(max(x, lo), hi))


def base_period(fin: Financials, use_ttm: bool = True) -> tuple[pd.Series, str]:
    """The statement column the projections start from (TTM if available)."""
    if use_ttm and fin.ttm is not None and np.isfinite(fin.ttm.get("revenue", np.nan)):
        return fin.ttm, f"TTM to {fin.ttm_end}"
    return fin.annual.iloc[:, -1], fin.annual.columns[-1]


def _ratio_avg(fin: Financials, num: str, den: str = "revenue", n: int = 3) -> float:
    a = fin.annual
    if num not in a.index or den not in a.index:
        return np.nan
    r = (a.loc[num] / a.loc[den]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(r.iloc[-n:].mean()) if len(r) else np.nan


def default_assumptions(fin: Financials, mkt: MarketData, use_ttm: bool = True,
                        **overrides) -> Assumptions:
    """Derive every assumption from the company's own history, then apply overrides."""
    a = fin.annual
    base, _ = base_period(fin, use_ttm)
    src: dict[str, str] = {}

    rev = a.loc["revenue"].dropna()
    if len(rev) >= 2:
        k = min(3, len(rev) - 1)
        cagr = (rev.iloc[-1] / rev.iloc[-1 - k]) ** (1 / k) - 1
        src["initial_growth"] = f"{k}-yr revenue CAGR {cagr:.1%} (clipped to -5%..25%)"
    else:
        cagr = np.nan
        src["initial_growth"] = "default (insufficient history)"
    growth = _clip(cagr, -0.05, 0.25, 0.05)

    margin = _clip(_ratio_avg(fin, "ebit"), -0.20, 0.60, 0.10)
    src["ebit_margin"] = "3-yr average EBIT margin"

    tax = _clip(a.loc["tax_rate"].dropna().iloc[-3:].mean() if a.loc["tax_rate"].notna().any()
                else np.nan, 0.10, 0.30, MARGINAL_TAX)
    src["tax_rate"] = "3-yr average effective tax rate (clipped to 10%..30%)"

    da = _clip(_ratio_avg(fin, "da"), 0.0, 0.25, 0.03)
    src["da_pct"] = "3-yr average D&A / revenue"
    capex = _clip(_ratio_avg(fin, "capex"), 0.0, 0.30, 0.04)
    src["capex_pct"] = "3-yr average CapEx / revenue"
    nwc_ratio = base.get("nwc", np.nan) / base.get("revenue", np.nan)
    src["nwc_pct"] = "base-period operating working capital / revenue"
    if not np.isfinite(nwc_ratio):
        nwc_ratio = a.loc["nwc"].iloc[-1] / a.loc["revenue"].iloc[-1]
        src["nwc_pct"] = "last fiscal year operating working capital / revenue"
    nwc = _clip(nwc_ratio, -0.15, 0.40, 0.05)

    # Cost of debt = interest expense / average debt, bounded around the risk-free rate.
    rf = mkt.risk_free
    debt = a.loc["total_debt"].replace(0, np.nan).dropna()
    interest = a.loc["interest_expense"].dropna()
    kd = np.nan
    if len(debt) and len(interest):
        avg_debt = debt.iloc[-2:].mean()
        kd = interest.iloc[-1] / avg_debt
        src["pre_tax_cost_of_debt"] = f"interest expense / average debt = {kd:.2%} (bounded rf+0.5%..rf+6%)"
    else:
        src["pre_tax_cost_of_debt"] = "risk-free + 1.5% (no interest expense reported)"
    kd = _clip(kd, rf + 0.005, rf + 0.06, rf + 0.015)

    src["risk_free"] = "10Y US Treasury yield"
    src["beta"] = "5-yr monthly regression vs S&P 500, Blume-adjusted" if mkt.raw_beta else "manual / default"
    src["erp"] = "Default; update from Damodaran's implied ERP"
    src["terminal_growth"] = "~long-run nominal GDP / inflation proxy"

    assumptions = Assumptions(
        initial_growth=growth, ebit_margin=margin, tax_rate=tax, da_pct=da,
        capex_pct=capex, nwc_pct=nwc, risk_free=rf, beta=mkt.beta,
        pre_tax_cost_of_debt=kd, sources=src,
    )
    for k, v in overrides.items():
        if k in src and getattr(assumptions, k) != v:
            src[k] = "manual override"
    return replace(assumptions, **overrides)


def compute_wacc(fin: Financials, mkt: MarketData, a: Assumptions, base: pd.Series) -> dict:
    equity = mkt.price * fin.shares_outstanding
    debt = float(np.nan_to_num(base.get("total_debt", 0.0)))
    ke = a.risk_free + a.beta * a.erp
    kd_after = a.pre_tax_cost_of_debt * (1 - a.tax_rate)
    we = equity / (equity + debt)
    wacc = we * ke + (1 - we) * kd_after
    if a.wacc_override is not None:
        wacc = a.wacc_override
    return {"market_cap": equity, "debt": debt, "cost_of_equity": ke,
            "pre_tax_cost_of_debt": a.pre_tax_cost_of_debt, "after_tax_cost_of_debt": kd_after,
            "weight_equity": we, "weight_debt": 1 - we, "wacc": wacc}


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
def run_dcf(fin: Financials, mkt: MarketData, a: Assumptions, use_ttm: bool = True) -> DCFResult:
    base, base_label = base_period(fin, use_ttm)
    warnings = list(fin.warnings)
    w = compute_wacc(fin, mkt, a, base)
    r, g, n = w["wacc"], a.terminal_growth, a.projection_years
    if r <= g:
        raise ValueError(f"WACC ({r:.2%}) must exceed terminal growth ({g:.2%}).")

    growth, margin = a.growth_path(), a.margin_path()
    revenue = base["revenue"] * np.cumprod(1 + growth)
    prev_rev = np.concatenate([[base["revenue"]], revenue[:-1]])
    ebit = revenue * margin
    taxes = ebit.clip(min=0) * a.tax_rate
    nopat = ebit - taxes
    da = revenue * a.da_pct
    capex = revenue * a.capex_pct
    d_nwc = (revenue - prev_rev) * a.nwc_pct
    fcff = nopat + da - capex - d_nwc

    t = np.arange(1, n + 1) - (0.5 if a.mid_year else 0.0)
    df = 1 / (1 + r) ** t
    pv = fcff * df

    tv = fcff[-1] * (1 + g) / (r - g)
    pv_tv = tv / (1 + r) ** n            # terminal value discounted from the end of year N
    ev = pv.sum() + pv_tv

    cash = float(np.nan_to_num(base.get("cash", 0))) + float(np.nan_to_num(base.get("st_investments", 0)))
    lt_inv = float(np.nan_to_num(base.get("lt_investments", 0))) if a.include_lt_investments else 0.0
    debt = w["debt"]
    minority = float(np.nan_to_num(base.get("minority_interest", 0)))
    equity = ev - debt + cash + lt_inv - minority
    per_share = equity / fin.shares_outstanding

    ebitda_n = ebit[-1] + da[-1]
    years = [f"Y{i}" for i in range(1, n + 1)]
    projection = pd.DataFrame({
        "Revenue": revenue, "Growth": growth, "EBIT": ebit, "EBIT margin": margin,
        "Taxes": -taxes, "NOPAT": nopat, "D&A": da, "CapEx": -capex,
        "Change in NWC": -d_nwc, "FCFF": fcff, "Discount factor": df, "PV of FCFF": pv,
    }, index=years).T

    if a.capex_pct < a.da_pct and g > 0:
        warnings.append("CapEx < D&A in the terminal year: growth without reinvestment "
                        "may overstate the terminal value.")
    if equity <= 0 or fcff[-1] <= 0:
        warnings.append("Negative free cash flow or equity value: a DCF on current margins does "
                        "not work here - set a target_ebit_margin for a normalised business.")
    if pv_tv / ev > 0.85:
        warnings.append(f"Terminal value is {pv_tv / ev:.0%} of EV - result is very "
                        "sensitive to WACC and terminal growth.")

    return DCFResult(
        company=fin.name, ticker=fin.ticker, base_label=base_label, projection=projection,
        wacc=w, enterprise_value=ev, pv_fcff=pv.sum(), pv_terminal=pv_tv, terminal_value=tv,
        bridge={"Enterprise value": ev, "Less: debt": -debt,
                "Plus: cash & ST investments": cash, "Plus: LT investments": lt_inv,
                "Less: minority interest": -minority, "Equity value": equity},
        equity_value=equity, shares=fin.shares_outstanding, value_per_share=per_share,
        price=mkt.price, upside=per_share / mkt.price - 1,
        implied_exit_multiple=tv / ebitda_n if ebitda_n > 0 else np.nan,
        assumptions=a, warnings=warnings, currency=fin.currency,
    )


def sensitivity(fin: Financials, mkt: MarketData, a: Assumptions,
                row: str, row_values, col: str, col_values, use_ttm: bool = True) -> pd.DataFrame:
    """Value per share across a grid of two assumptions (e.g. 'wacc_override' x 'terminal_growth')."""
    out = pd.DataFrame(index=list(row_values), columns=list(col_values), dtype=float)
    for rv in row_values:
        for cv in col_values:
            try:
                res = run_dcf(fin, mkt, replace(a, **{row: rv, col: cv}), use_ttm)
                out.loc[rv, cv] = res.value_per_share
            except ValueError:
                out.loc[rv, cv] = np.nan
    out.index.name, out.columns.name = row, col
    return out


def wacc_growth_grid(fin, mkt, a, result: DCFResult, step_w=0.005, step_g=0.005, n=2, use_ttm=True):
    """Classic WACC x terminal-growth table centred on the base case."""
    w0, g0 = result.wacc["wacc"], a.terminal_growth
    waccs = [round(w0 + i * step_w, 4) for i in range(-n, n + 1)]
    gs = [round(g0 + i * step_g, 4) for i in range(-n, n + 1)]
    return sensitivity(fin, mkt, a, "wacc_override", waccs, "terminal_growth", gs, use_ttm)


def scenarios(fin, mkt, a: Assumptions, use_ttm=True) -> pd.DataFrame:
    """Bear / base / bull on growth and margin."""
    cases = {
        "Bear": replace(a, initial_growth=a.initial_growth - 0.03,
                        target_ebit_margin=a.ebit_margin - 0.03),
        "Base": a,
        "Bull": replace(a, initial_growth=a.initial_growth + 0.03,
                        target_ebit_margin=a.ebit_margin + 0.03),
    }
    rows = {}
    for name, case in cases.items():
        res = run_dcf(fin, mkt, case, use_ttm)
        rows[name] = {"Y1 growth": case.initial_growth,
                      "Final EBIT margin": case.margin_path()[-1],
                      "Value / share": res.value_per_share, "Upside": res.upside}
    return pd.DataFrame(rows).T
