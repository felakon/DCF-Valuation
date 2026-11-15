"""
Growth-stage DCF for AI and high-growth technology companies.

Why a standard 5-year DCF undervalues these companies
------------------------------------------------------
* Five years is too short: growth has to collapse to GDP-like rates by year 5.
* Regression betas of fast-growing stocks are high (often ~2), and a standard DCF
  keeps that high discount rate forever, even after the company has matured.
* CapEx % of revenue says nothing about how much a software / chip company must
  reinvest to grow (R&D, acquisitions, working capital, data centres...).

What this model does differently (Damodaran-style "young growth company" DCF)
-----------------------------------------------------------------------------
1. 10-year forecast in two stages. Growth starts at the analysts' consensus (or recent
   history), loses half of its excess over terminal growth during stage 1, then glides
   to terminal growth by the end of year 10.
2. Margins move from today's level to a target (e.g. a loss-making company scales up
   to a mature-tech margin).
3. Reinvestment is driven by a SALES-TO-CAPITAL ratio: every extra $1 of revenue needs
   $1 / (sales-to-capital) of new capital. Growth is never free.
4. The discount rate starts at today's WACC and converges to a MATURE WACC (beta -> 1)
   in stage 2, because a company growing 5% in year 10 is no longer a high-beta stock.
5. Terminal value uses a terminal ROIC: reinvestment rate = g / ROIC, so the perpetuity
   is consistent with how much the company reinvests.
6. Cross-checks that matter for tech: probability-weighted scenarios, a reverse DCF
   (growth the market is pricing in) and trading multiples.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .classify import Classification
from .financials import Financials
from .market import MarketData
from .model import (DCFResult, _clip, base_period, default_assumptions, run_dcf)

PRESETS = {
    "AI / hyper-growth tech": {"projection_years": 10, "stage1_years": 5,
                               "growth_cap": 0.50, "terminal_roic": 0.25},
    "Growth tech": {"projection_years": 10, "stage1_years": 3,
                    "growth_cap": 0.40, "terminal_roic": 0.18},
}
MATURE_TECH_MARGIN = 0.20     # target margin for companies that are not yet profitable
MARGIN_CEILING = 0.35         # margins above this attract competition: half the excess fades


@dataclass
class GrowthAssumptions:
    projection_years: int = 10
    stage1_years: int = 5
    initial_growth: float = 0.20
    terminal_growth: float = 0.025
    ebit_margin: float = 0.20               # base-period EBIT margin
    target_ebit_margin: float | None = None  # reached in the final year (None = hold)
    tax_rate: float = 0.21
    sales_to_capital: float = 1.5           # $ of new revenue per $1 of reinvestment
    terminal_roic: float = 0.20
    risk_free: float = 0.0425
    beta: float = 1.0
    mature_beta: float = 1.0
    erp: float = 0.045
    pre_tax_cost_of_debt: float = 0.055
    mature_wacc_override: float | None = None
    da_pct: float = 0.03                    # only for the implied EV/EBITDA check
    mid_year: bool = True
    include_lt_investments: bool = False
    sources: dict = field(default_factory=dict, repr=False)

    def growth_path(self) -> np.ndarray:
        n, s = self.projection_years, max(2, min(self.stage1_years, self.projection_years - 1))
        g0, gT = self.initial_growth, self.terminal_growth
        g_mid = gT + 0.5 * (g0 - gT)        # stage 1 removes half of the excess growth
        return np.interp(np.arange(1, n + 1), [1, s, n + 1], [g0, g_mid, gT])

    def margin_path(self) -> np.ndarray:
        target = self.ebit_margin if self.target_ebit_margin is None else self.target_ebit_margin
        return np.linspace(self.ebit_margin, target, self.projection_years + 1)[1:]


def _waccs(fin: Financials, mkt: MarketData, g: GrowthAssumptions, base: pd.Series) -> dict:
    equity = mkt.price * fin.shares_outstanding
    debt = float(np.nan_to_num(base.get("total_debt", 0.0)))
    we = equity / (equity + debt)
    kd_after = g.pre_tax_cost_of_debt * (1 - g.tax_rate)
    ke = g.risk_free + g.beta * g.erp
    ke_mature = g.risk_free + g.mature_beta * g.erp
    w0 = we * ke + (1 - we) * kd_after
    wm = g.mature_wacc_override if g.mature_wacc_override is not None else we * ke_mature + (1 - we) * kd_after
    return {"market_cap": equity, "debt": debt, "cost_of_equity": ke, "mature_cost_of_equity": ke_mature,
            "pre_tax_cost_of_debt": g.pre_tax_cost_of_debt, "after_tax_cost_of_debt": kd_after,
            "weight_equity": we, "weight_debt": 1 - we, "wacc": w0, "terminal_wacc": wm}


# ----------------------------------------------------------------------------
def growth_assumptions(fin: Financials, mkt: MarketData, cls: Classification | None = None,
                       use_ttm: bool = True, consensus_growth: float | None = None,
                       consensus_note: str = "", **overrides) -> GrowthAssumptions:
    """Defaults for the growth model, built on top of the standard assumptions."""
    std = default_assumptions(fin, mkt, use_ttm=use_ttm)
    base, _ = base_period(fin, use_ttm)
    preset = PRESETS.get(cls.profile if cls else "", PRESETS["Growth tech"])
    cap = preset["growth_cap"]
    src = {k: v for k, v in std.sources.items()}

    # Starting growth: analysts first, then recent history.
    rev = fin.annual.loc["revenue"].dropna()
    if consensus_growth is not None and np.isfinite(consensus_growth):
        g0 = _clip(consensus_growth, -0.05, cap, 0.15)
        src["initial_growth"] = f"{consensus_note} (cap {cap:.0%})"
    else:
        # Last year's growth, not the 3-yr CAGR: a boom 2-3 years ago should not be extrapolated.
        latest = rev.iloc[-1] / rev.iloc[-2] - 1 if len(rev) >= 2 else np.nan
        g0 = _clip(latest, -0.05, cap, 0.15)
        src["initial_growth"] = f"last fiscal year revenue growth {latest:.1%} (cap {cap:.0%})"

    # Margin: today's margin matters more than a 3-year average for a fast grower.
    m0 = base.get("ebit", np.nan) / base.get("revenue", np.nan)
    if np.isfinite(m0):
        src["ebit_margin"] = "base-period EBIT margin"
    else:
        m0 = std.ebit_margin
    target = None
    if m0 < 0.10:
        target = MATURE_TECH_MARGIN
        src["target_ebit_margin"] = f"low/negative margin scales to a {MATURE_TECH_MARGIN:.0%} mature-tech margin"
    elif m0 > MARGIN_CEILING:
        target = MARGIN_CEILING + 0.5 * (m0 - MARGIN_CEILING)
        src["target_ebit_margin"] = (f"competition erodes half of the margin above {MARGIN_CEILING:.0%} "
                                     "by the final year")
    else:
        src["target_ebit_margin"] = "held at today's margin"

    # Sales-to-capital = revenue / invested capital (equity + debt - cash).
    ic = (np.nan_to_num(base.get("equity", np.nan), nan=fin.annual.loc["equity"].iloc[-1])
          + np.nan_to_num(base.get("total_debt", 0.0))
          - np.nan_to_num(base.get("cash", 0.0)) - np.nan_to_num(base.get("st_investments", 0.0)))
    stc = base.get("revenue", np.nan) / ic if ic and ic > 0 else np.nan
    src["sales_to_capital"] = ("revenue / invested capital (equity + debt - cash), clipped 0.8..5"
                               if np.isfinite(stc) else "default 3.0 (invested capital not positive)")
    stc = _clip(stc, 0.8, 5.0, 3.0)

    src["terminal_roic"] = f"preset for '{cls.profile if cls else 'Growth tech'}' (moat persists)"
    src["mature_beta"] = "beta converges to 1.0 in stage 2 as the company matures"
    src["stage1_years"] = "preset: years of high growth before the fade"

    ga = GrowthAssumptions(
        projection_years=preset["projection_years"], stage1_years=preset["stage1_years"],
        initial_growth=g0, terminal_growth=std.terminal_growth, ebit_margin=float(m0),
        target_ebit_margin=target, tax_rate=std.tax_rate, sales_to_capital=stc,
        terminal_roic=preset["terminal_roic"], risk_free=std.risk_free, beta=std.beta,
        erp=std.erp, pre_tax_cost_of_debt=std.pre_tax_cost_of_debt, da_pct=std.da_pct,
        sources=src,
    )
    for k, v in overrides.items():
        if hasattr(ga, k) and getattr(ga, k) != v:
            src[k] = "manual override"
    return replace(ga, **overrides)


def run_growth_dcf(fin: Financials, mkt: MarketData, g: GrowthAssumptions,
                   use_ttm: bool = True) -> DCFResult:
    base, base_label = base_period(fin, use_ttm)
    warnings = list(fin.warnings)
    w = _waccs(fin, mkt, g, base)
    n, s, gT = g.projection_years, g.stage1_years, g.terminal_growth
    if w["terminal_wacc"] <= gT:
        raise ValueError(f"Mature WACC ({w['terminal_wacc']:.2%}) must exceed terminal growth ({gT:.2%}).")

    t = np.arange(1, n + 1)
    growth, margin = g.growth_path(), g.margin_path()
    revenue = base["revenue"] * np.cumprod(1 + growth)
    prev_rev = np.concatenate([[base["revenue"]], revenue[:-1]])
    ebit = revenue * margin
    taxes = ebit.clip(min=0) * g.tax_rate
    nopat = ebit - taxes
    reinvestment = (revenue - prev_rev) / g.sales_to_capital
    fcff = nopat - reinvestment

    # Year-by-year discount rate: today's WACC in stage 1, converging to mature WACC.
    rates = np.interp(t, [s, n], [w["wacc"], w["terminal_wacc"]])
    cum = np.cumprod(1 + rates)
    prev_cum = np.concatenate([[1.0], cum[:-1]])
    df = 1 / (prev_cum * (1 + rates) ** 0.5) if g.mid_year else 1 / cum
    pv = fcff * df

    # Terminal value with reinvestment consistent with growth: reinvestment rate = g / ROIC.
    nopat_next = revenue[-1] * (1 + gT) * margin[-1] * (1 - g.tax_rate)
    fcff_next = nopat_next * (1 - gT / g.terminal_roic)
    tv = fcff_next / (w["terminal_wacc"] - gT)
    pv_tv = tv / cum[-1]
    ev = pv.sum() + pv_tv

    cash = float(np.nan_to_num(base.get("cash", 0))) + float(np.nan_to_num(base.get("st_investments", 0)))
    lt_inv = float(np.nan_to_num(base.get("lt_investments", 0))) if g.include_lt_investments else 0.0
    minority = float(np.nan_to_num(base.get("minority_interest", 0)))
    equity = ev - w["debt"] + cash + lt_inv - minority
    per_share = equity / fin.shares_outstanding

    projection = pd.DataFrame({
        "Revenue": revenue, "Growth": growth, "EBIT": ebit, "EBIT margin": margin,
        "Taxes": -taxes, "NOPAT": nopat, "Reinvestment": -reinvestment, "FCFF": fcff,
        "WACC": rates, "Discount factor": df, "PV of FCFF": pv,
    }, index=[f"Y{i}" for i in t]).T

    if fcff[:3].sum() < 0:
        warnings.append("Free cash flow is negative in the early years - value depends on the "
                        "company reaching its target margin.")
    if pv_tv / ev > 0.85:
        warnings.append(f"Terminal value is {pv_tv / ev:.0%} of EV.")
    if g.terminal_roic < w["terminal_wacc"]:
        warnings.append("Terminal ROIC is below the mature WACC: growth destroys value in the terminal period.")

    ebitda_n = ebit[-1] + revenue[-1] * g.da_pct
    return DCFResult(
        company=fin.name, ticker=fin.ticker, base_label=base_label, projection=projection,
        wacc=w, enterprise_value=ev, pv_fcff=pv.sum(), pv_terminal=pv_tv, terminal_value=tv,
        bridge={"Enterprise value": ev, "Less: debt": -w["debt"],
                "Plus: cash & ST investments": cash, "Plus: LT investments": lt_inv,
                "Less: minority interest": -minority, "Equity value": equity},
        equity_value=equity, shares=fin.shares_outstanding, value_per_share=per_share,
        price=mkt.price, upside=per_share / mkt.price - 1,
        implied_exit_multiple=tv / ebitda_n if ebitda_n > 0 else np.nan,
        assumptions=g, warnings=warnings, model="growth", currency=fin.currency,
    )


# ----------------------------------------------------------------------------
# Cross-checks
# ----------------------------------------------------------------------------
def weighted_scenarios(fin, mkt, g: GrowthAssumptions, use_ttm=True,
                       probabilities=(0.25, 0.50, 0.25)) -> pd.DataFrame:
    """Bear / base / bull with probabilities, and the probability-weighted value."""
    final_margin = g.margin_path()[-1]
    cases = {
        "Bear": replace(g, initial_growth=g.initial_growth * 0.6 if g.initial_growth > 0 else g.initial_growth - 0.03,
                        target_ebit_margin=max(final_margin - 0.10, 0.05)),
        "Base": g,
        "Bull": replace(g, initial_growth=g.initial_growth * 1.25 if g.initial_growth > 0 else g.initial_growth + 0.03,
                        target_ebit_margin=final_margin + 0.05),
    }
    rows = {}
    for (name, case), p in zip(cases.items(), probabilities):
        res = run_growth_dcf(fin, mkt, case, use_ttm)
        rows[name] = {"Probability": p, "Y1 growth": case.initial_growth,
                      "Final EBIT margin": case.margin_path()[-1],
                      "Value / share": res.value_per_share, "Upside": res.upside}
    df = pd.DataFrame(rows).T
    weighted = float((df["Probability"] * df["Value / share"]).sum())
    df.loc["Probability-weighted"] = {"Probability": 1.0, "Y1 growth": np.nan,
                                      "Final EBIT margin": np.nan, "Value / share": weighted,
                                      "Upside": weighted / mkt.price - 1}
    return df


def growth_margin_grid(fin, mkt, g: GrowthAssumptions, use_ttm=True,
                       growth_step=0.05, margin_step=0.05, n=2) -> pd.DataFrame:
    """Value/share across starting growth (rows) x final EBIT margin (columns)."""
    final_margin = g.margin_path()[-1]
    gs = [round(g.initial_growth + i * growth_step, 4) for i in range(-n, n + 1)]
    ms = [round(final_margin + i * margin_step, 4) for i in range(-n, n + 1)]
    out = pd.DataFrame(index=gs, columns=ms, dtype=float)
    for gv in gs:
        for mv in ms:
            try:
                out.loc[gv, mv] = run_growth_dcf(
                    fin, mkt, replace(g, initial_growth=gv, target_ebit_margin=mv), use_ttm).value_per_share
            except ValueError:
                out.loc[gv, mv] = np.nan
    out.index.name, out.columns.name = "Starting revenue growth", "Final EBIT margin"
    return out


def implied_growth(fin, mkt, assumptions, use_ttm=True, lo=-0.10, hi=1.50, tol=1e-4) -> float | None:
    """Reverse DCF: the starting revenue growth at which value per share = market price."""
    runner = run_growth_dcf if isinstance(assumptions, GrowthAssumptions) else run_dcf

    def gap(g0):
        try:
            return runner(fin, mkt, replace(assumptions, initial_growth=g0), use_ttm).value_per_share - mkt.price
        except ValueError:
            return np.nan

    f_lo, f_hi = gap(lo), gap(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or np.sign(f_lo) == np.sign(f_hi):
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        f_mid = gap(mid)
        if not np.isfinite(f_mid):
            return None
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def trading_multiples(fin: Financials, mkt: MarketData, results: dict[str, DCFResult],
                      use_ttm=True) -> pd.DataFrame:
    """Market multiples today vs the multiples implied by each DCF."""
    base, _ = base_period(fin, use_ttm)
    cash = np.nan_to_num(base.get("cash", 0)) + np.nan_to_num(base.get("st_investments", 0))
    debt = np.nan_to_num(base.get("total_debt", 0))
    mcap = mkt.price * fin.shares_outstanding
    rev, ebit, ni = base.get("revenue", np.nan), base.get("ebit", np.nan), base.get("net_income", np.nan)
    ebitda = base.get("ebitda", np.nan)

    def row(ev, eq):
        return {"EV / Revenue": ev / rev, "EV / EBIT": ev / ebit,
                "EV / EBITDA": ev / ebitda if np.isfinite(ebitda) and ebitda > 0 else np.nan,
                "P / E": eq / ni if np.isfinite(ni) and ni > 0 else np.nan}

    cols = {"Market today": row(mcap + debt - cash, mcap)}
    for name, r in results.items():
        cols[name] = row(r.enterprise_value, r.equity_value)
    return pd.DataFrame(cols)


def rule_of_40(fin: Financials, cls: Classification) -> float | None:
    """Revenue growth + FCF margin (last fiscal year). >= 40% is the SaaS/tech benchmark."""
    a = fin.annual
    fcf_margin = a.loc["fcf"].iloc[-1] / a.loc["revenue"].iloc[-1]
    if not (np.isfinite(fcf_margin) and np.isfinite(cls.latest_growth)):
        return None
    return float(cls.latest_growth + fcf_margin)
