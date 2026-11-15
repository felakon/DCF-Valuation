"""Charts, formatted tables and Excel export for a DCF run."""
from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .financials import Financials
from .model import CURRENCY_SYMBOLS, DCFResult

# Palette: blue = historical/positive, orange = projected/negative, warm-gray neutral.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b5b3ad"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("dcf_div", [ORANGE, "#f0efec", BLUE])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False,
})


def _millions(x, _pos=None):
    return f"{x / 1e6:,.0f}"


def _money(x: float, ccy: str = "USD", decimals: int = 2) -> str:
    """Currency amount safe for matplotlib text ('$' would otherwise start mathtext)."""
    sym = CURRENCY_SYMBOLS.get(ccy, ccy + " ").replace("$", "\\$")
    return f"{sym}{x:,.{decimals}f}"


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def plot_history_and_projection(fin: Financials, res: DCFResult):
    """Revenue and free cash flow: reported history vs. model projection (two panels)."""
    hist_rev = fin.annual.loc["revenue"]
    hist_fcf = fin.annual.loc["fcf"]
    proj = res.projection
    proj_labels = list(proj.columns)   # Y1..Yn from the base period (TTM or last FY)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, hist, fut, title in [
        (axes[0], hist_rev, proj.loc["Revenue"], "Revenue"),
        (axes[1], hist_fcf, proj.loc["FCFF"], "Free cash flow (hist: CFO - CapEx, proj: FCFF)"),
    ]:
        labels = list(hist.index) + proj_labels
        x = np.arange(len(labels))
        ax.bar(x[: len(hist)], hist.values, color=BLUE, width=0.6, label="Reported (10-K)")
        ax.bar(x[len(hist):], fut.values, color=ORANGE, width=0.6, label="Projected")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_millions))
        ax.set_ylabel(f"{fin.currency} millions")
        ax.set_title(title, loc="left")
        ax.axhline(0, color=INK_2, linewidth=0.8)
    axes[0].legend(loc="upper left")
    fig.suptitle(f"{res.company} ({res.ticker})", x=0.01, ha="left", color=INK, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_bridge(res: DCFResult):
    """Waterfall from enterprise value to value per share."""
    items = [(k, v) for k, v in res.bridge.items() if k == "Equity value" or k == "Enterprise value" or abs(v) > 0]
    fig, ax = plt.subplots(figsize=(9, 4))
    running = 0.0
    for i, (label, val) in enumerate(items):
        total = label in ("Enterprise value", "Equity value")
        bottom = 0 if total else (running if val >= 0 else running + val)
        height = val if total else abs(val)
        color = BLUE if total else (GRAY if val >= 0 else ORANGE)
        ax.bar(i, height, bottom=bottom, color=color, width=0.6)
        ax.text(i, bottom + height, f"{val / 1e6:,.0f}", ha="center", va="bottom", fontsize=9, color=INK_2)
        running = val if total else running + val
    ax.set_xticks(range(len(items)), [k for k, _ in items], rotation=20, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_millions))
    ax.set_ylabel(f"{res.currency} millions")
    ax.set_title(f"EV to equity value: {_money(res.value_per_share, res.currency)} per share vs "
                 f"{_money(res.price, res.currency)} price ({res.upside:+.1%})", loc="left")
    fig.tight_layout()
    return fig


def plot_sensitivity(grid: pd.DataFrame, price: float, row_label="WACC", col_label="Terminal growth",
                     currency: str = "USD"):
    """Heatmap of value/share; blue above today's price, orange below, gray at the price."""
    vals = grid.values.astype(float)
    spread = np.nanmax(np.abs(vals - price)) or 1.0
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.imshow(vals, cmap=DIVERGING, vmin=price - spread, vmax=price + spread, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(grid.shape[1]), [f"{c:.1%}" for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [f"{r:.1%}" for r in grid.index])
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, _money(vals[i, j], currency, 0), ha="center", va="center", fontsize=9, color=INK)
    ci, cj = vals.shape[0] // 2, vals.shape[1] // 2      # base case sits in the centre
    ax.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.5))
    ax.set_title(f"Value per share sensitivity (current price {_money(price, currency)}, base case boxed)",
                 loc="left")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def assumptions_table(res: DCFResult) -> pd.DataFrame:
    if res.model == "growth":
        return growth_assumptions_table(res)
    a = res.assumptions
    pct = lambda x: f"{x:.2%}"  # noqa: E731
    rows = [
        ("Year-1 revenue growth", pct(a.initial_growth), a.sources.get("initial_growth", "")),
        ("Terminal growth", pct(a.terminal_growth), a.sources.get("terminal_growth", "")),
        ("EBIT margin (base)", pct(a.ebit_margin), a.sources.get("ebit_margin", "")),
        ("EBIT margin (final year)", pct(a.margin_path()[-1]),
         "target override" if a.target_ebit_margin is not None else "held flat"),
        ("Tax rate", pct(a.tax_rate), a.sources.get("tax_rate", "")),
        ("D&A % revenue", pct(a.da_pct), a.sources.get("da_pct", "")),
        ("CapEx % revenue", pct(a.capex_pct), a.sources.get("capex_pct", "")),
        ("NWC % revenue", pct(a.nwc_pct), a.sources.get("nwc_pct", "")),
        ("Risk-free rate", pct(a.risk_free), a.sources.get("risk_free", "")),
        ("Beta", f"{a.beta:.2f}", a.sources.get("beta", "")),
        ("Equity risk premium", pct(a.erp), a.sources.get("erp", "")),
        ("Cost of equity", pct(res.wacc["cost_of_equity"]), "CAPM: rf + beta x ERP"),
        ("Pre-tax cost of debt", pct(a.pre_tax_cost_of_debt), a.sources.get("pre_tax_cost_of_debt", "")),
        ("Equity / debt weights", f"{res.wacc['weight_equity']:.0%} / {res.wacc['weight_debt']:.0%}",
         "market cap vs book debt"),
        ("WACC", pct(res.wacc["wacc"]),
         "manual override" if a.wacc_override is not None else "computed"),
        ("Discounting", "mid-year" if a.mid_year else "end-of-year", ""),
    ]
    return pd.DataFrame(rows, columns=["Assumption", "Value", "Source"]).set_index("Assumption")


def growth_assumptions_table(res: DCFResult) -> pd.DataFrame:
    a, w = res.assumptions, res.wacc
    pct = lambda x: f"{x:.2%}"  # noqa: E731
    src = a.sources.get
    rows = [
        ("Forecast length", f"{a.projection_years} years ({a.stage1_years} high-growth)", src("stage1_years", "")),
        ("Year-1 revenue growth", pct(a.initial_growth), src("initial_growth", "")),
        (f"Growth at end of year {a.stage1_years}", pct(a.growth_path()[a.stage1_years - 1]),
         "half of the excess over terminal growth is gone"),
        ("Terminal growth", pct(a.terminal_growth), src("terminal_growth", "")),
        ("EBIT margin (base)", pct(a.ebit_margin), src("ebit_margin", "")),
        ("EBIT margin (final year)", pct(a.margin_path()[-1]), src("target_ebit_margin", "")),
        ("Tax rate", pct(a.tax_rate), src("tax_rate", "")),
        ("Sales-to-capital", f"{a.sales_to_capital:.2f}x", src("sales_to_capital", "")),
        ("Terminal ROIC", pct(a.terminal_roic), src("terminal_roic", "")),
        ("Terminal reinvestment rate", pct(a.terminal_growth / a.terminal_roic), "g / ROIC"),
        ("Risk-free rate", pct(a.risk_free), src("risk_free", "")),
        ("Beta today -> mature", f"{a.beta:.2f} -> {a.mature_beta:.2f}", src("mature_beta", "")),
        ("Equity risk premium", pct(a.erp), src("erp", "")),
        ("Cost of equity today -> mature", f"{pct(w['cost_of_equity'])} -> {pct(w['mature_cost_of_equity'])}",
         "CAPM: rf + beta x ERP"),
        ("Pre-tax cost of debt", pct(a.pre_tax_cost_of_debt), src("pre_tax_cost_of_debt", "")),
        ("WACC today -> mature", f"{pct(w['wacc'])} -> {pct(w['terminal_wacc'])}",
         "manual override (mature)" if a.mature_wacc_override is not None else "converges in stage 2"),
        ("Discounting", "mid-year" if a.mid_year else "end-of-year", ""),
    ]
    return pd.DataFrame(rows, columns=["Assumption", "Value", "Source"]).set_index("Assumption")


def projection_table(res: DCFResult) -> pd.DataFrame:
    """Projection with $ rows in millions and ratio rows as percentages (strings for display)."""
    p = res.projection.copy().astype(object)
    for idx in p.index:
        if idx in ("Growth", "EBIT margin", "WACC"):
            p.loc[idx] = [f"{v:.1%}" for v in res.projection.loc[idx]]
        elif idx == "Discount factor":
            p.loc[idx] = [f"{v:.3f}" for v in res.projection.loc[idx]]
        else:
            p.loc[idx] = [f"{v / 1e6:,.0f}" for v in res.projection.loc[idx]]
    return p


def ratios_table(fin: Financials) -> pd.DataFrame:
    a = fin.annual
    def safe(num, den):
        return (a.loc[num] / a.loc[den]).replace([np.inf, -np.inf], np.nan)
    df = pd.DataFrame({
        "Revenue growth": a.loc["revenue"].pct_change(fill_method=None),
        "Gross margin": safe("gross_profit", "revenue"),
        "EBIT margin": safe("ebit", "revenue"),
        "Net margin": safe("net_income", "revenue"),
        "Effective tax rate": a.loc["tax_rate"],
        "CapEx % revenue": safe("capex", "revenue"),
        "FCF margin": safe("fcf", "revenue"),
        "ROE": safe("net_income", "equity"),
        "Debt / EBITDA": safe("total_debt", "ebitda"),
    }).T
    return df


def style_statement(df: pd.DataFrame):
    """Notebook styling: $m with thousands separators, EPS with 2 decimals."""
    sty = df.style.format("{:,.0f}", na_rep="–")
    eps = [i for i in df.index if "EPS" in i]
    if eps:
        sty = sty.format("{:,.2f}", na_rep="–", subset=pd.IndexSlice[eps, :])
    return sty


def style_ratios(df: pd.DataFrame):
    sty = df.style.format("{:.1%}", na_rep="–")
    if "Debt / EBITDA" in df.index:
        sty = sty.format("{:.2f}x", na_rep="–", subset=pd.IndexSlice[["Debt / EBITDA"], :])
    return sty


# ----------------------------------------------------------------------------
# Excel
# ----------------------------------------------------------------------------
def export_excel(fin: Financials, res: DCFResult, grid: pd.DataFrame | None = None,
                 path: str | None = None, extra_sheets: dict | None = None) -> str:
    path = path or f"{res.ticker}_DCF.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        res.summary().to_frame("Value").to_excel(xl, sheet_name="Summary")
        assumptions_table(res).to_excel(xl, sheet_name="Assumptions")
        (res.projection / pd.Series(
            {i: (1 if i in ("Growth", "EBIT margin", "WACC", "Discount factor") else 1e6)
             for i in res.projection.index}).values[:, None]
         ).to_excel(xl, sheet_name=f"Projection ({res.currency}m)")
        for name in ("Income Statement", "Balance Sheet", "Cash Flow"):
            fin.statement(name).to_excel(xl, sheet_name=f"{name} ({fin.currency}m)"[:31])
        ratios_table(fin).to_excel(xl, sheet_name="Ratios")
        if grid is not None:
            g = grid.copy()
            rname = "WACC" if g.index.name in (None, "wacc_override") else g.index.name
            cname = "g" if g.columns.name in (None, "terminal_growth") else g.columns.name
            g.index = [f"{rname} {w:.2%}" for w in g.index]
            g.columns = [f"{cname} {c:.2%}" for c in g.columns]
            g.to_excel(xl, sheet_name="Sensitivity")
        for name, df in (extra_sheets or {}).items():
            df.to_excel(xl, sheet_name=name[:31])
        for ws in xl.book.worksheets:
            ws.column_dimensions["A"].width = 34
            for col in "BCDEFGHIJKL":
                ws.column_dimensions[col].width = 15
    return path
