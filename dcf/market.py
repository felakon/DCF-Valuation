"""
Market inputs for the cost of capital: share price, beta and the risk-free rate.

Data comes from Yahoo Finance via ``yfinance``. Beta is *computed*, not scraped:
an OLS regression of 5 years of monthly stock returns on the S&P 500, then
Blume-adjusted (0.67 x raw + 0.33) toward the market average of 1.0.
Every value can be overridden by hand if Yahoo is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_RISK_FREE = 0.0425   # fallback only; the live 10Y Treasury yield is used when available
BENCHMARK = "^GSPC"          # S&P 500
TEN_YEAR = "^TNX"            # CBOE 10-year Treasury yield index

# Market index used for the beta regression, by trading currency.
BENCHMARKS = {"USD": "^GSPC", "EUR": "^STOXX50E", "GBP": "^FTSE", "CHF": "^SSMI",
              "SEK": "^STOXX", "DKK": "^STOXX", "NOK": "^STOXX", "PLN": "^STOXX",
              "JPY": "^N225", "CAD": "^GSPTSE", "HKD": "^HSI", "AUD": "^AXJO"}
# Quotes in minor units (London pence, Israeli agorot, SA cents).
MINOR_UNITS = {"GBp": ("GBP", 100), "GBX": ("GBP", 100), "ILA": ("ILS", 100), "ZAc": ("ZAR", 100)}


@dataclass
class MarketData:
    price: float                     # share price in the REPORTING currency of the statements
    beta: float
    raw_beta: float | None
    risk_free: float
    price_date: str | None = None
    notes: list[str] = field(default_factory=list)
    currency: str = "USD"            # reporting currency (what price is expressed in)
    price_currency: str = "USD"      # currency the stock trades in
    price_original: float | None = None
    fx_rate: float = 1.0             # 1 unit of price_currency = fx_rate units of reporting currency


def normalise_currency(price: float, ccy: str) -> tuple[float, str]:
    """Convert minor-unit quotes (e.g. London pence 'GBp') into the major currency."""
    if ccy in MINOR_UNITS:
        major, factor = MINOR_UNITS[ccy]
        return price / factor, major
    return price, ccy


def fx_rate(from_ccy: str, to_ccy: str, lookup=None) -> float:
    """Units of to_ccy per 1 from_ccy. `lookup(pair)` returns a quote for e.g. 'USDEUR=X'."""
    if from_ccy == to_ccy:
        return 1.0
    if lookup is None:
        import yfinance as yf

        def lookup(pair):
            return float(yf.Ticker(pair).history(period="5d")["Close"].dropna().iloc[-1])
    return float(lookup(f"{from_ccy}{to_ccy}=X"))


def regression_beta(stock: pd.Series, market: pd.Series) -> tuple[float, float, int]:
    """Raw OLS beta, R-squared and number of observations from two return series."""
    df = pd.concat([stock, market], axis=1).dropna()
    if len(df) < 24:
        raise ValueError("Need at least 24 monthly observations for a beta.")
    x, y = df.iloc[:, 1].values, df.iloc[:, 0].values
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    return float(beta), float(r2), len(df)


def blume_adjust(raw_beta: float) -> float:
    return 0.67 * raw_beta + 0.33


def fetch_consensus_growth(ticker: str) -> tuple[float | None, str]:
    """Analysts' average revenue growth for this and next fiscal year (Yahoo Finance)."""
    try:
        import yfinance as yf
        est = yf.Ticker(ticker).revenue_estimate
        growth = est.loc[[i for i in ("0y", "+1y") if i in est.index], "growth"].dropna().astype(float)
        if growth.empty:
            return None, "no analyst estimates"
        return float(growth.mean()), f"analyst consensus revenue growth ({', '.join(f'{g:.1%}' for g in growth)})"
    except Exception as exc:  # noqa: BLE001
        return None, f"analyst estimates unavailable ({type(exc).__name__})"


def fetch_market_data(ticker: str, price: float | None = None, beta: float | None = None,
                      risk_free: float | None = None, years: int = 5,
                      reporting_currency: str = "USD", price_currency: str | None = None) -> MarketData:
    """Fetch price, beta and risk-free rate; any argument you pass overrides the live value.

    The price is converted into the statements' reporting currency (e.g. ASML's US listing
    trades in USD but reports in EUR), so value per share and price are comparable.
    A price you pass yourself is taken to be in `price_currency` (default: reporting currency).
    """
    notes: list[str] = []
    raw_beta = None
    price_date = None
    manual_price = price is not None
    price_ccy = price_currency or (reporting_currency if manual_price else None)
    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover
        yf = None
        notes.append("yfinance not installed - using manual/default market inputs.")

    if yf is not None:
        tk = yf.Ticker(ticker)
        # ---- price -----------------------------------------------------------
        if price is None:
            try:
                hist = tk.history(period="5d", auto_adjust=False)
                price = float(hist["Close"].dropna().iloc[-1])
                price_date = hist.index[-1].date().isoformat()
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Could not fetch price ({exc}).")
        if price_ccy is None:
            try:
                price_ccy = tk.fast_info["currency"]
            except Exception:  # noqa: BLE001
                price_ccy = reporting_currency
                notes.append(f"Trading currency unknown - assumed {reporting_currency}.")
        # ---- beta ------------------------------------------------------------
        if beta is None:
            bench = BENCHMARKS.get(normalise_currency(1.0, price_ccy or "USD")[1], BENCHMARK)
            try:
                px = yf.download([ticker, bench], period=f"{years}y", interval="1mo",
                                 auto_adjust=True, progress=False)["Close"]
                rets = px.pct_change(fill_method=None).dropna()
                raw_beta, r2, n = regression_beta(rets[ticker], rets[bench])
                beta = blume_adjust(raw_beta)
                notes.append(f"Beta: {n} monthly returns vs {bench}, raw {raw_beta:.2f} "
                             f"(R² {r2:.2f}), Blume-adjusted {beta:.2f}.")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Could not compute beta ({exc}); using 1.0.")
        # ---- risk-free rate --------------------------------------------------
        if risk_free is None:
            try:
                tnx = float(yf.Ticker(TEN_YEAR).history(period="5d")["Close"].dropna().iloc[-1])
                risk_free = tnx / 100 if tnx < 20 else tnx / 1000
                notes.append(f"Risk-free rate: 10Y US Treasury yield {risk_free:.2%}.")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Could not fetch 10Y yield ({exc}).")

    if price is None:
        raise ValueError("No share price available - pass price=... manually.")
    price_ccy = price_ccy or reporting_currency

    # ---- currency alignment --------------------------------------------------
    original = price
    price, major = normalise_currency(price, price_ccy)
    if major != price_ccy:
        notes.append(f"Price quoted in {price_ccy}; converted to {major}.")
    rate = 1.0
    if major != reporting_currency:
        rate = fx_rate(major, reporting_currency)
        price = price * rate
        notes.append(f"Price {original:,.2f} {major} converted at {rate:.4f} = {price:,.2f} "
                     f"{reporting_currency} (statements are in {reporting_currency}). If this is an ADR "
                     "where one ADR is not one ordinary share (e.g. BP, Shell), use the home-exchange "
                     "ticker instead (e.g. SHEL.L).")
    if beta is None:
        beta = 1.0
    if risk_free is None:
        risk_free = DEFAULT_RISK_FREE
        notes.append(f"Risk-free rate defaulted to {risk_free:.2%}.")
    if reporting_currency != "USD" and "Risk-free rate:" in " ".join(notes):
        notes.append(f"Cash flows are in {reporting_currency} but the risk-free rate is the US 10Y yield. "
                     "Use the government bond yield of the same currency (e.g. German 10Y Bund for EUR) "
                     "via RISK_FREE for a consistent discount rate.")
    return MarketData(price=price, beta=beta, raw_beta=raw_beta, risk_free=risk_free,
                      price_date=price_date, notes=notes, currency=reporting_currency,
                      price_currency=major, price_original=original, fx_rate=rate)
