# AI & Tech-Aware DCF Valuation from SEC Filings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felakon/DCF-Valuation/blob/main/DCF_Valuation_AI_Tech.ipynb)
[![tests](https://github.com/felakon/DCF-Valuation/actions/workflows/tests.yml/badge.svg)](https://github.com/felakon/DCF-Valuation/actions/workflows/tests.yml)

**Type a ticker, get an intrinsic value per share, from a model that fits the business.**

The notebook pulls a company's latest annual report straight from the SEC (10-K for US companies,
20-F for foreign ones), rebuilds the three financial statements, works out **what kind of company it is**,
and then values it with the right model: a growth-stage DCF for AI and high-growth tech, and a standard
DCF for everything else. Both are run side by side, with a reverse DCF showing what the market is pricing in.

No paid data, no API keys. Runs in Google Colab in one click.

---

## Example: ASML Holding (run on 14 Nov 2026)

| | |
|---|---|
| Classification | Tech (R&D-intensive semiconductor equipment) → growth-stage DCF |
| Data | IFRS statements from ASML's 20-F, in EUR |
| Base-case value per share | **€741** |
| Market price | €1,338.80 |
| Probability-weighted value (bear 25% / base 50% / bull 25%) | €731 |
| Bull case (higher growth and margin) | €1,038 |
| Starting growth the market implies (reverse DCF) | *[fill in from the "Model comparison" table]* |

**Reading the result.** The model values ASML well below its share price. Even the most optimistic cell of the
sensitivity grid (42.6% starting growth, 44.6% final margin) reaches only €1,213. The reverse DCF shows why:
to justify €1,339, the market must expect high growth to last longer than the model's 5-year high-growth stage,
which is plausible for a company with a near-monopoly in EUV lithography. A DCF doesn't say the stock is
"wrong"; it shows what you have to believe to pay today's price.

<!-- Add your screenshots: create a docs/ folder, save the charts from the notebook, then uncomment:
![Sensitivity](docs/asml_sensitivity.png)
![EV bridge](docs/asml_bridge.png)
-->

---

## How it works

**1. Data.** Ticker → SEC company ID → every XBRL fact the company has filed. Handles US GAAP (10-K) and IFRS (20-F).
Companies not registered with the SEC (e.g. `MC.PA`, `SIE.DE`) fall back to Yahoo Finance statements.

**2. Statements.** Standardized income statement, balance sheet and cash flow for the last 5 years, plus a
trailing-twelve-month column from the latest 10-Q.

**3. Classification.** Decides which model fits:

| Company type | How it is detected | Model |
|---|---|---|
| AI / hyper-growth tech | tech industry code or R&D ≥ 8% of revenue, plus an AI-heavy annual report or 30%+ growth | Growth-stage DCF, 10 years, 5 high-growth years |
| Growth tech | tech, 10–30% revenue growth | Growth-stage DCF, 10 years, 3 high-growth years |
| Mature tech / traditional | everything else | Standard 5-year DCF |

The "AI score" counts AI-related terms (artificial intelligence, machine learning, LLMs…) per 10,000 words of the latest annual report.

**4. Market data.** Live share price (converted into the reporting currency), beta from a 5-year monthly regression
against a local index (S&P 500, Euro Stoxx 50…), 10-year government bond yield, analyst consensus growth.

**5. Valuation, sensitivity, scenarios, Excel export.**

## Why AI and tech companies get a different model

A standard 5-year DCF systematically undervalues fast-growing tech companies for structural reasons:
growth has to collapse to GDP-like rates within 5 years, a high regression beta keeps the discount rate high
forever, and CapEx says little about what a software or chip company must reinvest to grow.

| | Standard DCF | Growth-stage DCF (AI / tech) |
|---|---|---|
| Forecast | 5 years | 10 years, two stages |
| Starting growth | 3-yr revenue CAGR (capped 25%) | analyst consensus, else last year's growth (capped 40–50%) |
| Margin | 3-yr average, held flat | today's margin; above 35% half the excess fades (competition) |
| Reinvestment | D&A, CapEx, working capital as % of revenue | sales-to-capital ratio: growth is never free |
| Discount rate | one WACC forever | today's WACC, falling to a mature WACC (beta → 1) |
| Terminal value | FCFF × (1+g) / (WACC − g) | NOPAT × (1 − g/ROIC) / (mature WACC − g) |
| Extra checks | WACC × growth grid | growth × margin grid, probability-weighted scenarios |

Both models also get a **reverse DCF** (the starting growth that makes value = price) and a **trading multiples**
check (EV/Revenue, EV/EBITDA, P/E: market vs implied).

## The data problems this solves

- **Restatements and comparatives**: every annual report repeats prior years; values are keyed on period-end date and the latest filing wins.
- **Inconsistent tags**: companies label the same item differently and change labels over time; each line item has an ordered list of candidate tags, merged year by year.
- **Quarters inside annual reports**: only ~12-month periods count as annual.
- **US GAAP vs IFRS, USD vs EUR**: foreign filers are read in their own accounting standard and currency; the share price is converted to match.
- **Share counts**: cover-page shares (summed across share classes), cross-checked against diluted shares.

These cases are covered by 29 unit tests on synthetic SEC data, including hand-calculated DCFs the engine must match exactly.

## Quick start

1. Click **Open in Colab** above.
2. In **Inputs**, enter a ticker and your name and email (the SEC requires these to send data).
   - US companies: `NVDA`, `KO`, `MSFT`
   - Foreign companies listed in the US: `ASML`, `SAP`, `NVO`
   - Companies listed only in Europe: `ASML.AS`, `MC.PA`, `SIE.DE`
   - For EUR companies, set `RISK_FREE` to the German 10-year Bund yield.
3. **Runtime → Run all.** The Excel file downloads at the end.

## Project structure

```
DCF_Valuation_AI_Tech.ipynb   the notebook (self-contained, runs in Colab)
dcf/
  sec.py          SEC EDGAR client
  financials.py   XBRL → standardized statements (US GAAP and IFRS), TTM
  yahoo.py        Yahoo Finance fallback for non-SEC companies
  market.py       price, regression beta, risk-free rate, FX conversion
  classify.py     tech / AI / traditional classifier
  model.py        standard DCF, WACC, sensitivity, scenarios
  growth.py       growth-stage DCF, reverse DCF, weighted scenarios, multiples
  report.py       charts, tables, Excel export
tests/            pytest suite with synthetic SEC data
```

## Limitations

- Classification thresholds (R&D share, AI score, growth) are judgement calls; check them on a few known companies.
- Equity risk premium defaults to 4.5%; update it from [Damodaran's implied ERP](https://pages.stern.nyu.edu/~adamodar/).
- Operating leases are not treated as debt; stakes in associates (e.g. Coca-Cola's bottlers) are not added to equity value.
- Reported operating profit includes one-off charges; use `TARGET_EBIT_MARGIN` to normalize when they are large.
- Not suitable for banks and insurers. ADRs where one ADR ≠ one share (e.g. BP, Shell) should use the home ticker.
- Educational project, not investment advice.

## Author

**Eleftherios Anastasiou** · MSc Finance (Corporate Finance Honours) and MSc Accounting & Control, Vrije Universiteit Amsterdam
[LinkedIn](https://www.linkedin.com/in/YOUR_PROFILE)
