"""
Decide which valuation model fits a company.

Signals
-------
1. SIC industry code from EDGAR (computers, semiconductors, software, ...)
2. R&D intensity (R&D / revenue) - catches tech companies with "non-tech" SIC codes
3. Revenue growth (3-year CAGR and latest year)
4. AI score - how often the latest 10-K talks about AI, per 10,000 words

Routing
-------
* AI / hyper-growth tech  -> growth-stage DCF (long high-growth period)
* Growth tech             -> growth-stage DCF (shorter high-growth period)
* Mature tech, Traditional, Financial -> standard 5-year DCF

The thresholds are judgement calls and live in one place (THRESHOLDS) so they are
easy to tune. Running a few known names (e.g. NVDA, MSFT, KO) is a good calibration check.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

import numpy as np

from .financials import Financials

# SIC ranges treated as technology.
TECH_SIC_RANGES = [
    (3570, 3579, "Computer & office equipment"),
    (3660, 3679, "Communications equipment, semiconductors & electronic components"),
    (3820, 3829, "Measuring & control instruments"),
    (7370, 7379, "Software, internet & IT services"),
]

# Used when there is no SIC code (companies loaded from Yahoo Finance).
TECH_INDUSTRY_WORDS = ("semiconductor", "software", "internet", "information technology",
                       "computer", "electronic component", "communication equipment",
                       "scientific & technical instruments", "consumer electronics")

THRESHOLDS = {
    "rnd_intensity_tech": 0.08,   # R&D >= 8% of revenue counts as tech
    "ai_score_driven": 20.0,      # AI mentions per 10,000 words -> "AI-driven"
    "ai_score_exposed": 5.0,      # -> "AI-exposed"
    "hyper_growth": 0.30,         # 3-yr revenue CAGR
    "growth": 0.10,
}

AI_PATTERNS = [
    re.compile(r"\bartificial intelligence\b", re.I),
    re.compile(r"\bmachine learning\b", re.I),
    re.compile(r"\bdeep learning\b", re.I),
    re.compile(r"\bneural networks?\b", re.I),
    re.compile(r"\blarge language models?\b", re.I),
    re.compile(r"\bLLMs?\b"),
    re.compile(r"\bgenerative\b", re.I),
    re.compile(r"\bAI\b"),            # case-sensitive on purpose
]


@dataclass
class Classification:
    profile: str                 # e.g. "AI / hyper-growth tech"
    model: str                   # "growth" or "standard"
    is_tech: bool
    tech_reason: str | None
    rnd_intensity: float
    revenue_cagr: float
    latest_growth: float
    ai_score: float | None       # mentions per 10,000 words in the latest 10-K
    ai_mentions: int | None
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Profile: **{self.profile}** -> {self.model} DCF"]
        lines += [f"- {r}" for r in self.reasons]
        return "\n".join(lines)


# ----------------------------------------------------------------------------
def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|ix:header)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw))


def ai_score(text: str) -> tuple[float, int, int]:
    """(mentions per 10,000 words, total mentions, word count)."""
    words = len(text.split())
    hits = sum(len(p.findall(text)) for p in AI_PATTERNS)
    return (hits / words * 10_000 if words else 0.0), hits, words


def tech_sic(sic) -> str | None:
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return None
    for lo, hi, label in TECH_SIC_RANGES:
        if lo <= code <= hi:
            return label
    return None


def _growth_stats(fin: Financials) -> tuple[float, float]:
    rev = fin.annual.loc["revenue"].dropna()
    if len(rev) < 2:
        return np.nan, np.nan
    k = min(3, len(rev) - 1)
    cagr = (rev.iloc[-1] / rev.iloc[-1 - k]) ** (1 / k) - 1
    latest = rev.iloc[-1] / rev.iloc[-2] - 1
    return float(cagr), float(latest)


def classify(fin: Financials, filing_text: str | None = None) -> Classification:
    """Classify a company. Pass the 10-K text to enable the AI score."""
    t = THRESHOLDS
    a = fin.annual
    rev_last = a.loc["revenue"].dropna()
    rnd_last = a.loc["rnd"].dropna() if "rnd" in a.index else rev_last.iloc[0:0]
    rnd_int = float(rnd_last.iloc[-1] / rev_last.iloc[-1]) if len(rnd_last) and len(rev_last) else 0.0
    cagr, latest = _growth_stats(fin)

    reasons: list[str] = []
    sic_label = tech_sic(fin.sic)
    industry = (fin.industry or "").lower()
    if sic_label:
        tech_reason = f"SIC {fin.sic}: {sic_label}"
    elif fin.sic is None and (fin.sector == "Technology"
                              or any(w in industry for w in TECH_INDUSTRY_WORDS)):
        tech_reason = f"Yahoo sector/industry: {fin.sector} / {fin.industry}"
    elif rnd_int >= t["rnd_intensity_tech"]:
        tech_reason = f"R&D is {rnd_int:.1%} of revenue (>= {t['rnd_intensity_tech']:.0%})"
    else:
        tech_reason = None
    is_tech = tech_reason is not None
    if tech_reason is None:
        where = (f"SIC {fin.sic} ({fin.sic_description})" if fin.sic
                 else f"Industry '{fin.industry or 'unknown'}'")
        tech_reason_text = f"{where} is not a tech industry; R&D {rnd_int:.1%} of revenue"
    reasons.append(tech_reason or tech_reason_text)
    if np.isfinite(cagr):
        reasons.append(f"Revenue growth: {cagr:.1%} 3-yr CAGR, {latest:.1%} last year")

    score = mentions = None
    if filing_text:
        score, mentions, words = ai_score(filing_text)
        form = fin.latest_10k.form if fin.latest_10k else "annual report"
        reasons.append(f"AI score: {mentions} AI mentions in {words:,} words of the {form} "
                       f"= {score:.1f} per 10,000 words")
    ai_driven = score is not None and score >= t["ai_score_driven"]
    ai_exposed = score is not None and score >= t["ai_score_exposed"]

    if fin.is_financial:
        profile, model = "Financial", "standard"
        reasons.append("Banks/insurers need a dividend or excess-return model; standard DCF shown with caution")
    elif is_tech and (ai_driven or (np.isfinite(cagr) and cagr >= t["hyper_growth"])):
        profile, model = "AI / hyper-growth tech", "growth"
    elif is_tech and np.isfinite(cagr) and cagr >= t["growth"]:
        profile, model = "Growth tech", "growth"
    elif is_tech:
        profile, model = "Mature tech", "standard"
    else:
        profile, model = "Traditional", "standard"
        if ai_exposed:
            reasons.append("Mentions AI often, but is not a tech business -> standard model")

    return Classification(profile=profile, model=model, is_tech=is_tech, tech_reason=tech_reason,
                          rnd_intensity=rnd_int, revenue_cagr=cagr, latest_growth=latest,
                          ai_score=score, ai_mentions=mentions, reasons=reasons)


def fetch_10k_text(client, fin: Financials) -> str | None:
    """Download and clean the latest annual report (10-K or 20-F); None if unavailable."""
    if client is None or fin.latest_10k is None:
        return None
    try:
        return html_to_text(client.get_text(fin.latest_10k.url))
    except Exception:  # noqa: BLE001 - the AI score is optional
        return None
