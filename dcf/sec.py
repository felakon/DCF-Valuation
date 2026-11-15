"""
Thin client for the free SEC EDGAR APIs.

Endpoints used (no API key needed, but the SEC requires a descriptive
User-Agent with a contact e-mail, and max ~10 requests/second):

* https://www.sec.gov/files/company_tickers.json            ticker -> CIK
* https://data.sec.gov/submissions/CIK##########.json       filing index, SIC code
* https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
                                                             every XBRL fact ever filed
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}"


class SECError(RuntimeError):
    pass


@dataclass
class Filing:
    form: str
    filing_date: str
    report_date: str
    accession: str
    url: str


class SECClient:
    """Minimal, polite EDGAR client with in-memory caching."""

    def __init__(self, user_agent: str, min_interval: float = 0.15):
        if not user_agent or "@" not in user_agent or "example.com" in user_agent:
            raise SECError(
                "The SEC requires a real User-Agent, e.g. "
                "'Jane Doe jane.doe@gmail.com'. Set SEC_USER_AGENT in the notebook."
            )
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        )
        self._min_interval = min_interval
        self._last_call = 0.0

    # ------------------------------------------------------------------ HTTP
    def _get_json(self, url: str) -> dict:
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=30)
        self._last_call = time.time()
        if resp.status_code == 404:
            raise SECError(f"Not found on EDGAR: {url}")
        if resp.status_code == 403:
            raise SECError("SEC refused the request (403). Check your User-Agent string.")
        resp.raise_for_status()
        return resp.json()

    def get_text(self, url: str) -> str:
        """Raw text of a filing document (HTML), respecting the SEC rate limit."""
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=60)
        self._last_call = time.time()
        resp.raise_for_status()
        return resp.text

    # --------------------------------------------------------------- lookups
    @lru_cache(maxsize=1)
    def ticker_map(self) -> dict[str, dict]:
        raw = self._get_json(TICKERS_URL)
        return {
            row["ticker"].upper(): {"cik": int(row["cik_str"]), "name": row["title"]}
            for row in raw.values()
        }

    def lookup(self, ticker: str) -> dict:
        key = ticker.upper().strip().replace(".", "-")
        info = self.ticker_map().get(key)
        if info is None:
            raise SECError(
                f"Ticker '{ticker}' not found in SEC's list. Only SEC-registered "
                "companies are covered (US listings incl. ADRs that file with the SEC)."
            )
        return {"ticker": key, **info}

    @lru_cache(maxsize=32)
    def company_facts(self, cik: int) -> dict:
        return self._get_json(COMPANYFACTS_URL.format(cik=f"{cik:010d}"))

    @lru_cache(maxsize=32)
    def submissions(self, cik: int) -> dict:
        return self._get_json(SUBMISSIONS_URL.format(cik=f"{cik:010d}"))

    def latest_filing(self, cik: int, forms=("10-K", "10-K/A", "20-F", "20-F/A", "40-F")) -> Filing | None:
        """Most recent filing of the given form type, with a link to the document."""
        recent = self.submissions(cik).get("filings", {}).get("recent", {})
        for i, form in enumerate(recent.get("form", [])):
            if form in forms:
                accn = recent["accessionNumber"][i]
                return Filing(
                    form=form,
                    filing_date=recent["filingDate"][i],
                    report_date=recent["reportDate"][i],
                    accession=accn,
                    url=ARCHIVE_URL.format(
                        cik=cik, accn=accn.replace("-", ""), doc=recent["primaryDocument"][i]
                    ),
                )
        return None
