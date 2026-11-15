"""Command line: python -m dcf AAPL --user-agent "Jane Doe jane@gmail.com" [--excel]"""
import argparse
import os

from . import value_company, wacc_growth_grid
from .report import export_excel


def main():
    p = argparse.ArgumentParser(description="DCF valuation from the latest SEC filings")
    p.add_argument("ticker")
    p.add_argument("--user-agent", default=os.getenv("SEC_USER_AGENT"),
                   help="'Name email' required by the SEC (or set SEC_USER_AGENT)")
    p.add_argument("--fy-only", action="store_true", help="use last 10-K instead of TTM")
    p.add_argument("--erp", type=float, help="equity risk premium, e.g. 0.045")
    p.add_argument("--terminal-growth", type=float)
    p.add_argument("--excel", action="store_true", help="write <TICKER>_DCF.xlsx")
    args = p.parse_args()

    overrides = {k: v for k, v in {"erp": args.erp, "terminal_growth": args.terminal_growth}.items()
                 if v is not None}
    fin, mkt, res = value_company(args.ticker, args.user_agent, use_ttm=not args.fy_only, **overrides)
    print(f"\n{res.company} ({res.ticker}) - base period: {res.base_label}\n")
    print(res.summary().to_string())
    for w in res.warnings:
        print(f"! {w}")
    if args.excel:
        grid = wacc_growth_grid(fin, mkt, res.assumptions, res, use_ttm=not args.fy_only)
        print(f"\nSaved {export_excel(fin, res, grid)}")


if __name__ == "__main__":
    main()
