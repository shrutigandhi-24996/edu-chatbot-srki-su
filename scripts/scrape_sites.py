"""Scrape an institution's official website into the local web cache.

Usage:
  python scripts/scrape_sites.py --institution SRKI
  python scripts/scrape_sites.py --institution SU --max-pages 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.pipeline import web_scraper  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--institution", "-i", default="SRKI", choices=["SRKI", "SU"])
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args()

    inst = settings.institution(args.institution)
    print(f"Scraping {inst.full_name} ({inst.website}) ...")
    pages = web_scraper.scrape_site(inst, max_pages=args.max_pages)
    path = web_scraper.save_cache(inst.code, pages)
    good = len([p for p in pages if p.get("chunks")])
    print(f"Saved {good}/{len(pages)} content pages -> {path}")


if __name__ == "__main__":
    main()
