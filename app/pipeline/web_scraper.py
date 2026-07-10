"""Fetch and cache text from an institution's official website.

Generalized over the institution registry so the same crawler serves SRKI and
SU (or any future college) just by switching the active institution.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from app.config import Institution, settings

_SKIP_SUFFIXES = (
    ".css", ".js", ".jpeg", ".jpg", ".png", ".gif", ".webp", ".woff",
    ".woff2", ".pdf", ".ico", ".svg", ".zip", ".mp4", ".doc", ".docx",
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer", "header"}:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer", "header"} and self._skip:
            self._skip -= 1
        if tag in {"p", "h1", "h2", "h3", "h4", "li", "td", "th", "div"} and not self._skip:
            self._chunks.append("\n")

    def handle_data(self, data):
        if not self._skip and data.strip():
            self._chunks.append(data.strip())

    def text(self) -> str:
        raw = " ".join(self._chunks)
        return re.sub(r"\s+", " ", raw).strip()


def _cache_path(code: str, cache_dir: Path | None = None) -> Path:
    out = cache_dir or settings.web_cache_dir
    return out / f"{code.lower()}_web_cache.json"


def _allowed(url: str, hosts: tuple[str, ...]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == h or host.endswith("." + h) for h in hosts)


def _is_content_page(url: str) -> bool:
    lower = url.lower()
    if any(lower.endswith(ext) for ext in _SKIP_SUFFIXES):
        return False
    if "/theme/" in lower or "/assets/" in lower or "/gallery" in lower:
        return False
    return True


def fetch_html(url: str, timeout: int = 15) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": settings.web_user_agent,
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def extract_links(html: str, base_url: str, hosts: tuple[str, ...]) -> list[str]:
    links: list[str] = []
    for match in re.finditer(r'href=["\']([^"\']+)["\']', html, re.I):
        href = match.group(1).strip()
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        full = urljoin(base_url, href).split("#")[0]
        if full.startswith("http") and _allowed(full, hosts) and _is_content_page(full):
            links.append(full)
    return list(dict.fromkeys(links))


def chunk_text(text: str, size: int = 160, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        piece = " ".join(words[i : i + size])
        if len(piece) > 80:
            chunks.append(piece)
    return chunks


def scrape_site(
    inst: Institution,
    seed_urls: Iterable[str] | None = None,
    max_pages: int | None = None,
) -> list[dict]:
    seeds = list(seed_urls or inst.seed_urls)
    max_pages = max_pages or settings.web_max_pages
    hosts = inst.allowed_hosts
    seen: set[str] = set()
    pages: list[dict] = []
    queue: list[str] = list(seeds)

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            html = fetch_html(url, timeout=settings.web_request_timeout)
        except Exception as exc:
            pages.append({"url": url, "title": url, "text": "", "chunks": [], "error": str(exc)})
            continue
        text = html_to_text(html)
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else url
        if len(text) >= 60 and not re.search(r"\b404\b", title, re.I):
            pages.append(
                {
                    "url": url,
                    "title": title,
                    "text": text,
                    "chunks": chunk_text(text),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        for link in extract_links(html, url, hosts):
            if link not in seen and link not in queue:
                queue.append(link)
        time.sleep(settings.web_request_delay_sec)
    return pages


def save_cache(code: str, pages: list[dict], cache_dir: Path | None = None) -> Path:
    out = cache_dir or settings.web_cache_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "institution": code,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "pages": pages,
    }
    path = _cache_path(code, out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def cache_is_fresh(code: str, cache_dir: Path | None = None) -> bool:
    path = _cache_path(code, cache_dir)
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours <= settings.web_cache_ttl_hours


def load_cache(code: str, cache_dir: Path | None = None) -> list[dict]:
    path = _cache_path(code, cache_dir)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("pages") or []


def refresh_cache(inst: Institution, force: bool = False) -> int:
    if not settings.web_scrape_enabled:
        return 0
    if not force and cache_is_fresh(inst.code):
        return len(load_cache(inst.code))
    pages = scrape_site(inst)
    save_cache(inst.code, pages)
    return len([p for p in pages if p.get("chunks")])
