from __future__ import annotations

import base64
import json
import os
import re
import secrets
import ssl
import threading
import time
from datetime import date, datetime, timedelta, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PORT = int(os.environ.get("PORT", "4174"))
HOST = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
DEMO_USERNAME = os.environ.get("DEMO_USERNAME", "demo")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "")
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
CACHE_TTL_SECONDS = 5 * 60

MARKET_QUERIES = {
    "eu": '("EU ETS" OR "European Union Emissions Trading System" OR "European carbon market" OR "European carbon allowance")',
    "uk": '("UK ETS" OR "UK Emissions Trading Scheme" OR "UK carbon market" OR "UK allowance")',
    "california": '("California Cap-and-Trade" OR "California carbon market" OR "California Carbon Allowance" OR "California emissions trading")',
}

CATEGORY_TERMS = {
    "policy": ("policy", "regulation", "law", "legislation", "parliament", "government", "reform", "directive"),
    "regulation": ("regulation", "compliance", "directive", "legislation", "rule"),
    "industry": ("industry", "utility", "steel", "cement", "aviation", "refinery", "manufacturer"),
    "energy": ("energy", "power", "electricity", "gas", "coal", "renewable", "oil"),
}

_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()
FACTIVA_PATH = Path(__file__).with_name("carbon_market_factiva_news.json")
try:
    FACTIVA_PAYLOAD = json.loads(FACTIVA_PATH.read_text(encoding="utf-8"))
    FACTIVA_NEWS = FACTIVA_PAYLOAD.get("markets", {})
    FACTIVA_GENERATED_AT = FACTIVA_PAYLOAD.get("generatedAt", "")
except (FileNotFoundError, json.JSONDecodeError):
    FACTIVA_NEWS = {}
    FACTIVA_GENERATED_AT = ""


def json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_gdelt_datetime(value: str) -> str:
    match = re.match(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z?$", value or "")
    if not match:
        return value or ""
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}Z"


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def classify_article(title: str) -> str:
    lowered = (title or "").lower()
    for category, terms in CATEGORY_TERMS.items():
        if any(term in lowered for term in terms):
            return category.title()
    return "Market"


def fetch_factiva(market_id: str, start: date, end: date, category: str) -> list[dict]:
    articles = []
    for article in FACTIVA_NEWS.get(market_id, []):
        published = article.get("published_at", "")[:10]
        if not published or not (start.isoformat() <= published <= end.isoformat()):
            continue
        if category != "all" and article.get("category", "").lower() != category.lower():
            continue
        articles.append(article)
    return articles


def fetch_gdelt(market_id: str, start: date, end: date, category: str) -> dict:
    if market_id not in MARKET_QUERIES:
        return {"ok": False, "message": "Unknown market."}

    # GDELT is intended for recent coverage. Do not pretend it can backfill the old ICAP history.
    if start < date.today() - timedelta(days=366):
        return {
            "ok": False,
            "message": "GDELT news is available for recent coverage only. Use Factiva or archived news for this historical period.",
            "articles": [],
        }

    cache_key = f"{market_id}|{start}|{end}|{category}"
    now = time.time()
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and now - cached[0] < CACHE_TTL_SECONDS:
            return {**cached[1], "cached": True}

    query = MARKET_QUERIES[market_id]
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": "75",
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d235959"),
    }
    request_url = f"{GDELT_URL}?{urlencode(params)}"
    request = Request(request_url, headers={"User-Agent": "carbon-market-monitor/0.1"})

    try:
        try:
            with urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except ssl.SSLError:
            with urlopen(request, context=ssl._create_unverified_context(), timeout=60) as response:
                raw = response.read().decode("utf-8")
        source = json.loads(raw)
    except HTTPError as exc:
        if exc.code == 429:
            return {
                "ok": False,
                "rate_limited": True,
                "message": "GDELT is temporarily rate-limiting requests. Wait a few seconds before refreshing.",
                "articles": [],
            }
        return {"ok": False, "message": f"GDELT request failed with HTTP {exc.code}.", "articles": []}
    except Exception as exc:  # noqa: BLE001 - return a useful UI error instead of crashing the server
        return {"ok": False, "message": f"GDELT request failed: {exc}", "articles": []}

    seen: set[str] = set()
    articles = []
    for item in source.get("articles", []):
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        key = normalize_title(title) or url
        if not key or key in seen:
            continue
        seen.add(key)
        article_category = classify_article(title)
        if category != "all" and article_category.lower() != category.lower():
            continue
        articles.append(
            {
                "id": f"{market_id}-{len(articles) + 1}",
                "market": market_id,
                "title": title or "Untitled article",
                "url": url,
                "source": item.get("domain") or item.get("sourcecountry") or "Unknown source",
                "source_type": "GDELT",
                "published_at": parse_gdelt_datetime(item.get("seendate", "")),
                "date_precision": "timestamp",
                "category": article_category,
                "type": article_category,
                "summary": "",
                "score": None,
            }
        )

    result = {
        "ok": True,
        "market": market_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "category": category,
        "articles": articles,
        "count": len(articles),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cached": False,
    }
    with _cache_lock:
        _cache[cache_key] = (now, result)
    return result


def merge_articles(factiva_articles: list[dict], gdelt_articles: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for article in [*factiva_articles, *gdelt_articles]:
        key = article.get("url") or normalize_title(article.get("title", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(article)
    return sorted(merged, key=lambda article: article.get("published_at", ""), reverse=True)


def fetch_combined_news(market_id: str, start: date, end: date, category: str) -> dict:
    factiva_articles = fetch_factiva(market_id, start, end, category)
    gdelt_result = None
    recent_start = max(start, date.today() - timedelta(days=366))
    recent_end = min(end, date.today())
    if recent_start <= recent_end:
        gdelt_result = fetch_gdelt(market_id, recent_start, recent_end, category)

    gdelt_articles = gdelt_result.get("articles", []) if gdelt_result else []
    articles = merge_articles(factiva_articles, gdelt_articles)
    gdelt_ok = gdelt_result is None or gdelt_result.get("ok", False)
    result = {
        "ok": bool(articles) or gdelt_ok,
        "market": market_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "category": category,
        "articles": articles,
        "count": len(articles),
        "factiva_count": len(factiva_articles),
        "gdelt_count": len(gdelt_articles),
        "factiva_generated_at": FACTIVA_GENERATED_AT,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cached": bool(gdelt_result and gdelt_result.get("cached")),
    }
    if gdelt_result and not gdelt_result.get("ok"):
        result["gdelt_message"] = gdelt_result.get("message", "GDELT did not return data.")
        result["rate_limited"] = gdelt_result.get("rate_limited", False)
    if not articles and start < date.today() - timedelta(days=366):
        result["message"] = "No Factiva articles were found for this historical date range."
    return result


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"[server] {self.address_string()} - {format % args}")

    def is_authorized(self) -> bool:
        if not DEMO_PASSWORD:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError, base64.binascii.Error):
            return False
        return secrets.compare_digest(username, DEMO_USERNAME) and secrets.compare_digest(password, DEMO_PASSWORD)

    def request_authentication(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Carbon Market Monitor"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("Demo login required.\n".encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            json_response(self, {"ok": True, "service": "carbon-market-monitor"}, 200)
            return
        if not self.is_authorized():
            self.request_authentication()
            return
        if parsed.path == "/api/news":
            query = parse_qs(parsed.query)
            try:
                market = query.get("market", ["eu"])[0]
                start = parse_date(query.get("start", [""])[0])
                end = parse_date(query.get("end", [""])[0])
                category = query.get("category", ["all"])[0].lower()
                if start > end:
                    raise ValueError("start date must not be after end date")
            except ValueError as exc:
                json_response(self, {"ok": False, "message": str(exc), "articles": []}, 400)
                return
            result = fetch_combined_news(market, start, end, category)
            json_response(self, result, 200 if result.get("ok") else 422)
            return
        super().do_GET()


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    display_host = "127.0.0.1" if HOST == "0.0.0.0" else HOST
    print(f"Carbon Market Monitor running at http://{display_host}:{PORT}/carbon_market_monitor.html")
    print("News endpoint: /api/news?market=eu&start=YYYY-MM-DD&end=YYYY-MM-DD&category=all")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
