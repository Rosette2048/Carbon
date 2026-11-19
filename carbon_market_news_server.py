from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import ssl
import statistics
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
SENTIMENT_CACHE_TTL_SECONDS = 15 * 60
HKU_API_BASE_URL = os.environ.get("HKU_API_BASE_URL", "https://api.hku.hk")
HKU_API_DEPLOYMENT_ID = os.environ.get("HKU_API_DEPLOYMENT_ID", "claude-haiku-4.5")
HKU_API_KEY = os.environ.get("HKU_API_KEY", "")
HKU_API_KEY_HEADER = os.environ.get("HKU_API_KEY_HEADER", "api-key")
PUBLIC_FILES = {"/carbon_market_monitor.html", "/carbon_market_price_data.js"}

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
_sentiment_cache: dict[str, tuple[float, dict]] = {}
_sentiment_cache_lock = threading.Lock()
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


def parse_json_text(value: str) -> dict | list:
    cleaned = (value or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = min([index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0], default=-1)
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(cleaned[start:])
        return parsed


def sentiment_metrics(scores: list[float]) -> dict:
    if not scores:
        return {
            "median_score": None,
            "positive_share": None,
            "neutral_share": None,
            "negative_share": None,
            "news_count": 0,
            "score_dispersion": None,
        }
    count = len(scores)
    return {
        "median_score": round(statistics.median(scores), 4),
        "positive_share": round(sum(score > 0.1 for score in scores) / count, 4),
        "neutral_share": round(sum(-0.1 <= score <= 0.1 for score in scores) / count, 4),
        "negative_share": round(sum(score < -0.1 for score in scores) / count, 4),
        "news_count": count,
        "score_dispersion": round(statistics.pstdev(scores) if count > 1 else 0.0, 4),
    }


def sentiment_cache_key(articles: list[dict]) -> str:
    compact = [
        {
            "id": article.get("id", ""),
            "title": article.get("title", ""),
            "summary": article.get("summary", ""),
        }
        for article in articles
    ]
    return hashlib.sha256(json.dumps(compact, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def extract_hku_text(payload: dict) -> str:
    content = payload.get("output", {}).get("message", {}).get("content", [])
    return "\n".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("text"))


def fetch_hku_sentiment(articles: list[dict]) -> dict:
    if not HKU_API_KEY:
        return {
            "ok": True,
            "configured": False,
            "message": "HKU GenAI API key is not configured in the server environment.",
            "scores": {},
            "metrics": sentiment_metrics([]),
        }
    if not articles:
        return {"ok": True, "configured": True, "scores": {}, "metrics": sentiment_metrics([])}

    limited_articles = articles[:40]
    cache_key = sentiment_cache_key(limited_articles)
    now = time.time()
    with _sentiment_cache_lock:
        cached = _sentiment_cache.get(cache_key)
        if cached and now - cached[0] < SENTIMENT_CACHE_TTL_SECONDS:
            result = dict(cached[1])
            result["cached"] = True
            return result

    items = []
    for article in limited_articles:
        title = (article.get("title") or "").strip()
        summary = (article.get("summary") or "").strip()
        items.append({"id": article.get("id", ""), "title": title[:500], "summary": summary[:800]})
    prompt = (
        "You are a financial-news sentiment classifier for carbon allowance markets. "
        "Score each news item from -1.0 (strongly bearish for the relevant carbon price) "
        "to 1.0 (strongly bullish), where 0 is neutral. Return JSON only, with no markdown, "
        "using exactly this shape: {\"items\":[{\"id\":\"...\",\"score\":0.0,\"label\":\"positive|neutral|negative\"}]}. "
        "Judge the likely short-term effect on the carbon allowance price, not whether the article is morally positive.\n\n"
        + json.dumps(items, ensure_ascii=False)
    )
    endpoint = f"{HKU_API_BASE_URL.rstrip('/')}/claude/model/{HKU_API_DEPLOYMENT_ID}/converse"
    request_body = {
        "inferenceConfig": {"maxTokens": max(300, len(items) * 35), "temperature": 0},
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
    }
    request = Request(
        endpoint,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json", HKU_API_KEY_HEADER: HKU_API_KEY},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
        model_text = extract_hku_text(payload)
        decoded = parse_json_text(model_text)
        raw_items = decoded.get("items", []) if isinstance(decoded, dict) else decoded
        scores = {}
        for item in raw_items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            try:
                score = max(-1.0, min(1.0, float(item.get("score", 0))))
            except (TypeError, ValueError):
                continue
            label = item.get("label") or ("positive" if score > 0.1 else "negative" if score < -0.1 else "neutral")
            scores[item["id"]] = {"score": round(score, 4), "label": label}
        numeric_scores = [item["score"] for item in scores.values()]
        result = {
            "ok": True,
            "configured": True,
            "scores": scores,
            "metrics": sentiment_metrics(numeric_scores),
            "processed_count": len(scores),
            "requested_count": len(limited_articles),
            "cached": False,
        }
        with _sentiment_cache_lock:
            _sentiment_cache[cache_key] = (now, result)
        return result
    except HTTPError as exc:
        print(f"[sentiment] HKU API returned HTTP {exc.code}")
        return {"ok": False, "configured": True, "message": f"HKU GenAI API returned HTTP {exc.code}.", "scores": {}, "metrics": sentiment_metrics([])}
    except Exception as exc:  # noqa: BLE001 - keep provider details out of the browser response
        print(f"[sentiment] HKU API request failed: {exc}")
        return {"ok": False, "configured": True, "message": "HKU GenAI sentiment request failed.", "scores": {}, "metrics": sentiment_metrics([])}


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

    def do_POST(self) -> None:  # noqa: N802
        if not self.is_authorized():
            self.request_authentication()
            return
        if urlparse(self.path).path != "/api/sentiment":
            self.send_error(404, "Not Found")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > 2_000_000:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            articles = payload.get("articles", [])
            if not isinstance(articles, list):
                raise ValueError("articles must be a list")
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            json_response(self, {"ok": False, "message": str(exc), "scores": {}, "metrics": sentiment_metrics([])}, 400)
            return
        result = fetch_hku_sentiment(articles)
        json_response(self, result, 200 if result.get("ok") else 503)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            json_response(self, {"ok": True, "service": "carbon-market-monitor"}, 200)
            return
        if not self.is_authorized():
            self.request_authentication()
            return
        if parsed.path == "/":
            self.send_response(302)
            self.send_header("Location", "/carbon_market_monitor.html")
            self.end_headers()
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
        if parsed.path not in PUBLIC_FILES:
            self.send_error(404, "Not Found")
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
