"""Cloud storage for GDELT articles and their HKU sentiment scores.

This module intentionally has no local-file fallback.  The cache is stored in
PostgreSQL, which can be Supabase PostgreSQL or Render PostgreSQL.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import date, datetime, timedelta, timezone
import os

try:
    import psycopg
except ImportError:  # The web app can still serve Factiva while DB is unconfigured.
    psycopg = None


DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
_schema_lock = threading.Lock()
_schema_ready = False


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gdelt_articles (
    article_key TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL DEFAULT 'GDELT',
    published_at TEXT NOT NULL,
    published_day DATE NOT NULL,
    date_precision TEXT NOT NULL DEFAULT 'timestamp',
    category TEXT NOT NULL DEFAULT 'market',
    summary TEXT NOT NULL DEFAULT '',
    score DOUBLE PRECISION,
    label TEXT,
    score_model TEXT,
    score_updated_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS gdelt_articles_market_day_idx
    ON gdelt_articles (market_id, published_day);

CREATE TABLE IF NOT EXISTS gdelt_coverage (
    market_id TEXT NOT NULL,
    category TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    status TEXT NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market_id, category, start_date, end_date)
);
"""


def database_configured() -> bool:
    return bool(DATABASE_URL and psycopg is not None)


def _connect():
    if not database_configured():
        return None
    return psycopg.connect(DATABASE_URL, connect_timeout=10)


def ensure_schema() -> bool:
    global _schema_ready
    if _schema_ready:
        return True
    if not database_configured():
        return False
    with _schema_lock:
        if _schema_ready:
            return True
        try:
            with _connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(SCHEMA_SQL)
            _schema_ready = True
            return True
        except Exception as exc:  # noqa: BLE001 - surfaced as an API status instead of a crash
            print(f"[database] schema initialization failed: {exc}")
            return False


def article_key(market_id: str, title: str, url: str) -> str:
    identity = (url or title or "").strip().lower()
    return hashlib.sha256(f"{market_id}|{identity}".encode("utf-8")).hexdigest()


def _coverage_rows(market_id: str, category: str) -> list[tuple[date, date]]:
    categories = [category] if category == "all" else ["all", category]
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT start_date, end_date
                FROM gdelt_coverage
                WHERE market_id = %s AND category = ANY(%s) AND status = 'ok'
                ORDER BY start_date
                """,
                (market_id, categories),
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]


def missing_ranges(market_id: str, start: date, end: date, category: str) -> list[tuple[date, date]]:
    """Return uncovered ranges, preserving the existing cloud coverage."""
    if not ensure_schema():
        return [(start, end)]
    try:
        covered = _coverage_rows(market_id, category)
    except Exception as exc:  # noqa: BLE001
        print(f"[database] coverage lookup failed: {exc}")
        return [(start, end)]

    missing = []
    cursor = start
    for covered_start, covered_end in covered:
        if covered_end < cursor:
            continue
        if covered_start > cursor:
            missing.append((cursor, min(end, covered_start - timedelta(days=1))))
        cursor = max(cursor, covered_end + timedelta(days=1))
        if cursor > end:
            break
    if cursor <= end:
        missing.append((cursor, end))
    return [(left, right) for left, right in missing if left <= right]


def load_articles(market_id: str, start: date, end: date, category: str) -> list[dict]:
    if not ensure_schema():
        return []
    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                if category == "all":
                    cursor.execute(
                        """
                        SELECT article_key, title, url, source, source_type, published_at,
                               date_precision, category, summary, score, label
                        FROM gdelt_articles
                        WHERE market_id = %s AND published_day BETWEEN %s AND %s
                        ORDER BY published_at DESC
                        """,
                        (market_id, start, end),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT article_key, title, url, source, source_type, published_at,
                               date_precision, category, summary, score, label
                        FROM gdelt_articles
                        WHERE market_id = %s AND published_day BETWEEN %s AND %s
                          AND category = %s
                        ORDER BY published_at DESC
                        """,
                        (market_id, start, end, category),
                    )
                rows = cursor.fetchall()
    except Exception as exc:  # noqa: BLE001
        print(f"[database] article lookup failed: {exc}")
        return []

    articles = []
    for row in rows:
        key, title, url, source, source_type, published_at, date_precision, row_category, summary, score, label = row
        articles.append(
            {
                "id": f"gdelt-{key[:20]}",
                "article_key": key,
                "market": market_id,
                "title": title,
                "url": url,
                "source": source,
                "source_type": source_type,
                "published_at": published_at,
                "date_precision": date_precision,
                "category": row_category,
                "type": row_category,
                "summary": summary,
                "score": score,
                "label": label,
            }
        )
    return articles


def save_articles(articles: list[dict]) -> None:
    if not articles or not ensure_schema():
        return
    values = []
    for item in articles:
        published_at = str(item.get("published_at") or "")
        published_day = published_at[:10]
        try:
            day = date.fromisoformat(published_day)
        except ValueError:
            continue
        key = item.get("article_key") or article_key(item.get("market", ""), item.get("title", ""), item.get("url", ""))
        values.append(
            (
                key,
                item.get("market", ""),
                item.get("title", "Untitled article"),
                item.get("url", ""),
                item.get("source", "Unknown source"),
                item.get("source_type", "GDELT"),
                published_at,
                day,
                item.get("date_precision", "timestamp"),
                item.get("category", "market"),
                item.get("summary", ""),
                item.get("score"),
                item.get("label"),
            )
        )
    if not values:
        return
    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO gdelt_articles
                        (article_key, market_id, title, url, source, source_type,
                         published_at, published_day, date_precision, category, summary, score, label)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_key) DO UPDATE SET
                        title = EXCLUDED.title,
                        url = EXCLUDED.url,
                        source = EXCLUDED.source,
                        published_at = EXCLUDED.published_at,
                        published_day = EXCLUDED.published_day,
                        category = EXCLUDED.category,
                        summary = EXCLUDED.summary,
                        score = COALESCE(gdelt_articles.score, EXCLUDED.score),
                        label = COALESCE(gdelt_articles.label, EXCLUDED.label)
                    """,
                    values,
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[database] article save failed: {exc}")


def mark_coverage(market_id: str, category: str, start: date, end: date) -> None:
    if not ensure_schema():
        return
    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gdelt_coverage (market_id, category, start_date, end_date, status, fetched_at)
                    VALUES (%s, %s, %s, %s, 'ok', %s)
                    ON CONFLICT (market_id, category, start_date, end_date) DO UPDATE SET
                        status = 'ok', fetched_at = EXCLUDED.fetched_at
                    """,
                    (market_id, category, start, end, datetime.now(timezone.utc)),
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[database] coverage save failed: {exc}")


def save_scores(articles: list[dict], scores: dict, model: str) -> None:
    updates = []
    for item in articles:
        if item.get("source_type") != "GDELT":
            continue
        result = scores.get(item.get("id"))
        key = item.get("article_key")
        if not key or not isinstance(result, dict):
            continue
        try:
            score = max(-1.0, min(1.0, float(result["score"])))
        except (KeyError, TypeError, ValueError):
            continue
        updates.append((score, result.get("label", "neutral"), model, datetime.now(timezone.utc), key))
    if not updates or not ensure_schema():
        return
    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    UPDATE gdelt_articles
                    SET score = %s, label = %s, score_model = %s, score_updated_at = %s
                    WHERE article_key = %s
                    """,
                    updates,
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[database] score save failed: {exc}")
