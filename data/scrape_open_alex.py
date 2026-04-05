import os
import sqlite3
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")
OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL")

OPENALEX_BASE = "https://api.openalex.org/works"
DB_PATH = REPO_ROOT / "papers.db"

YEAR = 2026
TARGET = 200

OPENALEX_SELECT = ",".join([
    "id",
    "doi",
    "display_name",
    "publication_year",
    "cited_by_count",
    "authorships",
    "primary_topic",
])


def get_conn():
    return sqlite3.connect(DB_PATH)


def column_exists(conn, table_name, column_name):
    cur = conn.cursor()
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def normalize_doi(doi):
    if not doi:
        return None

    doi = doi.strip()
    doi = doi.removeprefix("https://doi.org/")
    doi = doi.removeprefix("http://doi.org/")
    doi = doi.removeprefix("doi:")
    doi = doi.strip().lower()

    return doi or None


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS openalex_papers (
            openalex_id TEXT PRIMARY KEY,
            doi TEXT,
            doi_normalized TEXT,
            title TEXT,
            publication_year INTEGER,
            cited_by_count INTEGER,
            author_count INTEGER,
            primary_topic TEXT,
            primary_subfield TEXT,
            primary_field TEXT,
            primary_domain TEXT
        )
    """)

    if not column_exists(conn, "openalex_papers", "doi_normalized"):
        cur.execute("ALTER TABLE openalex_papers ADD COLUMN doi_normalized TEXT")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS openalex_checkpoints (
            year INTEGER PRIMARY KEY,
            cursor TEXT,
            finished INTEGER DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_openalex_papers_doi_normalized
        ON openalex_papers(doi_normalized)
    """)

    conn.commit()
    conn.close()


def backfill_doi_normalized():
    conn = get_conn()
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT openalex_id, doi
        FROM openalex_papers
        WHERE doi IS NOT NULL
          AND (doi_normalized IS NULL OR doi_normalized = '')
    """).fetchall()

    for openalex_id, doi in rows:
        cur.execute("""
            UPDATE openalex_papers
            SET doi_normalized = ?
            WHERE openalex_id = ?
        """, (normalize_doi(doi), openalex_id))

    conn.commit()
    conn.close()


def request_json(method, url, *, params=None, headers=None, json_body=None, max_retries=5):
    for attempt in range(max_retries):
        resp = requests.request(
            method,
            url,
            params=params,
            headers=headers,
            json=json_body,
            timeout=60,
        )

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 404:
            return None

        if resp.status_code in (429, 500, 502, 503, 504):
            sleep_s = 2 ** attempt
            print(f"Retryable error {resp.status_code}. Sleeping {sleep_s}s...")
            time.sleep(sleep_s)
            continue

        raise RuntimeError(f"{resp.status_code} {resp.text[:500]}")

    raise RuntimeError(f"Failed after {max_retries} tries: {url}")


def load_checkpoint(year):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT cursor, finished FROM openalex_checkpoints WHERE year = ?",
        (year,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return {"cursor": "*", "finished": 0}

    return {"cursor": row[0], "finished": row[1]}


def save_checkpoint(year, cursor, finished):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO openalex_checkpoints (year, cursor, finished)
        VALUES (?, ?, ?)
        ON CONFLICT(year) DO UPDATE SET
            cursor = excluded.cursor,
            finished = excluded.finished
    """, (year, cursor, finished))
    conn.commit()
    conn.close()


def count_saved_rows(year):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM openalex_papers WHERE publication_year = ?",
        (year,),
    )
    count = cur.fetchone()[0]
    conn.close()
    return count


def upsert_openalex_results(results):
    conn = get_conn()
    cur = conn.cursor()

    for paper in results:
        authorships = paper.get("authorships") or []
        primary_topic = paper.get("primary_topic") or {}
        doi = paper.get("doi")

        cur.execute("""
            INSERT INTO openalex_papers (
                openalex_id,
                doi,
                doi_normalized,
                title,
                publication_year,
                cited_by_count,
                author_count,
                primary_topic,
                primary_subfield,
                primary_field,
                primary_domain
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(openalex_id) DO UPDATE SET
                doi = excluded.doi,
                doi_normalized = excluded.doi_normalized,
                title = excluded.title,
                publication_year = excluded.publication_year,
                cited_by_count = excluded.cited_by_count,
                author_count = excluded.author_count,
                primary_topic = excluded.primary_topic,
                primary_subfield = excluded.primary_subfield,
                primary_field = excluded.primary_field,
                primary_domain = excluded.primary_domain
        """, (
            paper.get("id"),
            doi,
            normalize_doi(doi),
            paper.get("display_name"),
            paper.get("publication_year"),
            paper.get("cited_by_count"),
            len(authorships),
            primary_topic.get("display_name"),
            (primary_topic.get("subfield") or {}).get("display_name"),
            (primary_topic.get("field") or {}).get("display_name"),
            (primary_topic.get("domain") or {}).get("display_name"),
        ))

    conn.commit()
    conn.close()


def scrape_openalex_year(year, target):
    ckpt = load_checkpoint(year)
    if ckpt["finished"]:
        print(f"[OpenAlex] year {year} already finished")
        return

    cursor = ckpt["cursor"]
    saved = count_saved_rows(year)

    while cursor and saved < target:
        per_page = min(100, target - saved)

        params = {
            "filter": f"primary_topic.field.id:17,has_doi:true,is_retracted:false,publication_year:{year}",
            "select": OPENALEX_SELECT,
            "per_page": per_page,
            "cursor": cursor,
        }

        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY
        if OPENALEX_EMAIL:
            params["mailto"] = OPENALEX_EMAIL

        payload = request_json("GET", OPENALEX_BASE, params=params)
        results = payload.get("results", [])

        if not results:
            save_checkpoint(year, None, 1)
            break

        upsert_openalex_results(results)
        saved += len(results)

        next_cursor = payload.get("meta", {}).get("next_cursor")
        print(f"[OpenAlex] saved {saved}/{target}")

        if saved >= target:
            save_checkpoint(year, next_cursor, 0)
            break

        if next_cursor is None:
            save_checkpoint(year, None, 1)
            break

        save_checkpoint(year, next_cursor, 0)
        cursor = next_cursor


if __name__ == "__main__":
    init_db()
    backfill_doi_normalized()
    scrape_openalex_year(YEAR, TARGET)
    print("Done.")
