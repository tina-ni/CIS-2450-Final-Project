import os
import sqlite3
import time
from pathlib import Path
from urllib.parse import quote

import requests

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

S2_BASE = "https://api.semanticscholar.org/graph/v1"
DB_PATH = REPO_ROOT / "papers.db"

SLEEP_SECONDS = 1.1


def get_conn():
    return sqlite3.connect(DB_PATH)


def normalize_doi(doi):
    if not doi:
        return None

    doi = doi.strip()
    doi = doi.removeprefix("https://doi.org/")
    doi = doi.removeprefix("http://doi.org/")
    doi = doi.removeprefix("doi:")
    doi = doi.strip().lower()

    return doi or None


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


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS semanticscholar_papers (
            doi_normalized TEXT PRIMARY KEY,
            openalex_id TEXT,
            s2_found INTEGER NOT NULL,
            tldr_text TEXT
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_semanticscholar_openalex_id
        ON semanticscholar_papers(openalex_id)
    """)

    conn.commit()
    conn.close()


def fetch_s2_tldr_for_doi(doi_normalized):
    if not doi_normalized:
        return None

    paper_id = f"DOI:{doi_normalized}"
    encoded_paper_id = quote(paper_id, safe="")
    url = f"{S2_BASE}/paper/{encoded_paper_id}"

    params = {"fields": "tldr"}
    headers = {}

    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    return request_json("GET", url, params=params, headers=headers)


def upsert_s2_result(openalex_id, doi_normalized, payload):
    conn = get_conn()
    cur = conn.cursor()

    if payload is None:
        cur.execute("""
            INSERT INTO semanticscholar_papers (
                doi_normalized,
                openalex_id,
                s2_found,
                tldr_text
            )
            VALUES (?, ?, 0, NULL)
            ON CONFLICT(doi_normalized) DO UPDATE SET
                openalex_id = excluded.openalex_id,
                s2_found = excluded.s2_found,
                tldr_text = excluded.tldr_text
        """, (doi_normalized, openalex_id))
    else:
        tldr_text = (payload.get("tldr") or {}).get("text")

        cur.execute("""
            INSERT INTO semanticscholar_papers (
                doi_normalized,
                openalex_id,
                s2_found,
                tldr_text
            )
            VALUES (?, ?, 1, ?)
            ON CONFLICT(doi_normalized) DO UPDATE SET
                openalex_id = excluded.openalex_id,
                s2_found = excluded.s2_found,
                tldr_text = excluded.tldr_text
        """, (doi_normalized, openalex_id, tldr_text))

    conn.commit()
    conn.close()


def get_pending_openalex_rows(limit=None):
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        SELECT oa.openalex_id, oa.doi_normalized
        FROM openalex_papers oa
        WHERE oa.doi_normalized IS NOT NULL
          AND oa.doi_normalized != ''
          AND NOT EXISTS (
              SELECT 1
              FROM semanticscholar_papers s2
              WHERE s2.doi_normalized = oa.doi_normalized
          )
        ORDER BY oa.doi_normalized
    """

    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    rows = cur.execute(sql).fetchall()
    conn.close()
    return rows


def enrich_semantic_scholar(limit=None):
    rows = get_pending_openalex_rows(limit=limit)

    for i, (openalex_id, doi_normalized) in enumerate(rows, start=1):
        payload = fetch_s2_tldr_for_doi(doi_normalized)
        upsert_s2_result(openalex_id, doi_normalized, payload)

        print(f"[S2] {i}/{len(rows)} {doi_normalized}")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    init_db()
    enrich_semantic_scholar(limit=None)
    print("Done.")
