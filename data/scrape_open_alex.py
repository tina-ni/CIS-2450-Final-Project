"""
Scrape research papers from the OpenAlex API.

This script fetches academic papers from OpenAlex (https://openalex.org)
filtered by field (Computer Science) and language (English), then stores
them in a SQLite database.
It supports checkpointing to resume interrupted scraping sessions.
"""

import os
import sqlite3
import time
from pathlib import Path

import requests

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Directory paths
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

# OpenAlex API credentials (loaded from environment variables)
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")
OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL")

# OpenAlex API endpoint and database file location
OPENALEX_BASE = "https://api.openalex.org/works"
DB_PATH = REPO_ROOT / "papers.db"

# Scraping parameters
YEARS = [2025]  # Publication years to scrape
TARGET = 500_000  # Target number of papers to collect per year

# OpenAlex API fields to retrieve for each paper
OPENALEX_SELECT = ",".join([
    "id",
    "doi",
    "display_name",
    "publication_year",
    "cited_by_count",
    "authorships",
    "primary_topic",
])

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_conn():
    """Create and return a new SQLite database connection."""
    return sqlite3.connect(DB_PATH)


def column_exists(conn, table_name, column_name):
    """
    Check if a column exists in a table.
    
    Args:
        conn: SQLite database connection
        table_name: Name of the table to check
        column_name: Name of the column to search for
        
    Returns:
        True if column exists, False otherwise
    """
    cur = conn.cursor()
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def normalize_doi(doi):
    """
    Normalize a DOI string to a standard format.
    
    Removes common prefixes and whitespace, converts to lowercase.
    Returns None if the DOI is empty or invalid.
    
    Args:
        doi: DOI string (may contain prefixes like https://doi.org/)
        
    Returns:
        Normalized lowercase DOI string, or None if empty
    """
    if not doi:
        return None

    # Remove common DOI prefixes and whitespace
    doi = doi.strip()
    doi = doi.removeprefix("https://doi.org/")
    doi = doi.removeprefix("http://doi.org/")
    doi = doi.removeprefix("doi:")
    doi = doi.strip().lower()

    return doi or None


def init_db():
    """
    Initialize the SQLite database with required tables.
    
    Creates:
    - openalex_papers: Table to store scraped paper data
    - openalex_checkpoints: Table to track scraping progress by year
    - Index on normalized DOI for faster lookups
    """
    conn = get_conn()
    cur = conn.cursor()

    # Create main papers table with metadata and topic classification
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

    # Add normalized DOI column if it doesn't exist (for backwards compatibility)
    if not column_exists(conn, "openalex_papers", "doi_normalized"):
        cur.execute("ALTER TABLE openalex_papers ADD COLUMN doi_normalized TEXT")

    # Create checkpoints table to track progress for resuming interrupted scrapes
    cur.execute("""
        CREATE TABLE IF NOT EXISTS openalex_checkpoints (
            year INTEGER PRIMARY KEY,
            cursor TEXT,
            finished INTEGER DEFAULT 0
        )
    """)

    # Create index on normalized DOI for efficient lookups
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_openalex_papers_doi_normalized
        ON openalex_papers(doi_normalized)
    """)

    conn.commit()
    conn.close()


def backfill_doi_normalized():
    """
    Populate the normalized DOI column for papers that are missing it.
    
    This function is useful for database migrations when the normalized DOI 
    column is added after initial data import.
    """
    conn = get_conn()
    cur = conn.cursor()

    # Find all papers with a DOI but missing normalized DOI
    rows = cur.execute("""
        SELECT openalex_id, doi
        FROM openalex_papers
        WHERE doi IS NOT NULL
          AND (doi_normalized IS NULL OR doi_normalized = '')
    """).fetchall()

    # Normalize each DOI and update the database
    for openalex_id, doi in rows:
        cur.execute("""
            UPDATE openalex_papers
            SET doi_normalized = ?
            WHERE openalex_id = ?
        """, (normalize_doi(doi), openalex_id))

    conn.commit()
    conn.close()


def request_json(method, url, *, params=None, headers=None, json_body=None, max_retries=5):
    """
    Make an HTTP request and return parsed JSON response.
    
    Implements exponential backoff retry logic for transient errors (429, 5xx).
    Returns None for 404 errors, raises exception for other errors.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: URL to request
        params: Query parameters dictionary
        headers: HTTP headers dictionary
        json_body: JSON body for POST requests
        max_retries: Maximum number of retry attempts (default: 5)
        
    Returns:
        Parsed JSON response, or None if 404 error
        
    Raises:
        RuntimeError: For non-retryable errors or failed retries
    """
    for attempt in range(max_retries):
        resp = requests.request(
            method,
            url,
            params=params,
            headers=headers,
            json=json_body,
            timeout=60,
        )

        # Success
        if resp.status_code == 200:
            return resp.json()

        # Not found - return None instead of raising error
        if resp.status_code == 404:
            return None

        # Retryable errors: rate limit, server errors
        if resp.status_code in (429, 500, 502, 503, 504):
            sleep_s = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s, ...
            print(f"Retryable error {resp.status_code}. Sleeping {sleep_s}s...")
            time.sleep(sleep_s)
            continue

        # Non-retryable error
        raise RuntimeError(f"{resp.status_code} {resp.text[:500]}")

    # Failed after all retries
    raise RuntimeError(f"Failed after {max_retries} tries: {url}")


def load_checkpoint(year):
    """
    Load the scraping progress checkpoint for a given year.
    
    Args:
        year: Publication year to load checkpoint for
        
    Returns:
        Dictionary with:
        - 'cursor': API cursor for pagination (or '*' if starting fresh)
        - 'finished': 1 if scraping complete, 0 otherwise
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT cursor, finished FROM openalex_checkpoints WHERE year = ?",
        (year,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        # No checkpoint exists, start from beginning
        return {"cursor": "*", "finished": 0}

    return {"cursor": row[0], "finished": row[1]}


def save_checkpoint(year, cursor, finished):
    """
    Save the scraping progress checkpoint for a given year.
    
    Args:
        year: Publication year to save checkpoint for
        cursor: API cursor for pagination (or None if done)
        finished: 1 if scraping is complete, 0 otherwise
    """
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
    """
    Count how many papers have been scraped for a given year.
    
    Args:
        year: Publication year to count papers for
        
    Returns:
        Number of papers in the database for the given year
    """
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
    """
    Insert or update paper records from OpenAlex API results.
    
    Uses UPSERT logic to handle duplicates: updates existing papers,
    inserts new ones. Extracts topic hierarchy (domain -> field -> subfield).
    
    Args:
        results: List of paper dictionaries from OpenAlex API response
    """
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
            len(authorships),  # Count number of authors
            primary_topic.get("display_name"),
            (primary_topic.get("subfield") or {}).get("display_name"),
            (primary_topic.get("field") or {}).get("display_name"),
            (primary_topic.get("domain") or {}).get("display_name"),
        ))

    conn.commit()
    conn.close()


def scrape_openalex_year(year, target):
    """
    Scrape papers from OpenAlex for a given year using pagination.
    
    Fetches English-language Computer Science papers (field ID 17) with DOIs
    that haven't been retracted. Implements checkpointing to resume
    interrupted scrapes. Stops when target number of papers is reached or no
    more papers available.
    
    Args:
        year: Publication year to filter by
        target: Target number of papers to collect
    """
    # Load previous progress checkpoint
    ckpt = load_checkpoint(year)
    if ckpt["finished"]:
        print(f"[OpenAlex] year {year} already finished")
        return

    cursor = ckpt["cursor"]
    saved = count_saved_rows(year)

    # Main pagination loop
    while cursor and saved < target:
        # Fetch up to remaining needed papers per page (max 100)
        per_page = min(100, target - saved)

        # Build API query parameters
        params = {
            "filter": (
                f"primary_topic.field.id:17,"
                f"has_doi:true,"
                f"is_retracted:false,"
                f"language:en,"
                f"publication_year:{year}"
            ),
            "select": OPENALEX_SELECT,
            "per_page": per_page,
            "cursor": cursor,
        }

        # Add API credentials if available
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY
        if OPENALEX_EMAIL:
            params["mailto"] = OPENALEX_EMAIL

        # Make API request
        payload = request_json("GET", OPENALEX_BASE, params=params)
        results = payload.get("results", [])

        # If no results, we've reached the end
        if not results:
            save_checkpoint(year, None, 1)
            break

        # Save papers to database
        upsert_openalex_results(results)
        saved += len(results)

        # Get pagination cursor for next batch
        next_cursor = payload.get("meta", {}).get("next_cursor")
        print(f"[OpenAlex] saved {saved}/{target}")

        # Check if target reached
        if saved >= target:
            save_checkpoint(year, next_cursor, 0)
            break

        # Check if no more pages available
        if next_cursor is None:
            save_checkpoint(year, None, 1)
            break

        # Save progress and continue
        save_checkpoint(year, next_cursor, 0)
        cursor = next_cursor

if __name__ == "__main__":
    # Initialize the database schema
    init_db()

    # Fill in any missing normalized DOI values from existing data
    backfill_doi_normalized()

    # Scrape each requested year independently so checkpoints remain year-specific.
    for year in YEARS:
        print(f"Starting OpenAlex scrape for {year}...")
        scrape_openalex_year(year, TARGET)

    print("Done.")
