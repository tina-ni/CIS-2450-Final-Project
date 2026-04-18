import json
import sqlite3
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


DB_PATH = Path(__file__).resolve().parent.parent / "papers.db"


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

columns = {row[1] for row in cur.execute("PRAGMA table_info(semanticscholar_papers)")}
if "abstract_tfidf_vector" not in columns:
    cur.execute("ALTER TABLE semanticscholar_papers ADD COLUMN abstract_tfidf_vector TEXT")

rows = cur.execute("""
    SELECT doi_normalized, abstract_text
    FROM semanticscholar_papers
    WHERE abstract_text IS NOT NULL
      AND TRIM(abstract_text) != ''
    ORDER BY doi_normalized
""").fetchall()

dois = [doi for doi, _ in rows]
abstracts = [abstract for _, abstract in rows]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=2000,
    min_df=5,
)
matrix = vectorizer.fit_transform(abstracts)

for doi, vector in zip(dois, matrix):
    vector = vector.tocoo()
    vector_json = json.dumps({
        "dimension": matrix.shape[1],
        "indices": vector.col.tolist(),
        "values": vector.data.tolist(),
    })

    cur.execute("""
        UPDATE semanticscholar_papers
        SET abstract_tfidf_vector = ?
        WHERE doi_normalized = ?
    """, (vector_json, doi))

conn.commit()
conn.close()

print(f"Saved TF-IDF vectors for {len(rows)} abstracts.")
print(f"Vector dimension: {matrix.shape[1]}")
