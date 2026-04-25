import json
import sqlite3
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


DB_PATH = Path(__file__).resolve().parent.parent / "papers.db"
ART_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ART_DIR.mkdir(exist_ok=True)

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

print(f"Fitting TF-IDF on {len(abstracts)} abstracts...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=2000,
    min_df=5,
)
matrix = vectorizer.fit_transform(abstracts)
print(f"Vector dimension: {matrix.shape[1]}")

# Build all (vector_json, doi) pairs then batch-update in one executemany call
print("Serializing and saving vectors...")
params = []
for doi, vector in zip(dois, matrix):
    vector = vector.tocoo()
    vector_json = json.dumps({
        "dimension": matrix.shape[1],
        "indices": vector.col.tolist(),
        "values": vector.data.tolist(),
    })
    params.append((vector_json, doi))

cur.executemany(
    "UPDATE semanticscholar_papers SET abstract_tfidf_vector = ? WHERE doi_normalized = ?",
    params,
)

conn.commit()
conn.close()

# Save vocabulary so build_artifacts.py can label features by actual word
vocab = vectorizer.get_feature_names_out().tolist()
(ART_DIR / "abstract_vocab.json").write_text(json.dumps(vocab))

print(f"Saved TF-IDF vectors for {len(rows)} abstracts.")
