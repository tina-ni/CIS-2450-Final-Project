"""Build dashboard artifacts from papers.db.

Runs each of the three project models (K-Means, Decision Tree, Autoencoder)
on the full joined dataset and dumps the per-paper outputs the dashboard needs:

    artifacts/paper_records.parquet   Per-paper metadata + model outputs
    artifacts/summary.json            Tuned hyperparameters + headline metrics
    artifacts/autoencoder.ckpt        PyTorch Lightning checkpoint
    artifacts/dt_feature_importance.parquet   Top-20 DT features

Run from the project root:

    source .venv/bin/activate
    python scripts/build_artifacts.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import lightning as L
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "papers.db"
ART_DIR = PROJECT_ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
L.seed_everything(SEED, workers=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
JOINED_QUERY = """
SELECT o.openalex_id, o.doi, o.doi_normalized, o.title,
       o.publication_year, o.cited_by_count, o.author_count,
       o.primary_topic, o.primary_subfield, o.primary_field, o.primary_domain,
       s.abstract_text, s.abstract_tfidf_vector, s.tldr_text
FROM openalex_papers AS o
JOIN semanticscholar_papers AS s ON o.doi_normalized = s.doi_normalized
WHERE s.abstract_text IS NOT NULL       AND TRIM(s.abstract_text) <> ''
  AND s.abstract_tfidf_vector IS NOT NULL AND TRIM(s.abstract_tfidf_vector) <> ''
  AND s.tldr_text IS NOT NULL           AND TRIM(s.tldr_text) <> ''
"""

print("[1/6] Loading papers.db…")
with sqlite3.connect(DB_PATH) as conn:
    df = pl.read_database(query=JOINED_QUERY, connection=conn)
print(f"       joined rows: {df.height}")

# 70 / 15 / 15 split — identical to the K-Means and Autoencoder notebooks
split_indices = np.arange(df.height)
train_idx, temp_idx = train_test_split(split_indices, test_size=0.30, random_state=SEED, shuffle=True)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED, shuffle=True)
train_df = df[train_idx]
val_df = df[val_idx]
test_df = df[test_idx]

# ---------------------------------------------------------------------------
# 2. Feature engineering — build notebook-specific matrices
# ---------------------------------------------------------------------------
print("[2/6] Building feature matrices…")
YEAR_FILL_VALUE = float(train_df["publication_year"].median())

def build_ae_structured(frame, reference_columns=None):
    encoded = frame.select([
        pl.col("publication_year").fill_null(YEAR_FILL_VALUE).cast(pl.Float64),
        pl.col("author_count").fill_null(0).clip(lower_bound=0).log1p().alias("log_author_count"),
    ])
    if reference_columns is None:
        return encoded, encoded.columns
    for c in reference_columns:
        if c not in encoded.columns:
            encoded = encoded.with_columns(pl.lit(0).alias(c))
    return encoded.select(reference_columns), reference_columns

KM_CATEGORICAL_COLUMNS = [
    "primary_topic",
    "primary_subfield",
    "primary_field",
    "primary_domain",
]

def build_km_structured(frame, reference_columns=None):
    encoded = (
        frame.select([
            pl.col("publication_year").fill_null(YEAR_FILL_VALUE).cast(pl.Float64),
            pl.col("cited_by_count").fill_null(0).clip(lower_bound=0).log1p().alias("log_cited_by_count"),
            pl.col("author_count").fill_null(0).clip(lower_bound=0).log1p().alias("log_author_count"),
            *[pl.col(c).fill_null("Unknown") for c in KM_CATEGORICAL_COLUMNS],
        ]).to_dummies(columns=KM_CATEGORICAL_COLUMNS)
    )
    if reference_columns is None:
        return encoded, encoded.columns
    missing = [c for c in reference_columns if c not in encoded.columns]
    if missing:
        encoded = encoded.with_columns([pl.lit(0).alias(c) for c in missing])
    return encoded.select(reference_columns), reference_columns

ae_train_struct, ae_struct_cols = build_ae_structured(train_df)
ae_val_struct, _ = build_ae_structured(val_df, ae_struct_cols)
ae_test_struct, _ = build_ae_structured(test_df, ae_struct_cols)
ae_all_struct, _ = build_ae_structured(df, ae_struct_cols)

km_train_struct, km_struct_cols = build_km_structured(train_df)
km_val_struct, _ = build_km_structured(val_df, km_struct_cols)
km_test_struct, _ = build_km_structured(test_df, km_struct_cols)
km_all_struct, _ = build_km_structured(df, km_struct_cols)

def stored_tfidf_to_csr(vector_jsons):
    data, indices, indptr = [], [], [0]
    dim = None
    for vj in vector_jsons:
        v = json.loads(vj)
        if dim is None: dim = v["dimension"]
        indices.extend(v["indices"]); data.extend(v["values"])
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), shape=(len(vector_jsons), dim or 0))

abs_train = stored_tfidf_to_csr(train_df["abstract_tfidf_vector"].to_list())
abs_val   = stored_tfidf_to_csr(val_df["abstract_tfidf_vector"].to_list())
abs_test  = stored_tfidf_to_csr(test_df["abstract_tfidf_vector"].to_list())
abs_all   = stored_tfidf_to_csr(df["abstract_tfidf_vector"].to_list())

tldr_vec = TfidfVectorizer(stop_words="english", max_features=750, min_df=3)
tldr_train = tldr_vec.fit_transform(train_df["tldr_text"].to_list())
tldr_val   = tldr_vec.transform(val_df["tldr_text"].to_list())
tldr_test  = tldr_vec.transform(test_df["tldr_text"].to_list())
tldr_all   = tldr_vec.transform(df["tldr_text"].to_list())

ae_struct_scaler = StandardScaler()
ae_struct_train_s = csr_matrix(ae_struct_scaler.fit_transform(ae_train_struct.to_numpy()))
ae_struct_val_s   = csr_matrix(ae_struct_scaler.transform(ae_val_struct.to_numpy()))
ae_struct_test_s  = csr_matrix(ae_struct_scaler.transform(ae_test_struct.to_numpy()))
ae_struct_all_s   = csr_matrix(ae_struct_scaler.transform(ae_all_struct.to_numpy()))

km_struct_scaler = StandardScaler()
km_struct_train_s = csr_matrix(km_struct_scaler.fit_transform(km_train_struct.to_numpy()))
km_struct_val_s   = csr_matrix(km_struct_scaler.transform(km_val_struct.to_numpy()))
km_struct_test_s  = csr_matrix(km_struct_scaler.transform(km_test_struct.to_numpy()))
km_struct_all_s   = csr_matrix(km_struct_scaler.transform(km_all_struct.to_numpy()))

AE_X_train = hstack([ae_struct_train_s, abs_train, tldr_train], format="csr")
AE_X_val   = hstack([ae_struct_val_s,   abs_val,   tldr_val],   format="csr")
AE_X_test  = hstack([ae_struct_test_s,  abs_test,  tldr_test],  format="csr")
AE_X_all   = hstack([ae_struct_all_s,   abs_all,   tldr_all],   format="csr")

KM_X_train = hstack([km_struct_train_s, abs_train, tldr_train], format="csr")
KM_X_val   = hstack([km_struct_val_s,   abs_val,   tldr_val],   format="csr")
KM_X_test  = hstack([km_struct_test_s,  abs_test,  tldr_test],  format="csr")
KM_X_all   = hstack([km_struct_all_s,   abs_all,   tldr_all],   format="csr")
print(f"       AE feature matrix shape: {AE_X_all.shape}")
print(f"       KM feature matrix shape: {KM_X_all.shape}")

# ---------------------------------------------------------------------------
# 3. Decision Tree — notebook-aligned pipeline
# ---------------------------------------------------------------------------
print("[3/6] Fitting Decision Tree…")

def binarize_popularity(frame: pl.DataFrame, threshold: int = 1) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("cited_by_count").fill_null(0) >= threshold).alias("quickly_cited_once")
    )

dt_df = binarize_popularity(df, threshold=1)
dt_y_all = dt_df["quickly_cited_once"].to_numpy().astype(int)

dt_train_val_idx, dt_test_idx = train_test_split(
    np.arange(dt_df.height),
    test_size=0.20,
    random_state=SEED,
    stratify=dt_y_all,
)
dt_y_train_val = dt_y_all[dt_train_val_idx]
dt_train_idx, dt_val_idx = train_test_split(
    dt_train_val_idx,
    test_size=0.25,
    random_state=SEED,
    stratify=dt_y_train_val,
)

dt_train_df = dt_df[dt_train_idx]
dt_val_df = dt_df[dt_val_idx]
dt_train_val_df = dt_df[dt_train_val_idx]
dt_test_df = dt_df[dt_test_idx]

dt_y_train = dt_y_all[dt_train_idx]
dt_y_val = dt_y_all[dt_val_idx]
dt_y_train_val = dt_y_all[dt_train_val_idx]
dt_y_test = dt_y_all[dt_test_idx]

dt_cat_cols = ["primary_topic", "primary_subfield", "primary_field", "primary_domain"]

def dt_num(frame):
    return frame.select([
        pl.col("publication_year").cast(pl.Float64).alias("publication_year"),
        pl.col("author_count").fill_null(0).clip(lower_bound=0).log1p().cast(pl.Float64).alias("log_author_count"),
    ]).to_numpy()

def clean_abstract_texts(frame):
    return frame["abstract_text"].fill_null("").to_list()

dt_cat_enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
dt_cat_train = dt_cat_enc.fit_transform(dt_train_df.select(dt_cat_cols).fill_null("Unknown").to_numpy())
dt_cat_val = dt_cat_enc.transform(dt_val_df.select(dt_cat_cols).fill_null("Unknown").to_numpy())

dt_abstract_vectorizer = TfidfVectorizer(stop_words="english", max_features=2000, min_df=5)
dt_abs_train = dt_abstract_vectorizer.fit_transform(clean_abstract_texts(dt_train_df))
dt_abs_val = dt_abstract_vectorizer.transform(clean_abstract_texts(dt_val_df))

DT_X_train = hstack([csr_matrix(dt_num(dt_train_df)), dt_cat_train, dt_abs_train], format="csr")
DT_X_val = hstack([csr_matrix(dt_num(dt_val_df)), dt_cat_val, dt_abs_val], format="csr")

CANDIDATE_DEPTHS = [2, 3, 5, 8, 12, None]
dt_results = []
for depth in CANDIDATE_DEPTHS:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=SEED,
        class_weight="balanced",
        min_samples_leaf=200,
        min_samples_split=400,
    ).fit(DT_X_train, dt_y_train)
    pred = model.predict(DT_X_val)
    dt_results.append({
        "depth": depth,
        "validation_accuracy": float(accuracy_score(dt_y_val, pred)),
        "f1": float(f1_score(dt_y_val, pred, zero_division=0)),
        "actual_depth": int(model.tree_.max_depth),
    })

best_dt = max(dt_results, key=lambda r: (r["f1"], r["validation_accuracy"], -r["actual_depth"]))
best_depth = best_dt["depth"]
print(f"       best max_depth = {best_depth} (val F1 = {best_dt['f1']:.3f})")

dt_cat_enc_final = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
dt_cat_train_val = dt_cat_enc_final.fit_transform(dt_train_val_df.select(dt_cat_cols).fill_null("Unknown").to_numpy())
dt_cat_test_final = dt_cat_enc_final.transform(dt_test_df.select(dt_cat_cols).fill_null("Unknown").to_numpy())
dt_cat_all_final = dt_cat_enc_final.transform(dt_df.select(dt_cat_cols).fill_null("Unknown").to_numpy())

dt_abstract_vectorizer_final = TfidfVectorizer(stop_words="english", max_features=2000, min_df=5)
dt_abs_train_val = dt_abstract_vectorizer_final.fit_transform(clean_abstract_texts(dt_train_val_df))
dt_abs_test_final = dt_abstract_vectorizer_final.transform(clean_abstract_texts(dt_test_df))
dt_abs_all_final = dt_abstract_vectorizer_final.transform(clean_abstract_texts(dt_df))

DT_X_train_val = hstack([csr_matrix(dt_num(dt_train_val_df)), dt_cat_train_val, dt_abs_train_val], format="csr")
DT_X_test = hstack([csr_matrix(dt_num(dt_test_df)), dt_cat_test_final, dt_abs_test_final], format="csr")
DT_X_all = hstack([csr_matrix(dt_num(dt_df)), dt_cat_all_final, dt_abs_all_final], format="csr")

dt_final = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=SEED,
    class_weight="balanced",
    min_samples_leaf=200,
    min_samples_split=400,
).fit(DT_X_train_val, dt_y_train_val)
dt_pred_test = dt_final.predict(DT_X_test)
dt_prob_all = dt_final.predict_proba(DT_X_all)[:, 1].astype(np.float32)

dt_feature_names = (
    ["publication_year", "log_author_count"]
    + dt_cat_enc_final.get_feature_names_out(dt_cat_cols).tolist()
    + [f"abstract_tfidf_{term}" for term in dt_abstract_vectorizer_final.get_feature_names_out()]
)
importances = dt_final.feature_importances_
top_imp_idx = np.argsort(importances)[::-1]
top_imp_idx = [i for i in top_imp_idx if importances[i] > 0][:20]
dt_feat_df = pl.DataFrame({
    "feature": [dt_feature_names[i] for i in top_imp_idx],
    "importance": [float(importances[i]) for i in top_imp_idx],
})

dt_metrics = {
    "cutoff_citations": 1,
    "best_max_depth": "unlimited" if best_depth is None else int(best_depth),
    "accuracy": float(accuracy_score(dt_y_test, dt_pred_test)),
    "precision": float(precision_score(dt_y_test, dt_pred_test, zero_division=0)),
    "recall": float(recall_score(dt_y_test, dt_pred_test, zero_division=0)),
    "f1": float(f1_score(dt_y_test, dt_pred_test, zero_division=0)),
    "positive_rate_train": float(dt_y_train.mean()),
}

# ---------------------------------------------------------------------------
# 4. Autoencoder
# ---------------------------------------------------------------------------
print("[4/6] Fitting Autoencoder…")
SVD_DIM = 150
pre_svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED).fit(AE_X_train)
D_train = pre_svd.transform(AE_X_train).astype(np.float32)
D_val   = pre_svd.transform(AE_X_val).astype(np.float32)
D_test  = pre_svd.transform(AE_X_test).astype(np.float32)
D_all   = pre_svd.transform(AE_X_all).astype(np.float32)

svd_scaler = StandardScaler().fit(D_train)
D_train = svd_scaler.transform(D_train).astype(np.float32)
D_val   = svd_scaler.transform(D_val).astype(np.float32)
D_test  = svd_scaler.transform(D_test).astype(np.float32)
D_all   = svd_scaler.transform(D_all).astype(np.float32)

class PaperAutoencoder(L.LightningModule):
    """Symmetric MLP autoencoder over the SVD-compressed feature matrix."""
    def __init__(self, input_dim: int = SVD_DIM, hidden_dim: int = 96, latent_dim: int = 32, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def _step(self, batch, stage):
        x = batch[0]
        xhat, _ = self(x)
        loss = self.loss_fn(xhat, x)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage == "train"))
        return loss

    def training_step(self, batch, batch_idx):   return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._step(batch, "val")
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def make_loader(X, batch_size=128, shuffle=False):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

HP_EPOCHS = 20
FINAL_EPOCHS = 60
param_space = {"latent_dim": [16, 32, 64], "hidden_dim": [64, 96, 128], "lr": [1e-3, 5e-4, 1e-4]}
sampler = list(ParameterSampler(param_space, n_iter=6, random_state=SEED))

best = {"cfg": None, "val_loss": float("inf")}
for cfg in sampler:
    model = PaperAutoencoder(latent_dim=int(cfg["latent_dim"]),
                             hidden_dim=int(cfg["hidden_dim"]), lr=float(cfg["lr"]))
    trainer = L.Trainer(max_epochs=HP_EPOCHS, accelerator="cpu", enable_progress_bar=False,
                        enable_model_summary=False, logger=False, enable_checkpointing=False)
    trainer.fit(model, make_loader(D_train, shuffle=True), make_loader(D_val))
    vloss = float(trainer.callback_metrics.get("val_loss", torch.tensor(float("nan"))))
    print(f"       {cfg} -> val_loss={vloss:.4f}")
    if vloss < best["val_loss"]:
        best = {"cfg": {k: (int(v) if k in {'latent_dim','hidden_dim'} else float(v)) for k, v in cfg.items()},
                "val_loss": vloss}

print(f"       best cfg: {best['cfg']}  val_loss={best['val_loss']:.4f}")

final = PaperAutoencoder(latent_dim=best["cfg"]["latent_dim"],
                         hidden_dim=best["cfg"]["hidden_dim"], lr=best["cfg"]["lr"])
trainer = L.Trainer(max_epochs=FINAL_EPOCHS, accelerator="cpu", enable_progress_bar=False,
                    enable_model_summary=False, logger=False, enable_checkpointing=False)
trainer.fit(final, make_loader(D_train, shuffle=True), make_loader(D_val))
trainer.save_checkpoint(str(ART_DIR / "autoencoder.ckpt"))

final.eval()
with torch.no_grad():
    ae_train = final.encoder(torch.from_numpy(D_train)).numpy().astype(np.float32)
    ae_val   = final.encoder(torch.from_numpy(D_val)).numpy().astype(np.float32)
    ae_test  = final.encoder(torch.from_numpy(D_test)).numpy().astype(np.float32)
    ae_all   = final.encoder(torch.from_numpy(D_all)).numpy().astype(np.float32)

def topk_same_label_agreement(embeddings, labels, k=5):
    sims = cosine_similarity(embeddings)
    np.fill_diagonal(sims, -np.inf)
    top_idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    labels_arr = np.asarray(labels)
    neighbor_labels = labels_arr[top_idx]
    return float((neighbor_labels == labels_arr[:, None]).mean())

test_subfield = test_df["primary_subfield"].fill_null("Unknown").to_list()
ae_top5 = topk_same_label_agreement(ae_test, test_subfield, k=5)
ae_top10 = topk_same_label_agreement(ae_test, test_subfield, k=10)

LATENT = int(best["cfg"]["latent_dim"])
baseline_svd = TruncatedSVD(n_components=LATENT, random_state=SEED)
svd_train_emb = baseline_svd.fit_transform(AE_X_train)
svd_test_emb = baseline_svd.transform(AE_X_test)
svd_test_emb_norm = svd_test_emb / (np.linalg.norm(svd_test_emb, axis=1, keepdims=True) + 1e-9)
ae_test_norm = ae_test / (np.linalg.norm(ae_test, axis=1, keepdims=True) + 1e-9)
svd_top5 = topk_same_label_agreement(svd_test_emb_norm, test_subfield, k=5)
svd_top10 = topk_same_label_agreement(svd_test_emb_norm, test_subfield, k=10)
ae_top5_norm = topk_same_label_agreement(ae_test_norm, test_subfield, k=5)
ae_top10_norm = topk_same_label_agreement(ae_test_norm, test_subfield, k=10)
majority_rate = max(pl.Series(test_subfield).value_counts().get_column("count").to_list()) / len(test_subfield)

# 2D projection of the latent space for the dashboard's scatter
ae_proj = TruncatedSVD(n_components=2, random_state=SEED).fit(ae_all)
ae_2d_all = ae_proj.transform(ae_all).astype(np.float32)

# ---------------------------------------------------------------------------
# 5. K-Means — notebook-aligned clustering on the sparse feature matrix
# ---------------------------------------------------------------------------
print("[5/6] Fitting K-Means…")
K_CANDIDATES = range(2, 11)

def sampled_sil(X, labels, n=3000):
    label_set = {int(label) for label in labels}
    if X.shape[0] < 2 or len(label_set) < 2:
        return float("-inf")
    return silhouette_score(X, labels, sample_size=min(n, X.shape[0]), random_state=SEED)

km_scores = []
for k in K_CANDIDATES:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(KM_X_train)
    km_scores.append({
        "k": int(k),
        "train_inertia": float(km.inertia_),
        "val_silhouette": float(sampled_sil(KM_X_val, km.predict(KM_X_val), n=2000)),
    })
    print(f"       k={k}  val_silhouette={km_scores[-1]['val_silhouette']:.3f}")
best_km = max(km_scores, key=lambda r: r["val_silhouette"])
best_k = int(best_km["k"])
print(f"       best k = {best_k} (val silhouette = {best_km['val_silhouette']:.3f})")

kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(KM_X_train)
cluster_all = kmeans.predict(KM_X_all)
val_labels = kmeans.predict(KM_X_val)
test_labels = kmeans.predict(KM_X_test)
validation_silhouette = sampled_sil(KM_X_val, val_labels, n=2000)
test_silhouette = sampled_sil(KM_X_test, test_labels, n=2000)

km_proj = TruncatedSVD(n_components=2, random_state=SEED).fit(KM_X_train)
km_2d_all = km_proj.transform(KM_X_all).astype(np.float32)

# ---------------------------------------------------------------------------
# 6. Write artifacts
# ---------------------------------------------------------------------------
print("[6/6] Writing artifacts…")
records = df.select([
    "openalex_id", "doi", "doi_normalized", "title", "publication_year",
    "cited_by_count", "author_count", "primary_topic", "primary_subfield",
    "primary_field", "primary_domain", "tldr_text",
]).with_columns([
    pl.Series("kmeans_cluster", cluster_all.astype(np.int32)),
    pl.Series("dt_prob",        dt_prob_all),
    pl.Series("dt_pred",        (dt_prob_all >= 0.5).astype(np.int32)),
    pl.Series("quickly_cited_once", dt_y_all.astype(np.int32)),
    pl.Series("highly_cited",   dt_y_all.astype(np.int32)),
    pl.Series("ae_2d_x",        ae_2d_all[:, 0]),
    pl.Series("ae_2d_y",        ae_2d_all[:, 1]),
    pl.Series("km_2d_x",        km_2d_all[:, 0]),
    pl.Series("km_2d_y",        km_2d_all[:, 1]),
])
records.write_parquet(ART_DIR / "paper_records.parquet")
np.savez(ART_DIR / "ae_embeddings.npz", ae=ae_all)
dt_feat_df.write_parquet(ART_DIR / "dt_feature_importance.parquet")

# Also stash the fitted DT itself (for optional "why this prediction" drilldown)
joblib.dump({"dt_model": dt_final, "dt_feature_names": dt_feature_names}, ART_DIR / "dt_model.joblib")

summary = {
    "n_papers": df.height,
    "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
    "kmeans": {
        "k": best_k,
        "val_silhouette": float(validation_silhouette),
        "test_silhouette": float(test_silhouette),
        "search_results": km_scores,
    },
    "decision_tree": dt_metrics,
    "autoencoder": {
        "best_config": best["cfg"],
        "val_loss": best["val_loss"],
        "top_5_same_subfield": float(ae_top5),
        "top_10_same_subfield": float(ae_top10),
        "top_5_same_subfield_normalized": float(ae_top5_norm),
        "top_10_same_subfield_normalized": float(ae_top10_norm),
        "svd_top_5_same_subfield": float(svd_top5),
        "svd_top_10_same_subfield": float(svd_top10),
        "majority_baseline": float(majority_rate),
        "search_results": [{"cfg": {k: (int(v) if k in {'latent_dim','hidden_dim'} else float(v))
                                    for k, v in c.items()}} for c in sampler],
    },
}
(ART_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

print("Done. Artifacts written to:", ART_DIR)
for p in sorted(ART_DIR.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
