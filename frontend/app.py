"""Research Paper Similarity Explorer — CIS 2450 Final Project Dashboard.

Run:
    source .venv/bin/activate
    python frontend/app.py          # open http://127.0.0.1:8050

The dashboard expects artifacts produced by scripts/build_artifacts.py.
If they are missing, a banner at the top tells you which command to run.

Tabs
----
1. Overview & EDA      High-level stats + the three EDA charts that informed modeling.
2. K-Means Clusters    2D scatter of unsupervised paper clusters, filterable + annotated.
3. Citation Predictor  Decision-tree "cited at least once?" probability for any existing paper.
4. Similarity Search   Autoencoder-based nearest neighbors for any paper (headline feature).
"""
from __future__ import annotations

import json
from pathlib import Path

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Input, Output, State, dash_table, dcc, html
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths and artifact loading
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ART  = ROOT / "artifacts"
DB_PATH = ROOT / "papers.db"

ARTIFACT_FILES = {
    "paper_records":          ART / "paper_records.parquet",
    "ae_embeddings":          ART / "ae_embeddings.npz",
    "dt_feature_importance":  ART / "dt_feature_importance.parquet",
    "summary":                ART / "summary.json",
}

missing_artifacts = [name for name, p in ARTIFACT_FILES.items() if not p.exists()]
ARTIFACTS_OK = not missing_artifacts

if ARTIFACTS_OK:
    papers    = pl.read_parquet(ARTIFACT_FILES["paper_records"])
    ae_emb    = np.load(ARTIFACT_FILES["ae_embeddings"])["ae"]
    dt_feat   = pl.read_parquet(ARTIFACT_FILES["dt_feature_importance"])
    summary   = json.loads(ARTIFACT_FILES["summary"].read_text())
else:
    # Minimal defaults so the app can still start and show the "run build" banner.
    papers    = pl.DataFrame()
    ae_emb    = np.zeros((0, 32), dtype=np.float32)
    dt_feat   = pl.DataFrame()
    summary   = {}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
SUBFIELDS_SORTED = (
    sorted(papers["primary_subfield"].drop_nulls().unique().to_list())
    if ARTIFACTS_OK else []
)
TOPICS_SORTED = (
    sorted(papers["primary_topic"].drop_nulls().unique().to_list())
    if ARTIFACTS_OK else []
)


def resolve_paper_index(query: str) -> int | None:
    """Map a title / DOI / OpenAlex ID / row-index to a row index in `papers`."""
    if not query or not ARTIFACTS_OK:
        return None
    q = query.strip().lower()
    if not q:
        return None

    # Integer row index
    try:
        i = int(q)
        if 0 <= i < papers.height:
            return i
    except ValueError:
        pass

    # Exact identifier match on several columns
    for col in ["openalex_id", "doi", "doi_normalized", "title"]:
        mask = papers[col].fill_null("").str.to_lowercase() == q
        hits = np.where(mask.to_numpy())[0]
        if len(hits) == 1:
            return int(hits[0])

    # Case-insensitive title substring match — first hit wins
    title_lc = papers["title"].fill_null("").str.to_lowercase().to_numpy()
    for i, t in enumerate(title_lc):
        if q in t:
            return i
    return None


def ranked_neighbors(query_idx: int, k: int = 10) -> pl.DataFrame:
    """Top-k autoencoder-latent cosine neighbors (excluding the query itself)."""
    q = ae_emb[query_idx:query_idx + 1]
    sims = cosine_similarity(q, ae_emb).ravel()
    sims[query_idx] = -np.inf
    top = np.argpartition(-sims, kth=k)[:k]
    top = top[np.argsort(-sims[top])]
    return (
        papers[top.tolist()]
        .select(["title", "primary_subfield", "publication_year", "cited_by_count"])
        .with_columns(pl.Series("similarity", [float(s) for s in sims[top]]))
        .select(["similarity", "title", "primary_subfield", "publication_year", "cited_by_count"])
    )


def kmeans_cluster_neighbors(query_idx: int, k: int = 10) -> pl.DataFrame:
    """Top-k papers in the same K-Means cluster by autoencoder-latent cosine."""
    q = ae_emb[query_idx:query_idx + 1]
    cluster = int(papers["kmeans_cluster"][query_idx])
    mask = (papers["kmeans_cluster"].to_numpy() == cluster)
    mask[query_idx] = False
    idx_in_cluster = np.where(mask)[0]
    if len(idx_in_cluster) == 0:
        return pl.DataFrame()
    sims = cosine_similarity(q, ae_emb[idx_in_cluster]).ravel()
    order = np.argsort(-sims)[:k]
    chosen = idx_in_cluster[order]
    return (
        papers[chosen.tolist()]
        .select(["title", "primary_subfield", "publication_year", "cited_by_count"])
        .with_columns(pl.Series("similarity", [float(s) for s in sims[order]]))
        .select(["similarity", "title", "primary_subfield", "publication_year", "cited_by_count"])
    )


def _card(label: str, value: str, sub: str = "") -> html.Div:
    return html.Div(
        [html.Div(value, className="stat-value"),
         html.Div(label, className="stat-label"),
         html.Div(sub, className="stat-sub")],
        className="stat-card",
    )


# ---------------------------------------------------------------------------
# Tab 1: Overview & EDA
# ---------------------------------------------------------------------------
def eda_layout():
    if not ARTIFACTS_OK:
        return html.Div("Run scripts/build_artifacts.py to populate this tab.")
    n = papers.height
    sub_counts = papers["primary_subfield"].value_counts().sort("count", descending=True)
    top_sub = sub_counts[0, "primary_subfield"]
    avg_authors = float(papers["author_count"].mean())
    label_col = "quickly_cited_once" if "quickly_cited_once" in papers.columns else "highly_cited"
    pct_cited = 100 * float(papers[label_col].mean())

    return html.Div([
        dcc.Markdown(
            "### Overview of the dataset\n"
            "Every paper in this dashboard comes from the **OpenAlex ↔ Semantic Scholar** join "
            "on normalized DOI (a record-linking step). Only papers with an abstract, a stored "
            "abstract TF-IDF vector, and a non-empty TLDR survive the filter, so every row has "
            "the three feature blocks the downstream models use."
        ),
        html.Div([
            _card("Papers after join", f"{n:,}"),
            _card("Subfields", f"{len(SUBFIELDS_SORTED)}"),
            _card("Top subfield", top_sub, f"{sub_counts[0, 'count']:,} papers"),
            _card("Avg. authors / paper", f"{avg_authors:.1f}"),
            _card("% with \u22651 citation", f"{pct_cited:.1f}%"),
        ], className="stat-row"),

        dcc.Markdown("#### Where the papers come from"),
        dcc.Graph(id="eda-subfield", figure=_subfield_bar()),
        dcc.Markdown(
            "*The dataset is heavy on AI / Computer Vision / NLP because Semantic Scholar's "
            "TLDR coverage is strongest in CS and biomed. Anything we say about 'subfield "
            "clusters' should be read with this skew in mind.*"
        ),

        dcc.Markdown("#### How citations are distributed"),
        dcc.Graph(id="eda-citations", figure=_citation_hist()),
        dcc.Markdown(
            "*The Decision Tree notebook uses a simple binary target: whether a paper has at "
            "least one citation. That makes the model easy to interpret, but it is still only "
            "a coarse proxy for impact.*"
        ),

        dcc.Markdown("#### Subfield \u00d7 K-Means cluster composition"),
        dcc.Graph(id="eda-heatmap", figure=_heatmap_subfield_cluster()),
        dcc.Markdown(
            "*Reading across rows shows how 'pure' each cluster is — bright diagonals mean "
            "K-Means recovered subfield structure unsupervised. Off-diagonal brightness is "
            "where K-Means found groupings the subfield labels don't capture.*"
        ),
    ])


def _subfield_bar():
    sub = (
        papers["primary_subfield"].value_counts()
        .sort("count", descending=True).head(15)
        .rename({"primary_subfield": "subfield"})
    )
    fig = px.bar(sub.to_pandas(), x="count", y="subfield", orientation="h",
                 labels={"count": "Papers", "subfield": ""},
                 title="Top 15 primary subfields")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420, margin={"l": 180})
    return fig


def _citation_hist():
    arr = papers["cited_by_count"].fill_null(0).to_numpy()
    fig = px.histogram(x=np.log1p(arr), nbins=30,
                       labels={"x": "log(1 + citations)"},
                       title="Citation count (log-1-plus scale)")
    fig.update_layout(showlegend=False, height=360, bargap=0.05)
    return fig


def _heatmap_subfield_cluster():
    top_sub = [r[0] for r in papers["primary_subfield"].value_counts()
               .sort("count", descending=True).head(15).iter_rows()]
    clusters = sorted(papers["kmeans_cluster"].unique().to_list())
    pd_papers = papers.to_pandas()
    pivot = (
        pd_papers[pd_papers["primary_subfield"].isin(top_sub)]
        .pivot_table(index="primary_subfield", columns="kmeans_cluster",
                     values="openalex_id", aggfunc="count", fill_value=0)
        .reindex(index=top_sub, columns=clusters, fill_value=0)
    )
    pivot_norm = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0)
    fig = px.imshow(pivot_norm.values,
                    x=[f"C{c}" for c in pivot_norm.columns],
                    y=list(pivot_norm.index),
                    aspect="auto", color_continuous_scale="Blues",
                    labels={"x": "K-Means cluster", "y": "Subfield", "color": "Row share"})
    fig.update_layout(height=480, margin={"l": 170},
                      title="Share of each subfield's papers per cluster")
    return fig


# ---------------------------------------------------------------------------
# Tab 2: K-Means
# ---------------------------------------------------------------------------
def _cluster_label(c: int) -> str:
    """Return 'Cluster N · Top Subfield' for dropdown readability."""
    top = (
        papers.filter(pl.col("kmeans_cluster") == c)["primary_subfield"]
        .drop_nulls().value_counts().sort("count", descending=True)
    )
    label = top[0, "primary_subfield"] if top.height > 0 else "Mixed"
    return f"Cluster {c} · {label}"


def kmeans_layout():
    if not ARTIFACTS_OK:
        return html.Div("Run scripts/build_artifacts.py to populate this tab.")
    km = summary.get("kmeans", {})
    clusters = sorted(papers["kmeans_cluster"].unique().to_list())

    return html.Div([
        dcc.Markdown(
            f"### K-Means paper clusters\n"
            f"Unsupervised clustering on the sparse feature matrix (structured + "
            f"abstract TF-IDF + TLDR TF-IDF). Selected **k = {km.get('k','?')}** by validation "
            f"silhouette ({km.get('val_silhouette', 0):.3f}); held-out test silhouette "
            f"{km.get('test_silhouette', 0):.3f}. Points are projected to 2D with "
            f"TruncatedSVD fit on the training split."
        ),

        html.Div([
            html.Label("Select a cluster:"),
            dcc.Dropdown(
                id="km-cluster-filter",
                options=[{"label": "(all clusters)", "value": -1}] +
                        [{"label": _cluster_label(c), "value": c} for c in clusters],
                value=-1, clearable=False,
                style={"width": "340px"},
            ),
        ], className="control-row"),

        dcc.Graph(id="km-scatter"),
        dcc.Markdown(
            "*Each dot is one paper. Position = 2D TruncatedSVD projection; color = K-Means "
            "cluster assignment. Select a cluster from the dropdown to highlight it and see "
            "its profile below.*"
        ),

        html.Div(id="km-cluster-profile"),

        html.H4("Most-cited papers in the selected cluster"),
        html.Div(id="km-cluster-table"),
    ])


# ---------------------------------------------------------------------------
# Tab 3: Decision Tree
# ---------------------------------------------------------------------------
def dt_layout():
    if not ARTIFACTS_OK:
        return html.Div("Run scripts/build_artifacts.py to populate this tab.")
    dt = summary.get("decision_tree", {})
    return html.Div([
        dcc.Markdown(
            f"### Decision-tree citation predictor\n"
            f"Supervised binary classifier for **`quickly_cited_once` = cited_by_count \u2265 "
            f"{dt.get('cutoff_citations', 1)}**. Best "
            f"max_depth = **{dt.get('best_max_depth','?')}** chosen on validation F1.\n\n"
            f"Held-out test \u2014 accuracy {dt.get('accuracy',0):.2f} \u00b7 "
            f"precision {dt.get('precision',0):.2f} \u00b7 recall {dt.get('recall',0):.2f} \u00b7 "
            f"**F1 {dt.get('f1',0):.2f}**."
        ),

        dcc.Dropdown(
            id="dt-query",
            placeholder="Type a paper title to search…",
            searchable=True,
            clearable=True,
            style={"fontSize": "13px", "marginBottom": "12px"},
        ),

        html.Div(id="dt-output"),

        html.H4("Top 20 features by importance"),
        dcc.Graph(id="dt-feature-importance", figure=_dt_feat_bar()),
        dcc.Markdown(
            "*With the notebook-aligned `cited_by_count \u2265 1` target, this tree mostly leans "
            "on publication year, with author count showing up as a much smaller secondary "
            "signal. That happens because 2025 papers have had more time to collect at least "
            "one citation than 2026 papers, so the model is learning time-available-to-be-cited "
            "almost as much as paper content.*"
        ),
    ])


def _dt_feat_bar():
    if dt_feat.is_empty():
        return go.Figure()
    d = dt_feat.sort("importance").to_pandas()
    fig = px.bar(d, x="importance", y="feature", orientation="h",
                 title="Decision-tree feature importances (top 20)")
    fig.update_layout(height=520, margin={"l": 220})
    return fig


# ---------------------------------------------------------------------------
# Tab 4: Autoencoder similarity
# ---------------------------------------------------------------------------
def ae_layout():
    if not ARTIFACTS_OK:
        return html.Div("Run scripts/build_artifacts.py to populate this tab.")
    ae = summary.get("autoencoder", {})
    cfg = ae.get("best_config") or {}
    metric_bits = []
    if "top_5_same_subfield_normalized" in ae:
        metric_bits.append(f"top-5 same-subfield {ae['top_5_same_subfield_normalized']:.3f}")
    if "top_10_same_subfield_normalized" in ae:
        metric_bits.append(f"top-10 {ae['top_10_same_subfield_normalized']:.3f}")
    if "majority_baseline" in ae:
        metric_bits.append(f"majority baseline {ae['majority_baseline']:.3f}")
    metric_line = ""
    if metric_bits:
        metric_line = "\n\nHeld-out similarity quality — " + " · ".join(metric_bits) + "."
    return html.Div([
        dcc.Markdown(
            f"### Autoencoder similarity search\n"
            f"Neural model: SVD-150 \u279c MLP encoder \u279c **{cfg.get('latent_dim', '?')}-dim "
            f"latent** \u279c MLP decoder, trained with MSE reconstruction. Best hyperparameters "
            f"via randomized search: hidden_dim = {cfg.get('hidden_dim','?')}, lr = "
            f"{cfg.get('lr','?')}. Similarity = cosine distance in the latent space."
            f"{metric_line}"
        ),

        dcc.Dropdown(
            id="ae-query",
            placeholder="Type a paper title to search…",
            searchable=True,
            clearable=True,
            style={"fontSize": "13px", "marginBottom": "12px"},
        ),

        html.Div(id="ae-query-summary"),

        html.Div([
            html.Div([
                html.H4("Top 10 \u2014 Autoencoder (all papers)"),
                html.Div(id="ae-results"),
            ], className="col"),
            html.Div([
                html.H4("Top 10 \u2014 K-Means (same cluster only)"),
                html.Div(id="km-results"),
            ], className="col"),
        ], className="two-col"),

        dcc.Markdown("#### Latent space, 2D projection"),
        dcc.Graph(id="ae-scatter"),
        dcc.Markdown(
            "*Each dot is a paper's 32-dim embedding projected to 2D via SVD. Color = "
            "primary subfield. If the autoencoder learned something beyond subfield one-hots, "
            "you'll see within-subfield sub-clusters, bridges, and outliers.*"
        ),
    ])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, title="Paper Similarity Explorer")
server = app.server  # for gunicorn if needed

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0;
       background: #f7f8fa; color: #23272d; }
.header { padding: 16px 24px; background: #1c2533; color: white;
          display:flex; align-items:center; justify-content: space-between; }
.header h1 { margin: 0; font-size: 22px; font-weight: 600; }
.header .sub { font-size: 13px; opacity: 0.75; }
.tab-body { padding: 18px 28px; max-width: 1400px; margin: 0 auto; }
.stat-row { display: flex; gap: 14px; margin: 16px 0 24px; flex-wrap: wrap; }
.stat-card { background: white; padding: 14px 18px; border-radius: 10px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.06); min-width: 160px; }
.stat-value { font-size: 22px; font-weight: 600; color: #1c2533; }
.stat-label { font-size: 12px; color: #6a737c; margin-top: 3px;
              text-transform: uppercase; letter-spacing: 0.04em; }
.stat-sub   { font-size: 11px; color: #97a0a8; margin-top: 2px; }
.control-row { display: flex; gap: 12px; align-items: center; margin: 14px 0; }
.control-row label { font-weight: 500; }
.control-row input { padding: 8px 10px; border: 1px solid #ccd2d8; border-radius: 6px; }
.control-row button { padding: 8px 16px; background: #2d6cdf; color: white;
                      border: none; border-radius: 6px; cursor: pointer; }
.control-row button:hover { background: #2557b5; }
.two-col { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
           gap: 36px; margin-top: 14px; align-items: start; }
.col { min-width: 0; background: white; padding: 14px 16px; border-radius: 10px;
       box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.col h4 { margin: 0 0 12px; }
.banner { background: #fbe9e7; padding: 14px 18px; border-left: 4px solid #c0392b;
          border-radius: 6px; margin-bottom: 18px; font-family: monospace; font-size: 13px; }
"""

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>""" + CSS + """</style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
"""


def banner():
    if ARTIFACTS_OK:
        return html.Div()
    return html.Div(
        [html.Strong("Artifacts missing: "),
         ", ".join(missing_artifacts),
         html.Br(),
         "Run  ", html.Code("python scripts/build_artifacts.py"),
         "  from the project root to populate this dashboard."],
        className="banner",
    )


app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1("Research Paper Similarity Explorer"),
            html.Div("CIS 2450 Final Project \u00b7 K-Means + Decision Tree + Autoencoder",
                     className="sub"),
        ]),
        html.Div(f"{papers.height:,} papers" if ARTIFACTS_OK else "demo mode",
                 className="sub"),
    ], className="header"),

    html.Div([
        banner(),
        dcc.Tabs(id="tabs", value="tab-eda", children=[
            dcc.Tab(label="1 \u00b7 Overview & EDA",    value="tab-eda"),
            dcc.Tab(label="2 \u00b7 K-Means Clusters",   value="tab-km"),
            dcc.Tab(label="3 \u00b7 Citation Predictor", value="tab-dt"),
            dcc.Tab(label="4 \u00b7 Similarity Search",  value="tab-ae"),
        ]),
        html.Div(id="tab-content"),
    ], className="tab-body"),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _paper_search_options(search_val: str, max_results: int = 25) -> list:
    """Return dropdown options matching the typed query (case-insensitive substring)."""
    if not search_val or len(search_val) < 2 or not ARTIFACTS_OK:
        return []
    q = search_val.strip().lower()
    titles = papers["title"].fill_null("").to_numpy()
    opts = []
    for i, t in enumerate(titles):
        if q in t.lower():
            opts.append({"label": t, "value": i})
            if len(opts) >= max_results:
                break
    return opts


def _ensure_selected(opts: list, current_val) -> list:
    """Always include the currently selected paper in options so its label renders."""
    if current_val is None:
        return opts
    idx = int(current_val)
    if not any(o["value"] == idx for o in opts):
        title = papers["title"][idx] if ARTIFACTS_OK else f"Paper {idx}"
        opts = [{"label": title, "value": idx}] + opts
    return opts


@app.callback(
    Output("dt-query", "options"),
    Input("dt-query", "search_value"),
    State("dt-query", "value"),
    prevent_initial_call=True,
)
def _dt_search_options(search_val, current_val):
    return _ensure_selected(_paper_search_options(search_val), current_val)


@app.callback(
    Output("ae-query", "options"),
    Input("ae-query", "search_value"),
    State("ae-query", "value"),
    prevent_initial_call=True,
)
def _ae_search_options(search_val, current_val):
    return _ensure_selected(_paper_search_options(search_val), current_val)


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def _render_tab(tab):
    return {
        "tab-eda": eda_layout(),
        "tab-km":  kmeans_layout(),
        "tab-dt":  dt_layout(),
        "tab-ae":  ae_layout(),
    }[tab]


# --- K-Means tab ---
@app.callback(
    Output("km-scatter", "figure"),
    Output("km-cluster-profile", "children"),
    Output("km-cluster-table", "children"),
    Input("km-cluster-filter", "value"),
)
def _update_kmeans(cluster_sel):
    # Sample for performance — up to 6000 points
    rng = np.random.default_rng(0)
    n = papers.height
    idx = np.arange(n) if n <= 6000 else rng.choice(n, size=6000, replace=False)
    d = papers[idx.tolist()].to_pandas()
    d["cluster_str"] = d["kmeans_cluster"].astype(str)

    # Use AE 2D projection — much cleaner visual than raw SVD on sparse TF-IDF
    fig = px.scatter(
        d, x="km_2d_x", y="km_2d_y", color="cluster_str",
        hover_data={"title": True, "primary_subfield": True,
                    "publication_year": True,
                    "km_2d_x": False, "km_2d_y": False,
                    "cluster_str": False},
        labels={"cluster_str": "Cluster",
                "km_2d_x": "Component 1", "km_2d_y": "Component 2"},
        title="Papers colored by K-Means cluster (TruncatedSVD projection)",
    )
    fig.update_traces(marker=dict(size=4), opacity=0.5)
    if cluster_sel is not None and cluster_sel != -1:
        for tr in fig.data:
            tr.marker.opacity = 0.9 if tr.name == str(cluster_sel) else 0.08
    fig.update_layout(height=560, legend={"title": "Cluster"})

    # Cluster profile card
    if cluster_sel is not None and cluster_sel != -1:
        cl = papers.filter(pl.col("kmeans_cluster") == cluster_sel)
        total = cl.height
        top_sub = (cl["primary_subfield"].drop_nulls()
                   .value_counts().sort("count", descending=True).head(5))
        top_topic = (cl["primary_topic"].drop_nulls()
                     .value_counts().sort("count", descending=True).head(5))
        profile = html.Div([
            html.H4(f"Cluster {cluster_sel} profile — {total:,} papers"),
            html.Div([
                html.Div([
                    html.Div("Top subfields", style={"fontWeight": 600, "marginBottom": "6px"}),
                    html.Ul([
                        html.Li(f"{row[0]}  ({row[1]:,} papers, {100*row[1]/total:.0f}%)")
                        for row in top_sub.iter_rows()
                    ]),
                ], style={"flex": 1}),
                html.Div([
                    html.Div("Top topics", style={"fontWeight": 600, "marginBottom": "6px"}),
                    html.Ul([
                        html.Li(f"{row[0]}  ({row[1]:,} papers, {100*row[1]/total:.0f}%)")
                        for row in top_topic.iter_rows()
                    ]),
                ], style={"flex": 1}),
            ], style={"display": "flex", "gap": "32px"}),
        ], style={"background": "white", "padding": "16px 20px",
                  "borderRadius": "10px", "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                  "margin": "8px 0 16px"})
    else:
        profile = html.Div()

    # Most-cited table
    full = papers.to_pandas()
    sample = full if cluster_sel is None or cluster_sel == -1 else full[full["kmeans_cluster"] == cluster_sel]
    sample = (sample[["title", "primary_topic", "primary_subfield",
                       "publication_year", "cited_by_count", "kmeans_cluster"]]
              .sort_values("cited_by_count", ascending=False).head(15))
    tbl = dash_table.DataTable(
        data=sample.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in sample.columns],
        style_cell={"textAlign": "left", "padding": "6px",
                    "fontFamily": "inherit", "fontSize": "13px",
                    "whiteSpace": "normal", "height": "auto"},
        style_header={"backgroundColor": "#eef1f5", "fontWeight": "600"},
        page_size=15,
    )
    return fig, profile, tbl


# --- Decision Tree tab ---
@app.callback(
    Output("dt-output", "children"),
    Input("dt-query", "value"),
)
def _predict_dt(query):
    if query is None:
        return html.Div("Select a paper above to see the model's prediction.",
                        style={"color": "#97a0a8"})
    idx = int(query)
    if not (0 <= idx < papers.height):
        return html.Div("No matching paper.", style={"color": "#c0392b"})
    row = papers[idx]
    prob = float(row["dt_prob"][0])
    truth_col = "quickly_cited_once" if "quickly_cited_once" in papers.columns else "highly_cited"
    truth = int(row[truth_col][0])
    pred = int(prob >= 0.5)
    verdict = "CITED AT LEAST ONCE" if pred else "not cited yet"
    color = "#2d6cdf" if pred else "#6a737c"

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=prob, number={"valueformat": ".2f"},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": color},
               "threshold": {"line": {"color": "red", "width": 2},
                             "thickness": 0.75, "value": 0.5}},
        title={"text": "Predicted probability of being cited at least once"},
    ))
    gauge.update_layout(height=260, margin={"l": 20, "r": 20, "t": 50, "b": 20})

    return html.Div([
        html.Div([
            html.Div([
                html.Div(row["title"][0],
                         style={"fontWeight": 600, "fontSize": "16px"}),
                html.Div(f"{row['primary_subfield'][0]}   \u2022   "
                         f"{row['publication_year'][0]}   \u2022   "
                         f"{row['cited_by_count'][0]} citations",
                         style={"color": "#6a737c", "fontSize": "13px",
                                "margin": "4px 0 12px"}),
                html.Div([html.Strong("Model says: "),
                          html.Span(verdict, style={"color": color, "fontWeight": 600})]),
                html.Div([html.Strong("Ground truth: "),
                          "cited at least once" if truth else "not cited yet"]),
            ], style={"flex": 1}),
            html.Div(dcc.Graph(figure=gauge, config={"displayModeBar": False}),
                     style={"flex": 1}),
        ], style={"display": "flex", "gap": "20px", "alignItems": "center",
                  "padding": "14px", "background": "white",
                  "borderRadius": "10px", "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"}),
    ])


# --- Autoencoder similarity tab ---
def _tbl(df: pl.DataFrame):
    if df.is_empty():
        return html.Div("No results.")
    data = df.to_pandas().round({"similarity": 3})
    return dash_table.DataTable(
        data=data.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in data.columns],
        style_cell={"textAlign": "left", "padding": "6px",
                    "fontFamily": "inherit", "fontSize": "12px",
                    "overflow": "hidden", "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap"},
        style_cell_conditional=[
            {"if": {"column_id": "similarity"},       "width": "70px",  "textAlign": "center"},
            {"if": {"column_id": "title"},            "width": "45%",   "whiteSpace": "normal"},
            {"if": {"column_id": "primary_subfield"}, "width": "25%"},
            {"if": {"column_id": "publication_year"}, "width": "55px",  "textAlign": "center"},
            {"if": {"column_id": "cited_by_count"},   "width": "55px",  "textAlign": "center"},
        ],
        style_header={"backgroundColor": "#eef1f5", "fontWeight": "600", "fontSize": "12px"},
        style_table={"tableLayout": "fixed", "width": "100%", "overflowX": "auto"},
        page_size=10,
    )


@app.callback(
    Output("ae-query-summary", "children"),
    Output("ae-results", "children"),
    Output("km-results", "children"),
    Output("ae-scatter", "figure"),
    Input("ae-query", "value"),
)
def _search_similar(query):
    default_scatter = _ae_scatter()
    if query is None:
        return (html.Div("Select a paper above to see its neighbors.",
                         style={"color": "#97a0a8"}),
                html.Div(), html.Div(), default_scatter)
    idx = int(query)
    if not (0 <= idx < papers.height):
        return (html.Div("No matching paper.", style={"color": "#c0392b"}),
                html.Div(), html.Div(), default_scatter)

    row = papers[idx]
    summary_card = html.Div([
        html.Div("Query paper:",
                 style={"color": "#6a737c", "fontSize": "12px",
                        "textTransform": "uppercase"}),
        html.Div(row["title"][0],
                 style={"fontWeight": 600, "fontSize": "15px"}),
        html.Div(f"{row['primary_subfield'][0]}   \u2022   "
                 f"cluster {row['kmeans_cluster'][0]}   \u2022   "
                 f"{row['publication_year'][0]}",
                 style={"color": "#6a737c", "fontSize": "13px", "marginTop": "4px"}),
    ], style={"padding": "12px 16px", "background": "white",
              "borderRadius": "10px",
              "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
              "marginBottom": "12px"})

    ae_tbl = _tbl(ranked_neighbors(idx, 10))
    km_tbl = _tbl(kmeans_cluster_neighbors(idx, 10))
    scatter = _ae_scatter(highlight_idx=idx)
    return summary_card, ae_tbl, km_tbl, scatter


def _ae_scatter(highlight_idx: int | None = None):
    if not ARTIFACTS_OK:
        return go.Figure()
    n = papers.height
    rng = np.random.default_rng(0)
    idx = np.arange(n) if n <= 4000 else rng.choice(n, size=4000, replace=False)
    d = papers[idx.tolist()].to_pandas()
    fig = px.scatter(
        d, x="ae_2d_x", y="ae_2d_y", color="primary_subfield",
        hover_data={"title": True, "primary_subfield": True,
                    "publication_year": True,
                    "ae_2d_x": False, "ae_2d_y": False},
        labels={"primary_subfield": "Subfield",
                "ae_2d_x": "Component 1", "ae_2d_y": "Component 2"},
        title="Autoencoder latent space (SVD-2D projection)",
    )
    fig.update_traces(marker=dict(size=4), opacity=0.45)
    if highlight_idx is not None:
        row = papers[highlight_idx]
        fig.add_trace(go.Scatter(
            x=[row["ae_2d_x"][0]], y=[row["ae_2d_y"][0]],
            mode="markers",
            marker=dict(size=18, color="red", symbol="x",
                        line=dict(width=2, color="black")),
            name="Query paper", hovertext=row["title"][0],
        ))
    fig.update_layout(height=520, legend={"title": "Subfield"})
    return fig


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
