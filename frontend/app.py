import sqlite3
from pathlib import Path

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import polars as pl

# --- Data loading ---

DB_PATH = Path(__file__).parent.parent / "papers.db"

def load_papers():
    conn = sqlite3.connect(DB_PATH)
    df = pl.read_database("SELECT * FROM openalex_papers", conn)
    conn.close()
    return df

df = load_papers()

# --- App setup ---

app = dash.Dash(__name__)
app.title = "Research Paper Similarity Explorer"

app.layout = html.Div([
    html.H1("Research Paper Similarity Explorer", style={"textAlign": "center"}),

    # Search box for filtering by topic
    html.Div([
        html.Label("Filter by subfield:"),
        dcc.Dropdown(
            id="subfield-dropdown",
            options=[{"label": s, "value": s} for s in sorted(df["primary_subfield"].drop_nulls().unique().to_list())],
            placeholder="Select a subfield...",
            clearable=True,
        ),
    ], style={"width": "50%", "margin": "20px auto"}),

    # Citation distribution chart
    html.Div([
        dcc.Graph(id="citation-chart"),
    ]),

    # Papers table preview
    html.Div([
        html.H3("Sample Papers", style={"textAlign": "center"}),
        html.Div(id="papers-table"),
    ]),
])


# --- Callbacks ---

@app.callback(
    Output("citation-chart", "figure"),
    Input("subfield-dropdown", "value"),
)
def update_chart(subfield):
    # Filter by subfield if one is selected
    filtered = df.filter(pl.col("primary_subfield") == subfield) if subfield else df

    fig = px.histogram(
        filtered.to_pandas(),
        x="cited_by_count",
        nbins=40,
        title="Citation Count Distribution",
        labels={"cited_by_count": "Citations"},
        color_discrete_sequence=["steelblue"],
    )
    fig.update_layout(bargap=0.1)
    return fig


@app.callback(
    Output("papers-table", "children"),
    Input("subfield-dropdown", "value"),
)
def update_table(subfield):
    filtered = df.filter(pl.col("primary_subfield") == subfield) if subfield else df
    sample = filtered.head(20).select(["title", "primary_subfield", "cited_by_count", "author_count"])

    # Build a simple HTML table
    rows = [
        html.Tr([html.Th(col) for col in sample.columns])
    ] + [
        html.Tr([html.Td(str(val)) for val in row])
        for row in sample.iter_rows()
    ]

    return html.Table(rows, style={
        "width": "90%",
        "margin": "0 auto",
        "borderCollapse": "collapse",
        "fontSize": "13px",
    })


if __name__ == "__main__":
    app.run(debug=True)
