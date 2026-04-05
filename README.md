# CIS 2450 Final Project — Research Paper Similarity Explorer

Grace Chi and Tina Ni

## Overview

An interactive dashboard that clusters research papers by similarity and lets users explore which papers/researchers are closest to each other. Uses K-Means clustering + PCA, built on data from OpenAlex and Semantic Scholar.

---

## Setup

### 1. Create the virtual environment

```bash
python3 -m venv .venv
```

### 2. Activate it

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

You need API keys set in your terminal before running the scraping scripts:

```bash
export S2_API_KEY=your_semantic_scholar_key
export OPENALEX_API_KEY=your_openalex_key   # optional, easier rate limits with it
```

---

## Running the Project

### Scrape data

```bash
python data/scrape_open_alex.py
python data/scrape_semantic_scholar.py
```

Data is written to `papers.db` (SQLite) in the project root. The scripts create the tables automatically.

### Run EDA

Open the notebook in VS Code and select the `.venv` kernel:

```
data/eda.ipynb
```

### Run the dashboard

```bash
python frontend/app.py
```

Then open your browser to `http://127.0.0.1:8050`

---

## Project Structure

```
.
├── data/
│   ├── scrape_open_alex.py       # scrapes papers from OpenAlex API
│   ├── scrape_semantic_scholar.py # scrapes tldr field from Semantic Scholar
│   └── eda.ipynb                 # exploratory data analysis
├── frontend/
│   └── app.py                    # Dash dashboard
├── papers.db                     # SQLite database (not committed to git)
├── requirements.txt
└── README.md
```
