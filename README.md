# UAE Tender Intelligence Platform

Modern data pipeline and Streamlit dashboard for aggregating UAE government tenders, enriching them with AI scoring, and enabling semantic search.

## Features

- **Automated scraping** of Abu Dhabi (API) and Dubai (Playwright) tender portals.
- **Cleaning & normalization** with translation support for Arabic text.
- **AI relevance scoring** via local LM Studio or OpenRouter-hosted LLMs with transparent caching.
- **Vector search** powered by SentenceTransformers embeddings and Chroma for semantic discovery.
- **Streamlit dashboard** offering filters, KPI cards, CSV export, and end-to-end pipeline execution.

## Repository layout

```
├── abu_dxb.py                 # Orchestrates scraping + merging via modular pipeline
├── cleaner.py                 # Cleans merged CSV, translates, and saves processed artefacts
├── pipeline/
│   ├── merge.py               # Utilities to combine and deduplicate dataframes
│   └── sources/
│       ├── abu_dhabi.py       # Abu Dhabi API scraper
│       └── dubai.py           # Dubai Playwright scraper
├── embeddings.py              # SentenceTransformer helpers
├── vectorstore.py             # Chroma vector database helpers
├── run_pipeline.py            # Programmatic pipeline runner (scrape → clean → embed)
├── ai_utils.py                # AI scoring, caching, provider abstractions
├── config.py                  # Central configuration + filesystem helpers
├── app.py                     # Streamlit UI
├── requirements.txt           # Python dependencies
└── README.md
```

## Prerequisites

1. **Python** 3.11+ recommended (virtualenv or venv suggested).
2. **System dependencies**
   - Playwright browser binaries (`playwright install` after installing requirements).
   - Google Chrome / Chromium (Playwright downloads automatically if missing).
   - LM Studio (optional) if running the local LLM path.
3. **Environment variables** (populate via `.env`):

   ```env
   # Local LM Studio defaults
   LMSTUDIO_URL=http://localhost:1234/v1/chat/completions
   LMSTUDIO_MODEL=google/gemma-3-1b

   # Optional remote provider (set AI_PROVIDER=openrouter to activate)
   AI_PROVIDER=lmstudio            # or openrouter
   OPENROUTER_API_KEY=sk-or-...
   OPENROUTER_MODEL=openrouter/auto
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions

   # Storage configuration
   DATA_DIR=/absolute/path/to/data
   RAW_DATA_DIR=/absolute/path/to/data/raw
   PROCESSED_DATA_DIR=/absolute/path/to/data/processed
   ARCHIVE_DATA_DIR=/absolute/path/to/data/archive
   VECTOR_DB_PATH=/absolute/path/to/data/vector_store
   AI_CACHE_PATH=/absolute/path/to/data/processed/ai_cache.json
   ```

   All paths default to the `data/` folder in the repository if unspecified.

## Setup

```bash
python -m venv tenderenv
source tenderenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
playwright install  # downloads browser drivers for Dubai scraper
```

If you plan to use the Streamlit dashboard with OpenRouter, ensure `OPENROUTER_API_KEY` is valid. For LM Studio, run the server at the `LMSTUDIO_URL` endpoint before launching the app.

## Usage

### 1. Run the end-to-end pipeline (CLI)

```bash
source tenderenv/bin/activate
python run_pipeline.py  # defaults to headless Playwright
```

This will:
1. Scrape Abu Dhabi and Dubai tenders.
2. Clean/normalize the merged dataset.
3. Persist timestamped CSVs to `data/processed/` and the latest artefacts.
4. Refresh the Chroma vector store with the cleaned records.

### 2. Streamlit dashboard

```bash
source tenderenv/bin/activate
streamlit run app.py
```

Key UI capabilities:
- Trigger the pipeline from the sidebar.
- Filter tenders by source, category, priority, publication window, and minimum AI score.
- View KPI metrics and export filtered results to CSV.
- Execute semantic search queries against the vector store.
- See AI summaries and priorities refreshed per keyword set.

### 3. Cleaner script (standalone)

If you already have a combined CSV (e.g., historical archive), you can run:

```bash
python cleaner.py data/processed/tenders_combined_latest.csv --print-path
```

## Troubleshooting

- **`python-dotenv` parse warnings** – ensure `.env` contains only `KEY=VALUE` lines (no shell commands).
- **Playwright timeouts** – rerun with `--no-headless` by using `python run_pipeline.py` from code or toggle in the Streamlit sidebar to observe UI actions.
- **Chroma `NotFoundError`** – handled automatically; if the vector store path changes, delete the directory to rebuild from scratch.
- **Tokenizers parallelism warning** – set `TOKENIZERS_PARALLELISM=false` if the Streamlit server is forked after initializing embeddings.

## Roadmap & contributions

- Automated test suite (pytest) and CI.
- Docker image for reproducible deployment.
- Additional data sources and analytics modules.

Pull requests welcome. Please open an issue for feature discussions or bug reports.
