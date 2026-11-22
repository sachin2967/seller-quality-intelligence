# Seller study operations

## Data and artifacts

Use Python 3.11 in a project-local environment. Obtain the Olist CSVs under the
source's terms, as described in the [README](../README.md#data-provenance). Keep raw
files under ignored `data/raw/`; do not publish them or Kaggle credentials.

```powershell
python -m pip install -r requirements.txt
python scripts/train_and_evaluate.py
python scripts/evaluate_temporal.py
python -m pytest -q
```

Tests are offline and create their own temporary CSV/model fixtures. Running
`python -m pytest -q` needs no raw Olist files; the training and observational
evaluation commands above do require them. Fixtures are not model-quality evidence.

The first pipeline creates the full-history retrospective model and processed seller
table used by serving. The second produces a separate point-in-time observational
benchmark; it does not replace the served model. Keep model, explainer, survival
artifact and feature table from the same run together. Only load operator-trusted
joblib artifacts: deserialization can execute code.

## Serving

```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8020 --workers 1
python -m streamlit run streamlit_app/app.py --server.address 127.0.0.1 --server.port 8520
```

Run these long-running commands in separate terminals. Local mode permits loopback
development without an API key. Set `APP_ENV=production` and a random
`SELLER_API_KEY` of at least 32 characters for a private remote pilot. Clients send
`X-API-Key` on non-health API routes. Native execution requires explicit environment
variables; the ignored `.env` is read by Compose, not automatically by the Python app.

`/live` checks the process; `/ready` and `/health` check loaded artifacts. Missing or
partial artifacts, an empty seller table, unavailable explainers and invalid scores
produce 503. An unknown seller produces 404 only when the database is available.
Bodies are bounded to 64 KiB. Artifact changes require process restart.

The Streamlit dashboard reads local trusted artifacts directly; it is not protected
by the API's operator key. Keep **both** API and UI on loopback and add external TLS
and authenticated access controls before remote use. No multi-user authorization,
per-user quota or durable audit ledger is implemented.

## Interpretation

API responses explicitly mark retrospective and uncalibrated scope. The compatibility
field `risk_probability` is an uncalibrated score for a historical rule-derived label,
not a future event probability. Operations playbooks are heuristics. No causal
intervention benefit, churn prediction or realized financial savings is established.
The temporal benchmark omits future orders without observed deliveries and repeats
sellers across snapshots; document selection bias and uncertainty before new claims.

## Containers, recovery and rollback

`docker compose up --build -d` starts a one-shot training dependency before the API.
Raw CSVs are mounted read-only; processed data and model artifacts live in named
volumes. API/UI bind to loopback 8020/8520. Copy `.env.example` to ignored `.env`
before configuring the production operator key. `docker compose down` retains the
named volumes; `down -v` deletes them and must not be used as an ordinary restart.

Preserve known-good image IDs plus their complete artifact/feature-table set and
reports. While stopped, restore that set into a separate prepared volume/directory,
restart, check `/ready` and compare a known seller's score and explanation. Do not
mix a new feature table with an old model or silently relabel data during rollback.
Cloud load/recovery testing and authenticated UI access remain launch gates.
