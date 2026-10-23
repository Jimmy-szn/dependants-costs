# dependant-costs  —  U-CS 43

**Dependants Budget Management** · Wasaa Lifestyle

What supporting a child or relative costs over one, five and ten years, and which year is worst.

---

## Your contract

This is the shape the rest of the system expects. Agree any change with the teams below,
then tell Benjamin so the Team Integration Map gets reissued.

```
POST /projection
  in : { "dependantId": "..." }
  out: { "oneYear": 0, "fiveYear": 0, "tenYear": 0, "peakYear": 2036, "breakdown": {}, "assumptions": [] }
```

**Call it `dependant-costs` everywhere** — your repo name, your service URL, your route, and the
variable Benjamin sets when integrating you: `DEPENDANT_COSTS_SERVICE_URL`

## Who you depend on

U-CS 37 (budget-forecast) for the household cost model.

## Who depends on you

U-CS 46 (ops-metrics)

## Done when

Projections come with a stated range and assumptions, plus a note on handling children’s data.

---

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> — FastAPI gives you interactive API docs for free, so you can
test your endpoint without writing a client. Open `demo/index.html` for the mini app.

```bash
pytest -q                 # the contract test
docker build -t dependant-costs . && docker run -p 8000:8000 dependant-costs
```

## Progress
## Implemented API Endpoints

| Method | Endpoint | Description | Model Engine |
| :--- | :--- | :--- | :--- |
| `POST` | `/projection` | Dependant 1/5/10-year projection contract | Demographic Baseline |
| `POST` | `/predict-budget` | 30-day student expenditure forecast | **Deep Multi-Layer Perceptron (MLP)** |
| `POST` | `/detect-anomaly` | Evaluates single transaction for fraud/velocity spikes | **Birch Autoencoder (BAE)** |
| `POST` | `/sync-transaction` | Syncs mobile transactions & logs to `ai_results` | Automated BAE Screening |
| `POST` | `/create-budget` | Sets category budget caps in Supabase | Supabase Auth Protected |
| `GET`  | `/view-budgets` | Retrieves user budget caps | Supabase Auth Protected |
| `GET`  | `/fetch-monthly-transactions` | Fetches active month transactions | Supabase Auth Protected |
| `GET`  | `/health` | Cloud health check endpoint | Core System |

---

## Progress & Benchmark Summary

| Step | Status | Evidence & Metrics |
|------|:------:|-------------------|
| **1 — Baseline** | **Complete** | Historical Budget Baseline MAE: **16,510.73 KES**; PCA Anomaly F1: **0.0610** |
| **2 — Evaluation** |  **Complete** | Chronological 70/15/15 train/val/test splits on 120,000 M-Pesa transaction records; Precision-Recall threshold tuning |
| **3 — Model Training** |  **Complete** | **Deep MLP** beat all tree ensembles (MAE: **16,004.86 KES**); **Birch Autoencoder (BAE)** achieved **0.1548 F1** and **11.55% precision** (>3x gain over PCA) |
| **4 — Integration & Deploy** |  **Complete** | Containerized with Docker and deployed live on **Render** with PostgreSQL persistence on **Supabase** |

---

### What exists right now

- **`app/main.py`** — Production FastAPI application with endpoints for budgeting, BAE anomaly detection, and Supabase database logging.
- **`models/`** — Serialized production model weights:
  - `mlp.pth` & scalers (`mlp_x_scaler.joblib`, `mlp_y_scaler.joblib`) — Budget forecasting.
  - `bae.pth` & `bae_bundle.joblib` — Birch Autoencoder weights, 29 cluster centers, scalers, and decision threshold (0.8831).
- **`tests/`** — Dedicated verification scripts:
  - `review_mlp.py` — Verifies MLP weights and scalers loading.
  - `review_bae.py` — End-to-end verification of Birch Autoencoder anomaly scoring.
- **`notebooks/`**:
  - `notebooks/mpesa-anomaly.ipynb` — 120,000 M-Pesa transaction evaluation for PCA vs. BAE.
  - `notebooks/aml.ipynb` — Tree ensemble (Random Forest, XGBoost) and MLP expenditure experiments.
  - `notebooks/Finaccess.ipynb` & `notebooks/khbs2015_2016.ipynb` — National household expenditure baselines.
- **`Dockerfile` & `.dockerignore`** — Production deployment configuration configured for Render.

### What you hand in

- This repo, on GitHub, with Benjamin added as a collaborator
- A live URL that answers
- The filled-in documents (templates are in the `documents/` folder of the skeleton repo)
- A README a stranger could run this from

## Rules

- No secrets in git. No `.env`, no keys. Commit `.env.example` instead.
- Label synthetic data as synthetic — in the code and in the report.
- Staging data only. Never production, never live M-Pesa.
- Tell Benjamin the same day if you find a real bug in the platform.
