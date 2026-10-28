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
### Table 1 — Anomaly detection, Kaggle M-Pesa fraud data (test: months 11–12, 575 fraud cases in 19,847 rows)

| Method | Precision | Recall | F1 | Alerts / caught | Test PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Overdraw rule (amount > sender balance) | 1.000 | 0.612 | 0.759 | 352 / 352 | n/a |
| PCA (95% variance, 18 components) | 0.987 | 0.661 | 0.792 | 385 / 380 | 0.7003 |
| BAE (BIRCH autoencoder) | 0.929 | 0.664 | 0.775 | 411 / 382 | 0.6998 |
| Supervised gradient boosting (diagnostic only) | n/a | n/a | n/a | n/a | 0.697 |

### Table 2 — Effect of preprocessing on the Kaggle data (F1)

| Model | Raw features, 70/15/15 split | Log-transformed features, month-based split |
|---|:---:|:---:|
| PCA | 0.0610 | 0.7917 |
| BAE | 0.1548 | 0.7748 |

### Table 3 — Anomaly detection, synthetic transactions (test: 94 injected anomalies in 1,876 rows)

| Model | Precision | Recall | F1 | Alerts / caught | Test PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| PCA (90% variance, 11 components) | 0.382 | 0.447 | 0.412 | 110 / 42 | 0.393 |
| BAE | 0.467 | 0.372 | 0.414 | 75 / 35 | 0.432 |

Bootstrap 95% interval for the PR-AUC difference (BAE minus PCA): −0.051 to 0.129 (includes zero, so the models are comparable).

### Table 4 — Budget forecasting, synthetic data (monthly-equivalent KES)

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
|---|:---:|:---:|:---:|:---:|
| Training-mean baseline | 32,476 | 49,612 | 31,777 | 33,699 |
| Random Forest | 41,378 | 54,467 | 30,915* | 38,155* |
| XGBoost | 49,295 | 64,635 | n/a | n/a |
| RF + XGBoost ensemble | 45,259 | 58,125 | n/a | n/a |
| MLP (seed 20260930) | 31,364 | 50,788 | 23,322 | 26,690 |
| MLP (mean of 5 seeds) | 30,780 | n/a | 22,634 (SD 2,832) | 26,388 |

\* Refit on train plus validation and compared with a baseline built from the same rows (test MAE 30,617, RMSE 32,573).
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
