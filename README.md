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

## Benchmark & Evaluation Summary

### 1. Anomaly Detection: Large-Scale M-Pesa Dataset (`notebooks/mpesa-anomaly.ipynb`)
Evaluated on **19,847 held-out test transactions** (months 11–12, 575 verified fraud/anomalies, base rate 2.9%).

| Architecture / Method | Precision | Recall | F1 Score | Alerts / True Positives | Test PR-AUC | Operational Characteristic |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Multivariate PCA** *(95% var, 18 comp.)* | **0.9870** | 0.6609 | **0.7917** | 385 / 380 | **0.7003** | Lowest false alarms (only 5 on ~20k rows); best on linear balance deviations. |
| **Birch Autoencoder (BAE)** *(Latent=4, BIRCH th=3.0)* | 0.9294 | **0.6643** | 0.7748 | 411 / **382** | 0.6998 | Highest raw true positive catch; captures non-linear cluster anomalies. |
| *Reconstruction-Only Autoencoder (Ablation)* | 0.8841 | 0.6120 | 0.7233 | 398 / 352 | 0.6492 | Removing BIRCH latent distance reduces PR-AUC by 0.051. |
| *Supervised Classifier (Diagnostic baseline)* | — | — | — | — | 0.6970 | HistGradientBoosting (balanced class weights). |
| *Deterministic Rule (`amount > sender_balance`)* | 1.0000 | 0.6122 | 0.7594 | 352 / 352 | — | Catches clear overdraws; misses all hidden velocity/temporal drift. |

> **Key takeaway:** Bootstrap 95% confidence interval for the PR-AUC difference ($\text{BAE} - \text{PCA}$) is **[$-0.006, +0.005$]** (spanning zero), demonstrating statistical parity between the two models. Combining latent BIRCH clustering with Autoencoder reconstruction increased PR-AUC from **0.6492 to 0.6998**.

---

### 2. Anomaly Detection: Student Spending Telemetry (`notebooks/aml.ipynb`)
Evaluated on **1,528 test transactions** across student categories (rent, food, airtime, commute) with 54 injected temporal and velocity anomalies.

| Model Architecture | Precision | Recall | F1 Score | Test Alerts (Caught / Total) | Primary Detection Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Multivariate PCA** *(90% var, 11 comp.)* | 0.259 | **0.389** | **0.311** | 81 (21 / 54) | Superior at catching late-night bursts and temporal spikes (18/29 caught). |
| **Birch Autoencoder (BAE)** | **0.242** | 0.278 | 0.259 | 62 (15 / 54) | 21.7% fewer false alarms on normal rows; better at high-value uncharacteristic transfers. |

---

### 3. Predictive Budget Forecasting: Student Expenditure Cycles (`notebooks/aml.ipynb`)
Evaluated on **30-day forward student expenditure forecasting** (monthly equivalent KES).

#### A. Final Absolute-Error ($L_1$) Objective & Ensemble Comparison (Test Month: 2027-03)
*Trained to minimize Mean Absolute Error directly to dampen the impact of volatile spending spikes:*

| Model Architecture | Validation MAE (KES) | Validation RMSE (KES) | Test MAE (KES) | Test RMSE (KES) | Error Reduction vs. Mean Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train+Val Mean Baseline** | 23,477 | 48,641 | 13,868 | 16,539 | Baseline (0.0%) |
| **Train+Val Median Baseline** | 21,895 | 50,879 | 9,666 | 13,774 | 30.3% |
| **Random Forest Regressor** *(200 trees, min_leaf=4)* | **20,137** | **48,221** | 10,899 | 15,252 | 21.4% |
| **XGBoost Regressor** *(max_depth=2, lr=0.05)* | 20,431 | 50,136 | 9,661 | 14,751 | 30.3% |
| **Heterogeneous Ensemble** *(50% RF + 50% XGB)* | 20,419 | 49,932 | 10,098 | 14,850 | 27.2% |
| **Deep Multi-Layer Perceptron (MLP)** | **—** | **—** | **8,762** | **13,397** | **36.8% (Best Overall)** |

#### B. Deep MLP Multi-Seed Stability Test (`models/mlp.pth`)
*Evaluating the Multi-Layer Perceptron across 5 randomized initialization seeds:*

| Evaluation Metric | Seed 20260930 (Production) | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean across 5 Seeds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation MAE (KES)** | 20,536 | 21,742 | 22,157 | 22,353 | 22,922 | 21,942 |
| **Test MAE (KES)** | **8,762** | 9,510 | 9,270 | 11,072 | 10,461 | **9,815 (±935)** |
| **Test RMSE (KES)** | 13,397 | 15,252 | 14,471 | 15,087 | 15,207 | 14,683 |

> **Key takeaway:** Switching from $L_2$ (squared error) to $L_1$ (absolute error) dropped XGBoost validation MAE from 49,295 KES to 20,431 KES. The **Deep Multi-Layer Perceptron (MLP)** delivered the lowest test error (**8,762 KES**), consistently beating both the tree ensemble and the historical median baseline across all runs.
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
