# dependant-costs  —  U-CS 43

**Deep Learning and Statistical Models for Dependants Budget Recommendation and Anomaly Detection** · Wasaa Lifestyle


## Deployed on Render: <https://dependants-costs.onrender.com>



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

## Implemented API Endpoints

| Method | Endpoint | Description | Model Engine |
| :--- | :--- | :--- | :--- |
| `POST` | `/projection` | Dependant 1/5/10-year projection contract | Demographic Baseline |
| `POST` | `/predict-budget` | 30-day student expenditure forecast | **Deep Multi-Layer Perceptron (MLP)** |
| `POST` | `/sync-transaction` | Syncs mobile transactions & logs to `ai_results` | Automated PCA AND BAE Screening |
| `POST` | `/create-budget` | Sets category budget caps in Supabase | Supabase Auth Protected |
|`POST`|`/student/link-sponsor` | Links a student to a sponsor | Supabase Auth Protected |
| `POST`  | `/sponsor/pairing-codes` | returns 201 (Created) | Supabase Auth Protected |
| `PATCH` | `/student/transactions/{transaction_id}/budget-category/{id}` | Tags a transaction with a budget category | Supabase Auth Protected |
| `GET`  | `/fetch-user-role` | Fetches the user's role | Supabase Auth Protected |
| `GET`  | `/student/sponsor` | Fetches the student's sponsor information | Supabase Auth Protected |
| `GET`  | `/sponsor/dependants` | Fetches the sponsor's dependants | Supabase Auth Protected |
| `DELETE`  | `/sponsor-links/{link_id}` | Deletes a specific sponsor link by ID | Supabase Auth Protected |
| `GET`  | `/view-budgets` | Retrieves user budget caps | Supabase Auth Protected |
| `GET`  | `/fetch-monthly-transactions` | Fetches active month transactions | Supabase Auth Protected |
| `GET`  | `/health` | Cloud health check endpoint | Core System |
| `GET`  | `/sponsor/dependants/{student_id}/budget-health` | Checks the health of the student's budget | Core System |
| `GET`  | `/admin` | Admin dashboard | Core System |
| `POST`  | `/admin/login` | Admin login page | Core System |
| `POST`  | `/admin/logout` | Admin logout page | Core System |
| `GET`  | `/admin/overview` | Admin overview page | Core System |
For BAE scoring via `/sync-transaction`, outgoing transactions must include `amount_out`, `balance` (closing balance), `sender_balance_before`, the original `transaction_time`, and `transaction_fee` (use `0` when there is no fee). Incoming transactions are stored but not scored by this outgoing-only model. `transaction_time`, `sender_balance_before`, and `transaction_fee` are inference inputs and are not stored in `transactions`.

---

## Benchmark & Evaluation Summary
All forecasting and student-telemetry datasets are synthetic derived from a single-user M-Pesa pilot. The following tables summarize the evaluation results for anomaly detection and predictive budget forecasting.
Table 1 : It uses the 120,000-record M-Pesa Corpus from Kaggle and is not the single-user pilot data used for the student-telemetry datasets, which can be downloaded from: https://www.kaggle.com/datasets/calebboen/mpesa-transactions-fraud/code
### 1. Anomaly Detection: Large-Scale M-Pesa Dataset (`notebooks/mpesa-anomaly.ipynb`)
Evaluated on **19,847 held-out test transactions** (months 11–12, 575 verified fraud/anomalies, base rate 2.9%).

| Architecture / Method | Precision | Recall | F1 Score | Alerts / True Positives | Test PR-AUC | Operational Characteristic |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Multivariate PCA** *(90% var, 6 comp.)* | **0.9845** | 0.6626 | 0.7921 | 387 / 381 | 0.7019 | Lowest false alarms (6 on ~20k rows); best on linear balance deviations. |
| **Birch Autoencoder (BAE)** *(Notebook/Colab run; Latent=4, BIRCH th=3.0)* | **0.9698** | **0.6713** | **0.7934** | **398** / **386** | **0.7035** | Captures non-linear cluster anomalies. |
| *Supervised Classifier (Diagnostic baseline)* | — | — | — | — | 0.6970 | HistGradientBoosting (balanced class weights). |
| *Deterministic Rule (`amount > sender_balance`)* | 1.0000 | 0.6122 | 0.7594 | 352 / 352 | — | Catches clear overdraws; misses all hidden velocity/temporal drift. |

> **Key takeaway:** Bootstrap 95% confidence interval for the PR-AUC difference ($\text{BAE} - \text{PCA}$) is **[$-0.004, +0.008$]** (spanning zero), demonstrating no statistically clear PR-AUC difference between the two models.

---

### 2. Anomaly Detection: Student Spending Telemetry (`notebooks/aml.ipynb`)
Evaluated on **1,528 test transactions** across student categories (rent, food, airtime, commute) with 54 injected temporal and velocity anomalies.

| Model Architecture | Precision | Recall | F1 Score | Test Alerts (Caught / Total) |Test PR-AUC | Primary Detection Strength |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Multivariate PCA** *(90% var, 12 comp.)* | 0.259 | **0.389** | **0.311** | 81 (21 / 54) | ~ 0.19 | Superior at catching late-night bursts and temporal spikes (18/29 caught). |
| **Birch Autoencoder (BAE)** | **0.242** | 0.278 | 0.259 | 62 (15 / 54) | ~ 0.19 | 21.7% fewer false alarms on normal rows; better at high-value uncharacteristic transfers. |

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

> **Key takeaway:** Switching from $L_2$ (squared error) to $L_1$ (absolute error) dropped XGBoost validation MAE from 49,295 KES to 20,431 KES. The **Deep Multi-Layer Perceptron (MLP)** delivered the lowest test error (**8,762 KES**), 3 out of 5 seeds beat the median baseline , and the mean across 5 seeds was **9,815 KES ± 935 KES**.
### What exists right now

- **`app/main.py`** — Production FastAPI application with endpoints for budgeting, BAE anomaly detection, and Supabase database logging.
- **`models/`** — Serialized production model weights:
  - `mlp.pth` & scalers (`mlp_x_scaler.joblib`, `mlp_y_scaler.joblib`) — Budget forecasting.
  - `bae.pth` & `bae_bundle.joblib` — Birch Autoencoder weights, 29 cluster centers, scalers, and decision threshold (0.9698).
  - `pca.joblib` — PCA weights and scalers for anomaly detection.
  - **`data/`** — Anonymised pilot dataset and M-Pesa Kaggle corpus.
- **`tests/`** — Dedicated verification scripts:
  - `review_mlp.py` — Verifies MLP weights and scalers loading.
  - `review_bae.py` — End-to-end verification of Birch Autoencoder anomaly scoring.
- **`notebooks/`**:
  - `notebooks/mpesa-anomaly.ipynb` — 120,000 M-Pesa transaction evaluation for PCA vs. BAE.
  - `notebooks/aml.ipynb` — Tree ensemble (Random Forest, XGBoost) and MLP expenditure experiments.
  - `notebooks/Finaccess.ipynb` & `notebooks/khbs2015_2016.ipynb` — National household expenditure baselines.
- **`Dockerfile` & `.dockerignore`** — Production deployment configuration configured for Render.

## For a Stranger to Run
Environment variables are required for Supabase authentication and database access. Create a `.env` file in the root directory with the following keys:

```bash
SUPABASE_URL=<your-supabase-url>
SUPABASE_ANON_KEY=<your-supabase-key>
SUPABASE_SERVICE_ROLE_KEY=<your-supabase-service-role-key>
ADMIN_USER_IDS=<comma-separated-admin-user-ids>
```