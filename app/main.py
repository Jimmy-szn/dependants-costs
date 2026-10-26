from datetime import datetime, timedelta, timezone
import secrets
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from supabase import create_client, Client
import os
import numpy as np
import uuid
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from typing import List
from zoneinfo import ZoneInfo
import os
import torch
import torch.nn as nn
import joblib
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
# Import schema boundaries securely
from app.schemas import Request, Response
from app.service import handle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
supabase_auth: Client = create_client(SUPABASE_URL, os.getenv("SUPABASE_ANON_KEY"))
supabase_db : Client = create_client(SUPABASE_URL, os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


app = FastAPI(title="U-CS 43 — dependant-costs", version="0.1.0")
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BirchAutoencoder(nn.Module):
    def __init__(self, in_d, hidden_d=16, latent_d=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_d, hidden_d),
            nn.BatchNorm1d(hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, latent_d)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_d, hidden_d),
            nn.BatchNorm1d(hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, in_d)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

BAE_MODEL_PATH = os.path.join(BASE_DIR, "../models/bae.pth")
BAE_BUNDLE_PATH = os.path.join(BASE_DIR, "../models/bae_bundle.joblib")

bae_model = None
bae_bundle = None

if os.path.exists(BAE_BUNDLE_PATH) and os.path.exists(BAE_MODEL_PATH):
    bae_bundle = joblib.load(BAE_BUNDLE_PATH)
    in_features = len(bae_bundle["feature_columns"])
    
    bae_model = BirchAutoencoder(in_d=in_features, hidden_d=16, latent_d=4)
    bae_model.load_state_dict(torch.load(BAE_MODEL_PATH, map_location=torch.device('cpu')))
    bae_model.eval()
    print(f" BAE (Birch Autoencoder) loaded successfully. (Threshold: {bae_bundle['threshold']:.4f})")
else:
    print(" BAE model or bundle files not found in /models.")
class BudgetMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, values):
        return self.network(values)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/mlp.pth")
X_SCALER_PATH = os.path.join(BASE_DIR, "../models/mlp_x_scaler.joblib")
Y_SCALER_PATH = os.path.join(BASE_DIR, "../models/mlp_y_scaler.joblib")
model = BudgetMLP(input_size=10) 

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print(f" Weights loaded successfully. Last sync timestamp: {os.path.getmtime(MODEL_PATH)}")
x_scaler = None
y_scaler = None
if os.path.exists(X_SCALER_PATH) and os.path.exists(Y_SCALER_PATH):
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
else:
    print("Scaler files not detected yet. Check your Desktop Sync status.")

class BudgetRequest(BaseModel):
    category: str
    limit_amount: float = Field(gt=0)
class SyncRequest(BaseModel):
    transactionData: List[dict]
class ForecastRequest(BaseModel):
    student_id: str
class AnomalyDetectionRequest(BaseModel):
    transaction_id: str = None 
    student_id: str
    amount: float
    sender_balance_before: float
    sender_balance_after: float
    receiver_balance_before: float = 0.0
    receiver_balance_after: float = 0.0
    hour: int = 12
    transaction_type: str = "peer"    # peer, till, paybill
    device_type: str = "smartphone"   # smartphone, feature
    region: str = "Nairobi"           # Nairobi, Eldoret, Kisumu, Nakuru, etc.
    day_of_week: str = "Mon"
async def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    
    try:
        user_response = supabase_auth.auth.get_user(token)
        if not user_response or not user_response.user or not user_response.user.id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session has expired or token is invalid. Please log in again."
            )
        return user_response.user.id  
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
  
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    """Hosting platforms call this to check the service is alive."""
    return {"status": "ok"}
NAIROBI = ZoneInfo("Africa/Nairobi")
WARN_AT = 0.8  # warning threshold for budget usage
@app.get("/fetch-user-role")
def fetch_user_role(user_id: str = Depends(get_current_user_id)):
    student = (
        supabase_db.table("students")
        .select("student_id")
        .eq("student_id", user_id)
        .limit(1)
        .execute()
    )
    if student.data:
        return {"status": "ok", "role": "student"}

    sponsor = (
        supabase_db.table("sponsors")
        .select("sponsor_id")
        .eq("sponsor_id", user_id)
        .limit(1)
        .execute()
    )
    if sponsor.data:
        return {"status": "ok", "role": "sponsor"}

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User role not found",
    )
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I
CODE_LENGTH = 8
CODE_TTL_HOURS = 24
DEFAULT_PERMISSION = "view_budget_health"

def now_utc() -> datetime:
    return datetime.now(timezone.utc)
def month_start_utc() -> datetime:
    local =datetime.now(NAIROBI)
    return local.replace(day=1, hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)

def compute_budget_health(student_id: str) -> dict:
    since = month_start_utc()
    transactions = (
        supabase_db.table("transactions")
        .select("amount_in, amount_out, category, balance, created_at, ""ai_results(risk_flag)")
        .eq("student_id", student_id)
        .gte("created_at", since.isoformat())
        .order("created_at", desc=True)
        .limit(5000)
        .execute().data
    ) or []
    caps =(
        supabase_db.table("budget_caps")
        .select("category, limit_amount")
        .eq("student_id", student_id)
        .execute().data
    ) or []
    spend_by_category, total_in, total_out, flagged, latest_balance = {}, 0.0, 0.0, 0, None
    for t in transactions:  # newest first
        out = float(t["amount_out"] or 0)
        total_in += float(t["amount_in"] or 0)
        total_out += out
        cat = (t["category"] or "uncategorized").strip().lower()
        spend_by_category[cat] = spend_by_category.get(cat, 0.0) + out
        if latest_balance is None and t["balance"] is not None:
            latest_balance = float(t["balance"])
        if any(r.get("risk_flag") for r in (t.get("ai_results") or [])):
            flagged += 1

    categories, over, warn = [], 0, 0
    for c in caps:
        limit_amount = float(c["limit_amount"] or 0)
        spent = spend_by_category.get((c["category"] or "").strip().lower(), 0.0)
        ratio = spent / limit_amount if limit_amount > 0 else 0.0
        state = "over" if ratio > 1 else "warning" if ratio >= WARN_AT else "ok"
        over += state == "over"
        warn += state == "warning"
        categories.append({
            "category": c["category"],
            "limit": limit_amount,
            "spent": round(spent, 2),
            "percent_used": round(ratio * 100, 1),
            "state": state,
        })

    return {
        "period_start": since,
        "overall": "over" if over else "warning" if (warn or flagged) else "ok",
        "total_in": round(total_in, 2),
        "total_out": round(total_out, 2),
        "latest_balance": latest_balance,
        "flagged_transactions": flagged,
        "categories": categories,
    }


def require_sponsor(user_id: str = Depends(get_current_user_id)) -> str:
    res = (
        supabase_db.table("sponsors")
        .select("sponsor_id")
        .eq("sponsor_id", user_id)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Sponsor account required")
    return user_id
def require_student(user_id: str = Depends(get_current_user_id)) -> str:
    res = (
        supabase_db.table("students")
        .select("student_id")
        .eq("student_id", user_id)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Student account required")
    return user_id
def generate_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
@app.post("/sponsor/pairing-codes", status_code=status.HTTP_201_CREATED)

def create_pairing_code(sponsor_id: str = Depends(require_sponsor)):
    expires_at = now_utc() + timedelta(hours=CODE_TTL_HOURS)

    for _ in range(5):  # retry on the rare code collision
        code = generate_code()
        try:
            res = (
                supabase_db.table("pairing_codes")
                .insert({
                    "code": code,
                    "sponsor_id": sponsor_id,
                    "expires_at": expires_at.isoformat(),
                    "used": False,
                })
                .execute()
            )
            if res.data:
                return {"status": "ok", "code": code, "expires_at": expires_at.isoformat()}
        except Exception:
            continue

    raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not generate a code")
class LinkSponsorBody(BaseModel):
    code: str

@app.post("/student/link-sponsor", status_code=status.HTTP_201_CREATED)
def link_to_sponsor(body: LinkSponsorBody, student_id: str = Depends(require_student)):
    code = body.code.strip().upper()
    if not code:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Code is required")

    # Claim the code atomically: only succeeds if it is unused and not expired.
    # This stops two students from redeeming the same code at the same time.
    claim = (
        supabase_db.table("pairing_codes")
        .update({"used": True})
        .eq("code", code)
        .eq("used", False)
        .gt("expires_at", now_utc().isoformat())
        .execute()
    )
    if not claim.data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired code")

    sponsor_id = claim.data[0]["sponsor_id"]

    # Don't create a duplicate link
    existing = (
        supabase_db.table("sponsor_links")
        .select("link_id")
        .eq("sponsor_id", sponsor_id)
        .eq("student_id", student_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        supabase_db.table("pairing_codes").update({"used": False}).eq("code", code).execute()
        raise HTTPException(status.HTTP_409_CONFLICT, "Already linked to this sponsor")

    try:
        link = (
            supabase_db.table("sponsor_links")
            .insert({
                "link_id": str(uuid.uuid4()),
                "sponsor_id": sponsor_id,
                "student_id": student_id,
                "permission_level": DEFAULT_PERMISSION,
            })
            .execute()
        )
    except Exception:
        supabase_db.table("pairing_codes").update({"used": False}).eq("code", code).execute()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not create link")

    return {"status": "ok", "link": link.data[0]}


@app.get("/student/sponsor")
def get_my_sponsor(student_id: str = Depends(require_student)):
    links = (
        supabase_db.table("sponsor_links")
        .select("link_id, sponsor_id, permission_level, created_at")
        .eq("student_id", student_id)
        .execute()
    )
    if not links.data:
        return {"status": "ok", "linked": False, "sponsors": []}

    sponsor_ids = [l["sponsor_id"] for l in links.data]
    sponsors = (
        supabase_db.table("sponsors").select("sponsor_id, fullname, email").in_("sponsor_id", sponsor_ids).execute()
    )
    by_id = {s["sponsor_id"]: s for s in sponsors.data}

    result = [{**l, "sponsor": by_id.get(l["sponsor_id"])} for l in links.data]
    return {"status": "ok", "linked": True, "sponsors": result}


@app.get("/sponsor/dependants")
def list_dependants(sponsor_id: str = Depends(require_sponsor)):
    links = (
        supabase_db.table("sponsor_links")
        .select("link_id, student_id, permission_level, created_at")
        .eq("sponsor_id", sponsor_id)
        .execute()
    )

    dependants = []
    if links.data:
        student_ids = [l["student_id"] for l in links.data]
        students = (
            supabase_db.table("students").select("student_id, fullname, email").in_("student_id", student_ids).execute()
        )
        by_id = {s["student_id"]: s for s in students.data}
        dependants = [{**l, "student": by_id.get(l["student_id"])} for l in links.data]

    pending = (
        supabase_db.table("pairing_codes")
        .select("code", count="exact")
        .eq("sponsor_id", sponsor_id)
        .eq("used", False)
        .gt("expires_at", now_utc().isoformat())
        .execute()
    )

    return {
        "status": "ok",
        "linked_count": len(dependants),
        "pending_invitations": pending.count or 0,
        "dependants": dependants,
    }



@app.delete("/sponsor-links/{link_id}")
def unlink(link_id: str, user_id: str = Depends(get_current_user_id)):
    link = (
        supabase_db.table("sponsor_links")
        .select("link_id, sponsor_id, student_id")
        .eq("link_id", link_id)
        .limit(1)
        .execute()
    )
    if not link.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Link not found")

    row = link.data[0]
    if user_id not in (row["sponsor_id"], row["student_id"]):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your link")

    supabase_db.table("sponsor_links").delete().eq("link_id", link_id).execute()
    return {"status": "ok"}

@app.get("/view-budgets")
async def budget(student_id: str = Depends(get_current_user_id)):
    response = supabase_db.table("budget_caps").select("*").eq("student_id", student_id).execute()
    return {"status": "ok", "data": response.data}
@app.get('/fetch-monthly-transactions')
async def fetch_monthly_transactions(student_id: str = Depends(get_current_user_id)):
    try:
        start_of_month = month_start_utc().isoformat()
        response = supabase_db.table("transactions").select("*").eq("student_id", student_id).gte("created_at", start_of_month).order("created_at", desc=True).execute()
        transactions_response = response.data if response.data else []
        return {"status": "success", "transactions": transactions_response}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching transactions: {str(e)}",
        )
@app.post('/create-budget')
async def create_budget(
    item: BudgetRequest, student_id: str = Depends(get_current_user_id)):
    sanitized_item = {
        "budget_id": str(uuid.uuid4()),
        "student_id": str(student_id),
        "category": item.category,
        "limit_amount": item.limit_amount
    }
    
    try:
        print(f"Verified Student ID Context: {student_id}")
        print(f"Outbound Normalized Payload: {sanitized_item}")

        response = supabase_db.table("budget_caps").insert(sanitized_item).execute()
        return {"status": "ok", "data": response.data}
    except Exception as e:
        print(f"Database Exception Encountered: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,           
            detail=str(e),
        )

@app.post("/projection", response_model=Response)
def endpoint(body: Request) -> Response:
    return handle(body)

@app.post("/sync-transaction")
async def sync_transaction(request: SyncRequest, student_id: str = Depends(require_student)):
    if not request.transactionData:
        return {"status": "ok", "message": "synced_count: 0", "anomalies_flagged": 0, "evaluations": [], "data": []}
    for transaction in request.transactionData:
        transaction["student_id"] = student_id
    try:
        response = supabase_db.table("transactions").upsert(request.transactionData, on_conflict="transaction_id", ignore_duplicates=True).execute()
        inserted_transactions = response.data if response.data else []
        evaluation_results = []
        for tx in inserted_transactions:
            t_id = tx.get("transaction_id") or tx.get("trans_id") or tx.get("id")
            ai_eval = score_transaction_with_bae(tx, trans_id=t_id)
            if ai_eval:
                evaluation_results.append(ai_eval)

        high_risk_count = sum(1 for res in evaluation_results if res.get("risk_flag"))

        return {
            "status": "ok",
            "synced_count": len(inserted_transactions),
            "anomalies_flagged": high_risk_count,
            "evaluations": evaluation_results,
            "data": inserted_transactions
        }
    except Exception as e:
        print(f"Database/Sync Exception Encountered: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,           
            detail=str(e),
        )


def assert_can_view_student(user_id: str, student_id: str):
    if user_id == student_id:
        return
    link = (
        supabase_db.table("sponsor_links")
        .select("link_id")
        .eq("sponsor_id", user_id)
        .eq("student_id", student_id)
        .limit(1)
        .execute()
    )
    if not link.data:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not allowed to view this student")


@app.post("/predict-budget")
async def predict_budget(request: ForecastRequest,
                         user_id: str = Depends(get_current_user_id)):
    assert_can_view_student(user_id, request.student_id)  
   
    try:
        
        start_of_month = month_start_utc().isoformat()
        response = supabase_db.table("transactions").select("category, amount_out").eq("student_id", request.student_id).gte("created_at", start_of_month).execute()
        transactions_response = response.data if response.data else []
        caregory_sums = {
            "agent_withdrawal":0.0,
            "airtime": 0.0,
            "buy_goods": 0.0,
            "data_bundle": 0.0,
            "mshwari_deposit": 0.0,
            "pochi": 0.0,
            "paybill": 0.0,
            "send_money": 0.0,
            "transaction_fee": 0.0
        }
        gross_outflow = 0.0
        for transaction in transactions_response:
            category = str(transaction.get("category", "uncategorized")).strip().lower()
            raw_amount = transaction.get("amount_out")
            if raw_amount is None:
                amount_out = 0.0
            else:
                try:
                    amount_out = float(raw_amount)
                except ValueError:
                    amount_out = 0.0

            if category in caregory_sums:
                caregory_sums[category] += amount_out
                gross_outflow += amount_out
        generated_features = [
            caregory_sums["agent_withdrawal"], 
            caregory_sums["airtime"],
            caregory_sums["buy_goods"],
            caregory_sums["data_bundle"],
            caregory_sums["mshwari_deposit"],
            caregory_sums["pochi"],
            caregory_sums["paybill"],
            caregory_sums["send_money"],
            caregory_sums["transaction_fee"],
            gross_outflow
        ]
        raw_features = np.array([generated_features], dtype=np.float32)
        if x_scaler is not None:
            scaled_features = x_scaler.transform(raw_features)
        else:
            scaled_features = raw_features
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor).item()
        if y_scaler is not None:
            forecasted_value = y_scaler.inverse_transform([[prediction]])[0][0]
        else:
            forecasted_value = prediction
        return {"status": "success", "forecasted_amount": round(forecasted_value, 2), "transactions_analyzed":len(transactions_response), "category_sums": caregory_sums, "gross_outflow": gross_outflow}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        ) 
    
def score_transaction_with_bae(tx_data: dict, trans_id: str = None):
    """Evaluates a single transaction dictionary with BAE and saves to ai_results."""
    if bae_model is None or bae_bundle is None:
        return None

    try:
        raw_amount = abs(float(tx_data.get("amount_out") or tx_data.get("amount") or 0.0))
        time_transaction = supabase_db.table("transactions").select("created_at").eq("transaction_id", trans_id).execute().data[0]["created_at"]
        hour = datetime.fromisoformat(time_transaction).hour
        day_of_week = datetime.fromisoformat(time_transaction).strftime("%a")
        balance = float(tx_data.get("balance") or 0.0)
        input_df = pd.DataFrame([{
            'amount': raw_amount,
            'sender_balance_before': balance + raw_amount,
            'sender_balance_after': balance,
            'receiver_balance_before': 0.0,
            'receiver_balance_after': 0.0,
            'hour': hour,
            'transaction_type': tx_data.get("category", "peer"),
            'device_type': 'smartphone',
            'region': 'Nairobi',
            'day_of_week': day_of_week
        }])

        cat_cols = ['transaction_type', 'device_type', 'region', 'day_of_week']
        encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True, dtype=float)
        aligned = encoded.reindex(columns=bae_bundle["feature_columns"], fill_value=0.0)

        scaled_x = bae_bundle["feature_scaler"].transform(aligned).astype(np.float32)

        with torch.no_grad():
            tensor_x = torch.tensor(scaled_x, dtype=torch.float32)
            latent = bae_model.encoder(tensor_x).numpy()
            reconstructed = bae_model(tensor_x).numpy()

        recon_error = float(np.mean((scaled_x - reconstructed) ** 2, axis=1)[0])

        latent_scaled = bae_bundle["latent_scaler"].transform(latent)
        _, cluster_dist = pairwise_distances_argmin_min(latent_scaled, bae_bundle["birch_centers"])
        cluster_dist_val = float(cluster_dist[0])

        stats = bae_bundle["stats"]
        recon_score = (recon_error - stats["recon_median"]) / stats["recon_iqr"]
        cluster_score = (cluster_dist_val - stats["cluster_median"]) / stats["cluster_iqr"]
        anomaly_score = float((recon_score + cluster_score) / 2)

        threshold = float(bae_bundle["threshold"])
        is_anomaly = anomaly_score >= threshold

        # Insert to public.ai_results table
        result_record = {
            "results_id": str(uuid.uuid4()),
            "transaction_id": trans_id,
            "track_a_score": None,
            "track_a_forecast": None,
            "track_b_reconstruct_err": round(recon_error, 4),
            "track_b_forecast": None,
            "risk_flag": is_anomaly
        }
        supabase_db.table("ai_results").insert(result_record).execute()

        return {
            "results_id": result_record["results_id"],
            "anomaly_score": round(anomaly_score, 4),
            "risk_flag": is_anomaly
        }
    except Exception as e:
        print(f"BAE evaluation error for transaction: {str(e)}")
        return None

@app.get("/sponsor/dependants/{student_id}/budget-health")
def dependant_budget_health(student_id: str,
                            sponsor_id: str = Depends(require_sponsor)):
    assert_can_view_student(sponsor_id, student_id)
    return {"status": "ok", "student_id": student_id, **compute_budget_health(student_id)}

class TagBody(BaseModel):
    budget_category: str | None = None

@app.patch("/student/transactions/{transaction_id}/budget-category")
def tag_transaction(transaction_id: str, body: TagBody,
                    student_id: str = Depends(require_student)):
    value = (body.budget_category or "").strip() or None
    res = (
        supabase_db.table("transactions")
        .update({"budget_category": value})
        .eq("transaction_id", transaction_id)
        .eq("student_id", student_id)   # a student can only tag their own rows
        .execute()
    )
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Transaction not found")
    return {"status": "ok", "data": res.data[0]}