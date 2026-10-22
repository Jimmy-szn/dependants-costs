from datetime import datetime

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
from pydantic import BaseModel
from typing import List

# Import schema boundaries securely
from app.schemas import Request, Response
from app.service import handle


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

import os
import torch
import torch.nn as nn
import joblib

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


class SyncRequest(BaseModel):
    transactionData: List[dict]
class ForecastRequest(BaseModel):
    student_id: str
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

@app.get("/view-budgets")
async def budget(student_id: str = Depends(get_current_user_id)):
    response = supabase_db.table("budget_caps").select("*").eq("student_id", student_id).execute()
    return {"status": "ok", "data": response.data}
@app.get('/fetch-monthly-transactions')
async def fetch_monthly_transactions(student_id: str = Depends(get_current_user_id)):
    try:
        now = datetime.now()
        start_of_month = datetime(now.year, now.month, 1).isoformat()
        response = supabase_db.table("transactions").select("*").eq("student_id", student_id).gte("created_at", start_of_month).order("created_at", desc=True).execute()
        transactions_response = response.data if response.data else []
        return {"status": "success", "transactions": transactions_response}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching transactions: {str(e)}",
        )
@app.post('/create-budget')
async def create_budget(item: dict, student_id: str = Depends(get_current_user_id)):
    sanitized_item = {
        "budget_id": str(uuid.uuid4()),
        "student_id": str(student_id),
        "category": item.get("category", "uncategorized"),
        "limit_amount": item.get("limit_amount", 0)
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
async def sync_transaction(request: SyncRequest):
    try:
        response = supabase_db.table("transactions").insert(request.transactionData).execute()
        return {"status": "ok", "data": response.data}
    except Exception as e:
        print(f"Database Exception Encountered: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,           
            detail=str(e),
        )


@app.post("/predict-budget")
async def predict_budget(request: ForecastRequest):
   
    try:
        now = datetime.now()
        start_of_month = datetime(now.year, now.month, 1).isoformat()
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
        if 'x_scaler' in globals():
            scaled_features = x_scaler.transform(raw_features)
        else:
            scaled_features = raw_features
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor).item()
        if 'y_scaler' in globals():
            forecasted_value = y_scaler.inverse_transform([[prediction]])[0][0]
        else:
            forecasted_value = prediction
        return {"status": "success", "forecasted_amount": round(forecasted_value, 2), "transactions_analyzed":len(transactions_response), "category_sums": caregory_sums, "gross_outflow": gross_outflow}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        ) 

