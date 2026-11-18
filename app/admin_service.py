import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not service_role_key:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured"
    )

admin_db = create_client(supabase_url, service_role_key)


def get_overview() -> dict:
    transactions = (
        admin_db.table("transactions")
        .select("transaction_id", count="exact", head=True)
        .execute()
    )
    flagged_results = (
        admin_db.table("ai_results")
        .select("results_id", count="exact", head=True)
        .eq("risk_flag", True)
        .execute()
    )

    if transactions.count is None or flagged_results.count is None:
        raise RuntimeError("Supabase did not return dashboard metric counts")

    return {
        "transactions_total": transactions.count,
        "flagged_ai_results_total": flagged_results.count,
    }

def get_ai_results() -> list[dict]:
    results = (
        admin_db.table("ai_results")
        .select("*")
        .order("evaluated_at", desc=True)
        .eq("risk_flag", True)
        .limit(100)
        .execute()
    )
    if results.data is None:
        raise RuntimeError("Supabase did not return AI results")
    if not results.data:
        return []

    transactions = (
        admin_db.table("transactions")
        .select("*")
        .in_("transaction_id", [result["transaction_id"] for result in results.data])
        .execute()
    )

    if transactions.data is None:
        raise RuntimeError("Supabase did not return transaction data")
    combined_results = []
    for result in results.data:
        transaction = next(
            (t for t in transactions.data if t["transaction_id"] == result["transaction_id"]),
            None,
        )
        combined_results.append({"ai_result": result, "transaction": transaction})
    return combined_results