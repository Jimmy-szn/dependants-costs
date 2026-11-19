"""Contract test: the shape other teams rely on. If this breaks, you broke someone else."""
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

import app.main as main

client = TestClient(main.app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_endpoint_responds():
    res = client.post("/projection", json={})
    assert res.status_code == 200, res.text
    assert "method" in res.json()


class FakeTable:
    def __init__(self):
        self.inserted = []

    def insert(self, record):
        self.inserted.append(record)
        return self

    def execute(self):
        return None


class FakeDatabase:
    def __init__(self):
        self.ai_results = FakeTable()

    def table(self, name):
        assert name == "ai_results"
        return self.ai_results


def test_full_bae_evaluation_persists_scores(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(main, "supabase_db", database)

    result = main.persist_bae_evaluation(
        trans_id="tx-1",
        stage="bae_full",
        scored=True,
        risk_flag=True,
        anomaly_score=2.34567,
        reconstruction_error=1.23456,
    )

    record = database.ai_results.inserted[0]
    assert record["transaction_id"] == "tx-1"
    assert record["anomaly_score"] == 2.3457
    assert record["track_b_reconstruct_err"] == 1.2346
    assert record["evaluation_stage"] == "bae_full"
    assert record["scored"] is True
    assert result["results_id"] == record["results_id"]


def test_not_scored_evaluation_persists_reason(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(main, "supabase_db", database)

    result = main.not_scored("Missing timestamp", "tx-2")

    record = database.ai_results.inserted[0]
    assert record["transaction_id"] == "tx-2"
    assert record["anomaly_score"] is None
    assert record["evaluation_stage"] == "not_scored"
    assert record["scored"] is False
    assert record["evaluation_reason"] == "Missing timestamp"
    assert result["stage"] == "not_scored"


def test_live_features_account_for_explicit_transaction_fee():
    features = main.add_features(
        pd.DataFrame(
            [{
                "amount": 250.0,
                "transaction_fee": 7.0,
                "sender_balance_before": 983.05,
                "sender_balance_after": 726.05,
                "hour": 16,
                "day_of_week": "Wed",
            }]
        )
    )

    assert features.loc[0, "sender_gap"] == 0.0
    assert features.loc[0, "amount"] == pytest.approx(np.log1p(250.0))


def test_incoming_transaction_is_persisted_as_not_scored(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(main, "supabase_db", database)
    monkeypatch.setattr(main, "bae_model", object())
    monkeypatch.setattr(main, "bae_bundle", {"feature_columns": []})

    result = main.score_transaction_with_bae(
        {
            "amount_in": 1000.0,
            "amount_out": 0.0,
            "balance": 1726.05,
            "transaction_time": "2026-11-18T16:55:00+03:00",
            "sender_balance_before": 726.05,
            "transaction_fee": 0.0,
        },
        trans_id="incoming-tx",
    )

    assert result["stage"] == "not_scored"
    assert "outgoing transactions" in result["reason"]
    assert database.ai_results.inserted[0]["scored"] is False


def test_outgoing_transaction_requires_explicit_fee(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(main, "supabase_db", database)
    monkeypatch.setattr(main, "bae_model", object())
    monkeypatch.setattr(main, "bae_bundle", {"feature_columns": []})

    result = main.score_transaction_with_bae(
        {
            "amount_out": 20.0,
            "amount_in": 0.0,
            "balance": 980.0,
            "transaction_time": "2026-11-18T16:55:00+03:00",
            "sender_balance_before": 1000.0,
        },
        trans_id="outgoing-without-fee",
    )

    assert result["stage"] == "not_scored"
    assert "transaction_fee is required" in result["reason"]


def test_outgoing_transaction_with_rising_balance_is_not_scored(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(main, "supabase_db", database)
    monkeypatch.setattr(main, "bae_model", object())
    monkeypatch.setattr(main, "bae_bundle", {"feature_columns": []})

    result = main.score_transaction_with_bae(
        {
            "amount_out": 20.0,
            "amount_in": 0.0,
            "balance": 1000.05,
            "transaction_time": "2026-11-18T16:55:00+03:00",
            "sender_balance_before": 1000.0,
            "transaction_fee": 0.0,
        },
        trans_id="outgoing-rising-balance",
    )

    assert result["stage"] == "not_scored"
    assert "increased the balance" in result["reason"]
