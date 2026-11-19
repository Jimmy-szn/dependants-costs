"""Contract test: the shape other teams rely on. If this breaks, you broke someone else."""
from fastapi.testclient import TestClient

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
