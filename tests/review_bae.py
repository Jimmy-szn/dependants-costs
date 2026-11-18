from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import pairwise_distances_argmin_min


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mpesa_synthetic.csv"
MODEL_PATH = ROOT / "models" / "bae.pth"
BUNDLE_PATH = ROOT / "models" / "bae_bundle.joblib"

FEATURE_COLUMNS = [
    "amount",
    "sender_balance_before",
    "sender_balance_after",
    "amount_ratio",
    "drained",
    "sender_gap",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]


class BirchAutoencoder(nn.Module):
    def __init__(self, in_d: int, hidden_d: int = 32, latent_d: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, latent_d),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, in_d),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "amount",
        "sender_balance_before",
        "sender_balance_after",
        "hour",
        "day_of_week",
    }
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    features = transactions.copy()
    features["sender_gap"] = (
        features["sender_balance_before"]
        - features["amount"]
        - features["sender_balance_after"]
    )
    features.loc[features["sender_gap"].abs() < 1e-6, "sender_gap"] = 0.0
    features["amount_ratio"] = features["amount"] / (
        features["sender_balance_before"] + 1
    )
    features["drained"] = (features["sender_balance_after"] <= 0).astype(float)
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)

    weekday_number = features["day_of_week"].map(
        {
            "Mon": 0,
            "Tue": 1,
            "Wed": 2,
            "Thu": 3,
            "Fri": 4,
            "Sat": 5,
            "Sun": 6,
        }
    )
    if weekday_number.isna().any():
        unexpected = sorted(features.loc[weekday_number.isna(), "day_of_week"].unique())
        raise ValueError(f"Unexpected or missing day_of_week values: {unexpected}")

    features["weekday_sin"] = np.sin(2 * np.pi * weekday_number / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday_number / 7)
    features["sender_gap"] = np.sign(features["sender_gap"]) * np.log1p(
        np.abs(features["sender_gap"])
    )

    for column in ("amount", "sender_balance_before", "sender_balance_after"):
        features[column] = np.log1p(features[column].clip(lower=0))

    result = features[FEATURE_COLUMNS].astype(np.float32)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Feature pipeline produced non-finite values")
    return result


def main() -> None:
    for path in (DATA_PATH, MODEL_PATH, BUNDLE_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required BAE test file not found: {path}")

    bundle = joblib.load(BUNDLE_PATH)
    bundle_columns = list(bundle["feature_columns"])
    if bundle_columns != FEATURE_COLUMNS:
        raise ValueError(
            "BAE bundle does not match the reduced-feature pipeline. "
            f"Expected {FEATURE_COLUMNS}; found {bundle_columns}. "
            "Copy the matching retrained bae.pth and bae_bundle.joblib into models/."
        )

    transactions = pd.read_csv(DATA_PATH)
    if "month_2026" not in transactions or "is_fraud" not in transactions:
        raise ValueError("Dataset must include month_2026 and is_fraud columns")

    test_mask = transactions["month_2026"] >= 11
    test_transactions = transactions.loc[test_mask].reset_index(drop=True)
    if test_transactions.empty:
        raise ValueError("The chronological test split (month_2026 >= 11) is empty")

    y_test = test_transactions["is_fraud"].astype(bool).to_numpy()
    test_features = build_features(test_transactions)
    scaled_test = bundle["feature_scaler"].transform(
        test_features[bundle_columns]
    ).astype(np.float32)

    model = BirchAutoencoder(in_d=len(bundle_columns))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        values = torch.from_numpy(scaled_test)
        latent = model.encoder(values).numpy()
        reconstructed = model(values).numpy()

    reconstruction_error = np.mean(
        (scaled_test - reconstructed) ** 2,
        axis=1,
    )
    latent_scaled = bundle["latent_scaler"].transform(latent)
    _, cluster_distances = pairwise_distances_argmin_min(
        latent_scaled,
        bundle["birch_centers"],
    )

    stats = bundle["stats"]
    recon_iqr = float(stats["recon_iqr"])
    cluster_iqr = float(stats["cluster_iqr"])
    if recon_iqr <= 0 or cluster_iqr <= 0:
        raise ValueError("BAE bundle has invalid score-normalization IQR values")

    recon_score = (reconstruction_error - stats["recon_median"]) / recon_iqr
    cluster_score = (cluster_distances - stats["cluster_median"]) / cluster_iqr
    scores = (recon_score + cluster_score) / 2
    if not np.isfinite(scores).all():
        raise ValueError("BAE produced non-finite anomaly scores")

    threshold = float(bundle["threshold"])
    predictions = scores >= threshold
    print(f"Test rows: {len(test_transactions)}")
    print(f"Fraud labels: {int(y_test.sum())} / {len(y_test)}")
    print(f"Input features: {len(bundle_columns)}")
    print(f"Anomaly threshold: {threshold:.6f}")
    print(
        "Reconstruction error p50/p95/max: "
        f"{np.percentile(reconstruction_error, 50):.6g} / "
        f"{np.percentile(reconstruction_error, 95):.6g} / "
        f"{reconstruction_error.max():.6g}"
    )
    print(
        "Combined score p50/p95/max: "
        f"{np.percentile(scores, 50):.4f} / "
        f"{np.percentile(scores, 95):.4f} / {scores.max():.4f}"
    )
    print(f"Alerts: {int(predictions.sum())}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, scores):.4f}")

    worst_indices = np.argsort(reconstruction_error)[-5:][::-1]
    worst_rows = test_transactions.iloc[worst_indices][
        [
            "transaction_id",
            "amount",
            "sender_balance_before",
            "sender_balance_after",
            "hour",
            "day_of_week",
            "is_fraud",
        ]
    ].copy()
    worst_rows["reconstruction_error"] = reconstruction_error[worst_indices]
    worst_rows["combined_score"] = scores[worst_indices]
    worst_rows["flagged"] = predictions[worst_indices]
    print("\nWorst five reconstruction errors:")
    print(worst_rows.to_string(index=False))

    worst_index = int(worst_indices[0])
    feature_diagnostics = pd.DataFrame(
        {
            "feature": bundle_columns,
            "raw_feature": test_features.iloc[worst_index].to_numpy(),
            "scaler_mean": bundle["feature_scaler"].mean_,
            "scaler_scale": bundle["feature_scaler"].scale_,
            "scaled_input": scaled_test[worst_index],
            "reconstructed": reconstructed[worst_index],
        }
    )
    print("\nWorst row feature scaling and reconstruction:")
    print(feature_diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
