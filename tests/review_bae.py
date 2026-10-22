import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.metrics import pairwise_distances_argmin_min

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/bae.pth")
BUNDLE_PATH = os.path.join(BASE_DIR, "../models/bae_bundle.joblib")

if not os.path.exists(MODEL_PATH) or not os.path.exists(BUNDLE_PATH):
    print(" BAE files not found in /models. Ensure bae.pth and bae_bundle.joblib exist.")
else:
    bundle = joblib.load(BUNDLE_PATH)
    feature_scaler = bundle["feature_scaler"]
    latent_scaler = bundle["latent_scaler"]
    birch_centers = bundle["birch_centers"]
    threshold = bundle["threshold"]
    feature_columns = bundle["feature_columns"]
    stats = bundle["stats"]

    model = BirchAutoencoder(in_d=len(feature_columns), hidden_d=16, latent_d=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    print(f" BirchAutoencoder (BAE) weights loaded successfully. Timestamp: {os.path.getmtime(MODEL_PATH)}")
    print(f"   - Input features: {len(feature_columns)}")
    print(f"   - BIRCH clusters: {len(birch_centers)}")
    print(f"   - Anomaly threshold: {threshold:.4f}")

    sample_raw = np.zeros((1, len(feature_columns)), dtype=np.float32)
    sample_scaled = feature_scaler.transform(sample_raw)

    with torch.no_grad():
        sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)
        latent_val = model.encoder(sample_tensor).numpy()
        reconstructed = model(sample_tensor).numpy()

    recon_error = np.mean((sample_scaled - reconstructed) ** 2, axis=1)
    latent_scaled = latent_scaler.transform(latent_val)
    _, cluster_dist = pairwise_distances_argmin_min(latent_scaled, birch_centers)

    recon_score = (recon_error[0] - stats["recon_median"]) / stats["recon_iqr"]
    cluster_score = (cluster_dist[0] - stats["cluster_median"]) / stats["cluster_iqr"]
    final_score = (recon_score + cluster_score) / 2

    print(f" Inference sanity test passed. Score: {final_score:.4f} | Anomaly: {final_score >= threshold}")