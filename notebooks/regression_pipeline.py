"""
Basic regression pipeline in TensorFlow/Keras.
Input: 24 numeric features (X), 1 continuous target (y).

Assumptions (flagged):
  - All 24 features are numeric and already encoded (no categoricals).
  - Rows with missing values are handled before this step.
  - Replace the synthetic data block with your real data load.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

N_FEATURES = 24

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
# Synthetic placeholder. Swap with e.g. pd.read_csv("your_file.csv"),
# then X = df[feature_cols].values and y = df["target"].values
X = np.random.randn(5000, N_FEATURES).astype("float32")
true_w = np.random.randn(N_FEATURES)
y = (X @ true_w + 0.5 * np.random.randn(5000)).astype("float32")

# ---------------------------------------------------------------------------
# 2. Split: 70% train, 15% validation, 15% test
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

# ---------------------------------------------------------------------------
# 3. Scale features (fit on train only to avoid data leakage)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Scale the target too; helps optimisation when y has a large range.
# Predictions are inverse-transformed back to the original units below.
y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

# ---------------------------------------------------------------------------
# 4. Build model: small MLP with a single linear output unit
# ---------------------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(N_FEATURES,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.1),            # light regularisation
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),                # linear output for regression
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"],
)
model.summary()

# ---------------------------------------------------------------------------
# 5. Train with early stopping on validation loss
# ---------------------------------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train, y_train_s,
    validation_data=(X_val, y_val_s),
    epochs=200,
    batch_size=64,
    callbacks=[early_stop],
    verbose=2,
)

# ---------------------------------------------------------------------------
# 6. Evaluate on the held-out test set, in original target units
# ---------------------------------------------------------------------------
pred_s = model.predict(X_test).ravel()
pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()

print("\nTest results")
print(f"  MAE : {mean_absolute_error(y_test, pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f}")
print(f"  R2  : {r2_score(y_test, pred):.4f}")

# ---------------------------------------------------------------------------
# 7. Save model (scalers should be saved too if you plan to reuse them)
# ---------------------------------------------------------------------------
model.save("regression_model.keras")
