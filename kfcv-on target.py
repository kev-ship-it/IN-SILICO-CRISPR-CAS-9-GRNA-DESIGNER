import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# =========================
# CONFIG
# =========================
SEQ_LEN = 23
N_SPLITS = 5
EPOCHS = 25
BATCH_SIZE = 32

BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

# =========================
# ONE-HOT ENCODING
# =========================
def one_hot_encode(seq):
    seq = seq.upper().replace("U", "T")

    if len(seq) < SEQ_LEN:
        seq = seq.ljust(SEQ_LEN, "A")
    elif len(seq) > SEQ_LEN:
        seq = seq[:SEQ_LEN]

    x = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in BASE_MAP:
            x[i, BASE_MAP[b]] = 1.0
    return x

# =========================
# LOAD DATA (DeepCRISPR)
# =========================
# gRNA sequences
with open("eg_reg_on_target_seq.rsgt") as f:
    sequences = [line.strip() for line in f]

# labels (contains metadata → use pandas)
df_labels = pd.read_csv(
    "eg_reg_on_target.repisgt",
    sep="\t",
    header=None
)

labels = df_labels.iloc[:, -1].values.astype(np.float32)
min_len = min(len(sequences), len(labels))

sequences = sequences[:min_len]
labels = labels[:min_len]

X = np.array([one_hot_encode(s) for s in sequences])
y = labels


X = np.array([one_hot_encode(s) for s in sequences])
y = labels

print("Input shape:", X.shape)
print("Labels shape:", y.shape)

# =========================
# MODEL FACTORY
# =========================
def build_model():
    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(SEQ_LEN, 4)),
        MaxPooling1D(2),

        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model

# =========================
# K-FOLD CROSS VALIDATION
# =========================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
mae_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    print(f"Training fold {fold}/{N_SPLITS}...")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_model()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    _, mae = model.evaluate(X_val, y_val, verbose=0)
    mae_scores.append(mae)

# =========================
# PLOT RESULTS
# =========================
folds = np.arange(1, N_SPLITS + 1)
mean_mae = np.mean(mae_scores)

plt.figure(figsize=(8, 5))
plt.plot(folds, mae_scores, marker="o", linewidth=2, label="Fold MAE")
plt.axhline(mean_mae, linestyle="--", label=f"Mean MAE = {mean_mae:.4f}")

plt.xlabel("Fold")
plt.ylabel("MAE")
plt.title("K-Fold Cross-Validation – On-Target CNN")
plt.xticks(folds)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
