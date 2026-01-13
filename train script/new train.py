import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CONSTANTS
# =========================
SEQ_LEN = 23
BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

# =========================
# 1. LOAD DATA
# =========================
SEQ_FILE = "eg_reg_on_target_seq.rsgt"  # replace with your file path
df = pd.read_csv(SEQ_FILE, sep="\t", header=None)

sequences = df.iloc[:, 4].astype(str).tolist()
labels = df.iloc[:, 5].astype(np.float32).values

print("Samples:", len(sequences))
print("Example seq:", sequences[0])
print("Example label:", labels[0])

# =========================
# 2. SCALE LABELS TO 0-1
# =========================
y_max = labels.max()
labels_scaled = labels / y_max
print("Max scaled label:", labels_scaled.max())

# =========================
# 3. ONE-HOT ENCODING
# =========================
def one_hot_encode(seq):
    seq = seq.upper().replace("U", "T")
    seq = seq.ljust(SEQ_LEN, "A")[:SEQ_LEN]
    enc = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in BASE_MAP:
            enc[i, BASE_MAP[b]] = 1.0
    return enc

X = np.array([one_hot_encode(s) for s in sequences], dtype=np.float32)
y = labels_scaled.astype(np.float32)

print("Input shape:", X.shape)  # (N, 23, 4)

# =========================
# 4. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. BUILD CNN MODEL
# =========================
model = Sequential([
    Conv1D(64, 2, activation="relu", padding="same", input_shape=(SEQ_LEN, 4)),
    BatchNormalization(),
    Conv1D(64, 3, activation="relu", padding="same"),
    MaxPooling1D(2),
    Dropout(0.25),

    Conv1D(128, 3, activation="relu", padding="same"),
    BatchNormalization(),
    GlobalMaxPooling1D(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.25),
    Dense(1, activation="sigmoid")  # output 0-1
])

model.compile(
    optimizer=Adam(0.0005),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =========================
# 6. TRAIN WITH EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor="val_mae",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# =========================
# 7. EVALUATE
# =========================
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE (scaled 0-1): {mae:.4f}")

# =========================
# 8. SAVE MODEL
# =========================
model.save("models/on_target_model_improved.keras")
print("Saved: models/on_target_model_scaled.keras")

# =========================
# 9. PREDICTION HELPER
# =========================
def predict_efficiency(seq):
    X_seq = np.array([one_hot_encode(seq)], dtype=np.float32)
    pred = model.predict(X_seq, verbose=0)[0][0]
    return float(pred)

# Example usage
example_seq = "GCGTACGATCGGATCGTACGG"
print(f"Sequence: {example_seq}")
print(f"Predicted on-target efficiency (0-1): {predict_efficiency(example_seq):.4f}")
