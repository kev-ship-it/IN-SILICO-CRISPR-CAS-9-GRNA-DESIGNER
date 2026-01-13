import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# =========================
# 1. LOAD DATA
# =========================
SEQ_FILE = "eg_reg_on_target_seq.rsgt"    # your input file with gRNA sequences + other info

# Load sequences as DataFrame
seq_df = pd.read_csv(SEQ_FILE, sep="\t", header=None)

# Extract gRNA sequences (column index 4) and numeric labels (column index 5)
sequences = seq_df.iloc[:, 4].astype(str).tolist()
labels = seq_df.iloc[:, 5].astype(np.float32).tolist()

# Sanity check
print("Number of sequences:", len(sequences))
print("Number of labels:", len(labels))
print("First sequence:", sequences[0])
print("First label:", labels[0])

# Ensure sequences and labels match
assert len(sequences) == len(labels), "Number of sequences and labels must match!"

# =========================
# 2. ONE-HOT ENCODING WITH PADDING
# =========================
max_len = max(len(seq) for seq in sequences)

def one_hot_encode_padded(seq, max_len):
    mapping = {"A":0, "C":1, "G":2, "T":3}
    enc = np.zeros((max_len, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            enc[i, mapping[base]] = 1
    return enc

X = np.array([one_hot_encode_padded(seq, max_len) for seq in sequences])
y = np.array(labels, dtype=np.float32)

print("Input shape after padding:", X.shape)

# =========================
# 3. TRAIN/TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. BUILD CNN MODEL (INPUT-LENGTH AGNOSTIC)
# =========================
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(max_len, 4)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation="relu"),
    GlobalMaxPooling1D(),   # allows variable-length sequences
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1)  # regression output
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =========================
# 5. TRAIN MODEL
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32
)

# =========================
# 6. EVALUATE MODEL
# =========================
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")

# =========================
# 7. SAVE MODEL
# =========================
model.save("on_target_model_variable_input_padded.h5")
print("Model saved as on_target_model_variable_input_padded.h5")
