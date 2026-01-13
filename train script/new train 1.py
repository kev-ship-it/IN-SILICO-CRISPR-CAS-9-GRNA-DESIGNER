import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout,
    LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# =========================
# CONFIG
# =========================
SEQ_LEN_GRNA = 20
SEQ_LEN_TARGET = 53
VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4}
VOCAB_SIZE = 5

# =========================
# ENCODING
# =========================
def encode_seq(seq, max_len):
    seq = seq.upper().replace("U", "T")
    arr = np.zeros(max_len, dtype=np.int32)
    for i, b in enumerate(seq[:max_len]):
        arr[i] = VOCAB.get(b, 0)
    return arr

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("on_target_regression.csv")

grna_seqs = df["gRNA"].astype(str).tolist()
target_seqs = df["Extended Target"].astype(str).tolist()
labels = df["Edit Efficiency"].astype(np.float32).values

# 🔥 normalize labels (IMPORTANT)
labels = labels / 100.0

Xg = np.array([encode_seq(s, SEQ_LEN_GRNA) for s in grna_seqs])
Xt = np.array([encode_seq(s, SEQ_LEN_TARGET) for s in target_seqs])
y = labels

Xg_train, Xg_test, Xt_train, Xt_test, y_train, y_test = train_test_split(
    Xg, Xt, y, test_size=0.2, random_state=42
)

# =========================
# TRANSFORMER BLOCK
# =========================
def transformer_block(x, heads=4, dim=64, drop=0.2):
    attn = MultiHeadAttention(num_heads=heads, key_dim=dim)(x, x)
    x = LayerNormalization()(x + attn)

    ff = Dense(dim * 2, activation="relu")(x)
    ff = Dense(dim)(ff)
    x = LayerNormalization()(x + ff)

    return Dropout(drop)(x)

def transformer_encoder(seq_len):
    inp = Input(shape=(seq_len,))
    x = Embedding(VOCAB_SIZE, 64)(inp)

    x = transformer_block(x)
    x = transformer_block(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)

    return Model(inp, x)

# =========================
# BUILD MODEL
# =========================
grna_encoder = transformer_encoder(SEQ_LEN_GRNA)
target_encoder = transformer_encoder(SEQ_LEN_TARGET)

g_in = Input(shape=(SEQ_LEN_GRNA,), name="gRNA_input")
t_in = Input(shape=(SEQ_LEN_TARGET,), name="Target_input")

g_feat = grna_encoder(g_in)
t_feat = target_encoder(t_in)

x = Concatenate()([g_feat, t_feat])
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
out = Dense(1)(x)

model = Model([g_in, t_in], out)

model.compile(
    optimizer=Adam(1e-3),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =========================
# TRAIN
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    [Xg_train, Xt_train], y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# =========================
# EVALUATE
# =========================
loss, mae = model.evaluate([Xg_test, Xt_test], y_test)
print(f"\nTest MAE (normalized): {mae:.4f}")
print(f"Test MAE (%): {mae * 100:.2f}%")

# =========================
# BASELINE CHECK
# =========================
baseline = np.mean(np.abs(y_test - np.mean(y_train)))
print(f"Baseline MAE (%): {baseline * 100:.2f}%")

# =========================
# PLOTS
# =========================
plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.title("Training Loss")
plt.show()

preds = model.predict([Xg_test, Xt_test]).flatten()

plt.figure()
plt.scatter(y_test * 100, preds * 100, alpha=0.5)
plt.xlabel("True Efficiency (%)")
plt.ylabel("Predicted Efficiency (%)")
plt.title("Prediction vs Ground Truth")
plt.show()

# =========================
# SAVE MODEL
# =========================
model.save("on_target_transformer.keras")
print("Model saved ✅")
