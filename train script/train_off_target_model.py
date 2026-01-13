import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, concatenate, Dropout
from tensorflow.keras.optimizers import Adam

# =========================
# CONFIG
# =========================
DATA_FILE = "eg_cls_off_target.epiotrt"
SEQ_LEN = 23   # DeepCRISPR standard

# =========================
# 1. LOAD DATA
# =========================
grna_seqs = []
off_seqs = []
labels = []

with open(DATA_FILE, "r") as f:
    for line in f:
        parts = line.strip().split("\t")

        # Defensive parsing (DeepCRISPR-safe)
        grna_seqs.append(parts[0])
        off_seqs.append(parts[1])
        labels.append(float(parts[-1]))  # label is ALWAYS last column

labels = np.array(labels, dtype=np.float32)

print("Loaded samples:", len(labels))
print("Example gRNA:", grna_seqs[0])
print("Example off-target:", off_seqs[0])
print("Example label:", labels[0])

# =========================
# 2. SEQUENCE NORMALIZATION
# =========================
def pad_or_truncate(seq, max_len=23):
    seq = seq.upper()
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + "N" * (max_len - len(seq))

def one_hot(seq):
    mapping = {"A":0, "C":1, "G":2, "T":3}
    enc = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    seq = pad_or_truncate(seq, SEQ_LEN)

    for i, base in enumerate(seq):
        if base in mapping:
            enc[i, mapping[base]] = 1.0
        # N -> zero vector
    return enc

X_grna = np.array([one_hot(s) for s in grna_seqs])
X_off  = np.array([one_hot(s) for s in off_seqs])
y = labels

print("X_grna shape:", X_grna.shape)
print("X_off shape:", X_off.shape)

# =========================
# 3. TRAIN / TEST SPLIT
# =========================
Xg_train, Xg_test, Xo_train, Xo_test, y_train, y_test = train_test_split(
    X_grna, X_off, y, test_size=0.2, random_state=42
)

# =========================
# 4. SIAMESE CNN MODEL
# =========================
def build_encoder():
    inp = Input(shape=(SEQ_LEN, 4))
    x = Conv1D(64, 3, activation="relu")(inp)
    x = Conv1D(128, 3, activation="relu")(x)
    x = Flatten()(x)
    return Model(inp, x)

encoder = build_encoder()

grna_input = Input(shape=(SEQ_LEN, 4))
off_input  = Input(shape=(SEQ_LEN, 4))

grna_feat = encoder(grna_input)
off_feat  = encoder(off_input)

merged = concatenate([grna_feat, off_feat])
x = Dense(256, activation="relu")(merged)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=[grna_input, off_input], outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 5. TRAIN
# =========================
model.fit(
    [Xg_train, Xo_train],
    y_train,
    validation_data=([Xg_test, Xo_test], y_test),
    epochs=20,
    batch_size=64
)

# =========================
# 6. EVALUATE
# =========================
loss, acc = model.evaluate([Xg_test, Xo_test], y_test)
print(f"Test Accuracy: {acc:.4f}")

# =========================
# 7. SAVE MODEL
# =========================
model.save("off_target_model.keras")
print("Saved as off_target_model.keras")
