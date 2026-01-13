import numpy as np
import tensorflow as tf

# ------------------
# CONFIG
# ------------------
MODEL_PATH = "models/on_target_dual_transformer.keras"   # or .keras
SEQ_LEN = 23
BASE_MAP = {"A":0, "C":1, "G":2, "T":3}

# ------------------
# ENCODER
# ------------------
def one_hot_encode(seq):
    seq = seq.upper().replace("U", "T")

    if len(seq) < SEQ_LEN:
        seq = seq.ljust(SEQ_LEN, "A")
    else:
        seq = seq[:SEQ_LEN]

    x = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in BASE_MAP:
            x[i, BASE_MAP[b]] = 1.0

    return x

# ------------------
# LOAD MODEL
# ------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False   # IMPORTANT (avoids metric deserialization issues)
)

# ------------------
# PREDICT ON-TARGET
# ------------------
def predict_on_target(seq, target_seq=None):
    X1 = np.expand_dims(one_hot_encode(seq), axis=0)
    if target_seq is None:
        target_seq = seq
    X2 = np.expand_dims(one_hot_encode(target_seq), axis=0)
    
    # feed as list for dual-input model
    score = model.predict([X1, X2], verbose=0)[0][0]
    return float(score)

# ------------------
# LOCAL TEST
# ------------------
if __name__ == "__main__":
    seq = "GAGGTGTCCGGCATCAAGGCCGCCTACGAGGCCGAGGGCGGGGATGCCCGCAA"
    score = predict_on_target(seq)
    print(f"Sequence: {seq}")
    print(f"Predicted on-target efficiency: {score:.4f}")
