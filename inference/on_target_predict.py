import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.encoding import one_hot_encode_fixed

MODEL_PATH = "models/on_target_dual_transformer.keras"

# Load model ONCE
model = tf.keras.models.load_model(MODEL_PATH)
def preprocess_seq_integer(seq, seq_len):
    VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4}
    seq = seq.upper().replace("U", "T")
    arr = np.zeros(seq_len, dtype=np.int32)
    for i, b in enumerate(seq[:seq_len]):
        arr[i] = VOCAB.get(b, 0)
    return arr.reshape(1, seq_len)

def predict_on_target(grna_seq, target_seq):
    """
    Inputs:
        grna_seq   : 20 bp string
        target_seq : 53 bp string
    Output:
        float score (0–1)
    """

    # Encode separately
    grna_enc = preprocess_seq_integer(grna_seq, 20)     # (20, 4)
    target_enc = preprocess_seq_integer(target_seq, 53) # (53, 4)

    # Transformer prediction
    score = model.predict(
        [grna_enc, target_enc],
        verbose=0
    )[0][0]

    return float(score)

