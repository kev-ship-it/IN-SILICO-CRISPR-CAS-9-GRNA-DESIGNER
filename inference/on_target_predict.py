import numpy as np
import tensorflow as tf

MODEL_PATH = "models/on_target_dual_transformer.keras" # Ensure this name matches your saved file
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_seq_integer(seq, seq_len):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
    seq = str(seq).upper()[:seq_len].ljust(seq_len, 'N')
    arr = np.array([mapping.get(base, 0) for base in seq], dtype=np.int32)
    return arr.reshape(1, seq_len)

def predict_on_target(grna_seq, context_seq):
    # Encode separately for the Dual-Input architecture
    grna_enc = preprocess_seq_integer(grna_seq, 20)
    context_enc = preprocess_seq_integer(context_seq, 54)

    # Predict
    score = model.predict([grna_enc, context_enc], verbose=0)
    return float(score[0][0])