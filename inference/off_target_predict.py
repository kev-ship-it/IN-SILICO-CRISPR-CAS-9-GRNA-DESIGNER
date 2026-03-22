import numpy as np
import tensorflow as tf
from tensorflow.keras import ops, layers

# =========================================================
# 1. CUSTOM DESERIALIZATION CLASSES
# =========================================================

# CRITICAL: Without this, Keras cannot open the 'Shared_Encoder' inside your model
@tf.keras.utils.register_keras_serializable(package="Custom", name="Embedding")
class PatchedEmbedding(layers.Embedding):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

@tf.keras.utils.register_keras_serializable(package="Custom")
class SiameseDiff(layers.Layer):
    def call(self, inputs):
        # Calculates absolute difference between the two Transformer branches
        return ops.abs(inputs[0] - inputs[1])

# =========================================================
# 2. MODEL INITIALIZATION
# =========================================================

MODEL_PATH = "models/off_target_siamese.keras"

# We load the model once when the module is imported to save memory/time
try:
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={
            "Embedding": PatchedEmbedding, 
            "SiameseDiff": SiameseDiff
        }
    )
except Exception as e:
    print(f"⚠️ Warning: Could not load off-target model from {MODEL_PATH}: {e}")
    model = None

# =========================================================
# 3. EXPORTED PREDICTION FUNCTION
# =========================================================

def predict_off_target(grna, target_with_pam):
    """
    Module function: Returns the float risk score for a single gRNA/Target pair.
    """
    if model is None:
        return 0.0

    VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 0}
    
    def encode(seq, length=23):
        # Normalize to 23bp and map to integers
        seq = str(seq).upper()[:length].ljust(length, "N")
        return np.array([[VOCAB.get(b, 0) for b in seq]], dtype=np.int32)

    # Encode inputs
    Xg = encode(grna) 
    Xt = encode(target_with_pam)

    # Siamese Inference (Requires a list of two inputs)
    # verbose=0 keeps the console clean during UI usage
    prediction = model.predict([Xg, Xt], verbose=0)
    
    # Return ONLY the risk score as a standard float
    return float(prediction[0][0])
    print(prediction[0][0])