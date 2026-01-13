import numpy as np
from tensorflow.keras.models import load_model
from utils.encoding import one_hot_encode_fixed

MODEL_PATH = "models/off_target_siamese.keras"
model = load_model(MODEL_PATH)

def predict_off_target(grna, target):
    """
    Siamese off-target prediction
    """
    Xg = np.expand_dims(one_hot_encode_fixed(grna, max_len=23), axis=0)
    Xt = np.expand_dims(one_hot_encode_fixed(target, max_len=23), axis=0)

    risk = model.predict([Xg, Xt], verbose=0)[0][0]
    return float(risk)
