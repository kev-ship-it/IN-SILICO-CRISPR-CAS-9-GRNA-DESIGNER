import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")

ON_TARGET_MODEL = os.path.join(MODELS_DIR, "on_target_model.h5")
OFF_TARGET_MODEL = os.path.join(MODELS_DIR, "off_target_model.keras")

SEQ_LEN = 23
