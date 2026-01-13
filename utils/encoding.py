import numpy as np

BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot_encode_fixed(seq, max_len):
    """
    Pads or trims sequence to max_len and one-hot encodes it.
    Output shape: (max_len, 4)
    """
    seq = seq.upper().replace("U", "T")

    # pad or trim
    if len(seq) < max_len:
        seq = seq.ljust(max_len, "A")
    else:
        seq = seq[:max_len]

    encoded = np.zeros((max_len, 4), dtype=np.float32)

    for i, base in enumerate(seq):
        if base in BASE_MAP:
            encoded[i, BASE_MAP[base]] = 1.0

    return encoded
