import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

SEQ_LEN = 23
BASE_MAP = {"A":0,"C":1,"G":2,"T":3}

def encode(seq):
    seq = seq.upper().replace("U","T")[:SEQ_LEN].ljust(SEQ_LEN,"A")
    x = np.zeros((SEQ_LEN,4))
    for i,b in enumerate(seq):
        if b in BASE_MAP:
            x[i,BASE_MAP[b]] = 1
    return x

# ------------------
# LOAD DATA
# ------------------
grnas, targets, labels = [], [], []

with open("eg_cls_off_target.epiotrt") as f:
    for line in f:
        parts = line.strip().split()

        g = parts[0]      
        t = parts[1]      
        y = parts[-1]    

        grnas.append(encode(g))
        targets.append(encode(t))
        labels.append(int(y))

Xg = np.array(grnas)
Xt = np.array(targets)
y  = np.array(labels)

# ------------------
# SHARED CNN
# ------------------
def shared_cnn():
    inp = Input(shape=(SEQ_LEN,4))
    x = Conv1D(64,3,activation="relu")(inp)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    return Model(inp,x)

cnn = shared_cnn()

g_in = Input(shape=(SEQ_LEN,4))
t_in = Input(shape=(SEQ_LEN,4))

g_feat = cnn(g_in)
t_feat = cnn(t_in)

merged = Concatenate()([g_feat, t_feat])
x = Dense(64, activation="relu")(merged)
out = Dense(1, activation="sigmoid")(x)

model = Model([g_in, t_in], out)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit([Xg, Xt], y, epochs=10, batch_size=32)
model.save("models/off_target_siamese.keras")
