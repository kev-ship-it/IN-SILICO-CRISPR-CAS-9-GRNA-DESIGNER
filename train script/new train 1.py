import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, ops
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from tensorflow.keras import mixed_precision

# 0. SPEED BOOST
mixed_precision.set_global_policy('mixed_float16')

# ==========================================
# 1. DATA PREPARATION (2 INPUTS)
# ==========================================
def load_dual_input_data(csv_path="on_target_multi_variant.csv"):
    print("📊 Loading dataset for Dual-Input...")
    df = pd.read_csv(csv_path)
    mapping = {'A':1, 'C':2, 'G':3, 'T':4, 'N':0}

    # Assuming 'gRNA_Input' is 20bp and 'Target_Context' is the remaining 54bp
    # If you only have one 74bp string, we slice it here:
    def split_tokenize(seq):
        seq = str(seq).upper()[:74].ljust(74, 'N')
        grna = [mapping.get(c, 0) for c in seq[20:40]] # Example slice for gRNA
        context = [mapping.get(c, 0) for c in seq[:20] + seq[40:]] # Remainder
        return grna, context

    X_g, X_c = [], []
    for s in df['Sequence_Input']:
        g, c = split_tokenize(s)
        X_g.append(g)
        X_c.append(c)

    y = df['Efficiency_Score'].values.astype(np.float32)
    
    return train_test_split(np.array(X_g), np.array(X_c), y, test_size=0.15, random_state=42)

# ==========================================
# 2. SHARED COMPONENTS
# ==========================================
def transformer_block(x, embed_dim, num_heads, ff_dim, rate=0.1):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn = layers.Dropout(rate)(attn)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = layers.Dense(ff_dim, activation="gelu")(x)
    ffn = layers.Dense(embed_dim)(ffn)
    ffn = layers.Dropout(rate)(ffn)
    x = layers.Add()([x, ffn])
    return layers.LayerNormalization(epsilon=1e-6)(x)

# ==========================================
# 3. DUAL-INPUT MODEL
# ==========================================
def build_dual_input_model(g_len=20, c_len=54):
    embed_dim = 192
    
    # --- INPUTS ---
    input_g = layers.Input(shape=(g_len,), name="gRNA_In")
    input_c = layers.Input(shape=(c_len,), name="Context_In")

    # --- SHARED EMBEDDING ---
    embedding_layer = layers.Embedding(input_dim=6, output_dim=embed_dim)
    
    g_embed = embedding_layer(input_g)
    c_embed = embedding_layer(input_c)

    # --- CONCATENATE IN LATENT SPACE ---
    # We join them here so the Multi-scale CNN and Transformer can see the whole picture
    x = layers.Concatenate(axis=1)([g_embed, c_embed]) # Shape: (None, 74, 192)

    # Positional Encoding (Applied to the combined 74bp representation)
    positions = tf.range(start=0, limit=74)
    pos_embed = layers.Embedding(input_dim=74, output_dim=embed_dim)(positions)
    x = x + pos_embed

    # --- MULTI-SCALE CNN ---
    conv3 = layers.Conv1D(64, 3, padding="same", activation="swish")(x)
    conv5 = layers.Conv1D(64, 5, padding="same", activation="swish")(x)
    conv7 = layers.Conv1D(64, 7, padding="same", activation="swish")(x)
    x = layers.Concatenate()([conv3, conv5, conv7])
    x = layers.Dense(embed_dim)(x)
    x = layers.BatchNormalization()(x)

    # --- TRANSFORMER BLOCKS ---
    for _ in range(2):
        x = transformer_block(x, embed_dim, num_heads=8, ff_dim=384)

    # --- REGRESSION HEAD ---
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="swish")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = models.Model(inputs=[input_g, input_c], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.AdamW(2e-4), loss="mse", metrics=["mae"])
    return model

# ==========================================
# 4. EXECUTION
# ==========================================
X_g_train, X_g_test, X_c_train, X_c_test, y_train, y_test = load_dual_input_data()

model = build_dual_input_model()

print("🚀 Dual-Input Training Started...")
model.fit(
    [X_g_train, X_c_train], y_train,
    validation_data=([X_g_test, X_c_test], y_test),
    epochs=30,
    batch_size=512,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
)

# SAVE
model.save("on_target_dual_input.keras")
print("✅ Saved as 'on_target_dual_input.keras'. This will now match your UI logic.")