import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, ops, models
from sklearn.model_selection import train_test_split


# This 'intercepts' the Embedding layer and deletes the broken config key
@tf.keras.utils.register_keras_serializable(package="Custom", name="Embedding")
class PatchedEmbedding(layers.Embedding):
    def __init__(self, *args, **kwargs):
        # Remove the bug-causing key if it exists
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# Now, when you build your model, use layers.Embedding as usual, 
# or if you are loading a sub-model, Keras will now use this patched version.

# 1. --- GLOBAL COMPATIBILITY PATCH ---
# This must be defined before any model building/loading
@tf.keras.utils.register_keras_serializable(package="Custom", name="Embedding")
class PatchedEmbedding(layers.Embedding):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

@tf.keras.utils.register_keras_serializable(package="Custom")
class SiameseDiff(layers.Layer):
    def call(self, inputs):
        return ops.abs(inputs[0] - inputs[1])

# 2. --- GPU ACCELERATION CONFIG ---
# This makes training ~3x faster on Colab T4 GPUs
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 3. --- DATA PIPELINE ---
def prepare_dataset(csv_path, batch_size=128):
    print("📂 Loading Dataset...")
    df = pd.read_csv(csv_path)
    VOCAB = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
    
    def encode(seq):
        seq = str(seq).upper()[:23].ljust(23, 'N')
        return [VOCAB.get(b, 0) for b in seq]

    print("🧬 Encoding sequences...")
    X_g = np.array([encode(s) for s in df['gRNA_Input']], dtype='int32')
    X_t = np.array([encode(s) for s in df['Target_Input']], dtype='int32')
    y = df['Risk_Score'].values.astype('float32')

    X_g_tr, X_g_val, X_t_tr, X_t_val, y_tr, y_val = train_test_split(
        X_g, X_t, y, test_size=0.1, random_state=42
    )

    # Create high-performance tf.data objects
    train_ds = tf.data.Dataset.from_tensor_slices(((X_g_tr, X_t_tr), y_tr))
    train_ds = train_ds.shuffle(20000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(((X_g_val, X_t_val), y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# 4. --- MODEL ARCHITECTURE ---
def build_transformer_siamese():
    # Shared Encoder
    input_seq = layers.Input(shape=(23,))
    x = PatchedEmbedding(input_dim=6, output_dim=128)(input_seq)
    
    # Transformer Block
    attn_output = layers.MultiHeadAttention(num_heads=8, key_dim=128)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    encoder = models.Model(input_seq, x, name="Shared_Encoder")

    # Siamese Inputs
    g_in = layers.Input(shape=(23,), name="gRNA_In")
    t_in = layers.Input(shape=(23,), name="Target_In")
    
    # Extract Features
    g_feat = encoder(g_in)
    t_feat = encoder(t_in)
    
    # Compare and Predict
    diff = SiameseDiff()([g_feat, t_feat])
    dense = layers.Dense(128, activation='relu')(diff)
    dense = layers.Dropout(0.2)(dense)
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(dense) # Force float32 for stability
    
    model = models.Model(inputs=[g_in, t_in], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss='mse', metrics=['mae'])
    return model

# 5. --- RUN TRAINING ---
if __name__ == "__main__":
    # Upload your file to Colab first!
    CSV_FILE = 'off_target_multi_variant.csv' 
    
    train_data, val_data = prepare_dataset(CSV_FILE)
    model = build_transformer_siamese()
    
    print("🚀 Training starting on T4 GPU...")
    
    # Callbacks to prevent overfitting and save the best version
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("_off_target_siamese.keras", save_best_only=True)
    ]
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=50,
        callbacks=callbacks
    )
    
    print("✅ Training Complete. Model saved as 'best_off_target_model.keras'")