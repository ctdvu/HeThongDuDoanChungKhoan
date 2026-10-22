import os
import numpy as np
import tensorflow as tf

MODELS_DIR = r"D:\N4_K1\HTTM\Code\TieuLuan\RNN_CoPhieu\models"
WEBAPP_MODELS_DIR = r"D:\N4_K1\HTTM\Code\TieuLuan\RNN_CoPhieu\webapp\models"

models = ['VanillaRNN', 'LSTM', 'GRU', 'BiLSTM', 'DeepStackedLSTM', 'best_model']

for m_name in models:
    path = os.path.join(MODELS_DIR, f"{m_name}.keras")
    if os.path.exists(path):
        m = tf.keras.models.load_model(path)
        weights = m.get_weights()
        out_path = os.path.join(MODELS_DIR, f"{m_name}_weights.npz")
        np.savez(out_path, *weights)
        
        # Also copy to webapp/models
        out_webapp = os.path.join(WEBAPP_MODELS_DIR, f"{m_name}_weights.npz")
        np.savez(out_webapp, *weights)
        print(f"Exported {m_name}: {len(weights)} arrays -> {out_path}")

print("All weights exported successfully!")
