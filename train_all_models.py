# -*- coding: utf-8 -*-
"""
Script huấn luyện tất cả 5 mô hình RNN trên dữ liệu VN-Index
và xuất file models + metrics để phục vụ Web App và Notebook
"""

import os
import sys
import json
import time
import joblib
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_curve, f1_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, SimpleRNN, Dropout, BatchNormalization,
    Bidirectional, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Set random seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'HOSE_DLY_VNINDEX1D.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Loading data from: {DATA_PATH}")
df_raw = pd.read_csv(DATA_PATH)
df_raw['date'] = pd.to_datetime(df_raw['time'], unit='s')
df_raw = df_raw.sort_values('date').reset_index(drop=True)
print(f"Raw shape: {df_raw.shape}, From {df_raw['date'].min().date()} to {df_raw['date'].max().date()}")

# Feature Engineering
def feature_engineering(df):
    d = df.copy()
    
    # 1. Returns
    d['daily_return'] = d['close'].pct_change() * 100
    d['log_return'] = np.log(d['close'] / d['close'].shift(1))
    d['high_low_pct'] = (d['high'] - d['low']) / d['close'] * 100
    d['open_close_pct'] = (d['close'] - d['open']) / d['open'] * 100
    
    # 2. Moving Averages
    for w in [5, 10, 20, 50]:
        d[f'sma_{w}'] = d['close'].rolling(w).mean()
        d[f'ema_{w}'] = d['close'].ewm(span=w, adjust=False).mean()
        
    d['price_sma20_ratio'] = d['close'] / d['sma_20']
    d['price_ema20_ratio'] = d['close'] / d['ema_20']
    d['ema5_ema20_ratio'] = d['ema_5'] / d['ema_20']
    
    # 3. Momentum
    delta = d['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d['rsi_14'] = 100 - (100 / (1 + rs))
    
    low_14 = d['low'].rolling(14).min()
    high_14 = d['high'].rolling(14).max()
    d['stoch_k'] = (d['close'] - low_14) / (high_14 - low_14 + 1e-10) * 100
    d['stoch_d'] = d['stoch_k'].rolling(3).mean()
    d['williams_r'] = (high_14 - d['close']) / (high_14 - low_14 + 1e-10) * (-100)
    d['roc_10'] = d['close'].pct_change(10) * 100
    d['roc_20'] = d['close'].pct_change(20) * 100
    
    typical_price = (d['high'] + d['low'] + d['close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mean_dev = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    d['cci_20'] = (typical_price - sma_tp) / (0.015 * mean_dev + 1e-10)
    
    # 4. Volatility
    bb_sma = d['close'].rolling(20).mean()
    bb_std = d['close'].rolling(20).std()
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    d['bb_width'] = (bb_upper - bb_lower) / (bb_sma + 1e-10)
    d['bb_pct'] = (d['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    prev_close = d['close'].shift(1)
    tr = pd.concat([
        d['high'] - d['low'],
        (d['high'] - prev_close).abs(),
        (d['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    d['atr_14'] = tr.ewm(span=14, adjust=False).mean()
    d['atr_pct'] = d['atr_14'] / d['close'] * 100
    d['hv_20'] = d['log_return'].rolling(20).std() * np.sqrt(252)
    
    # 5. Volume Indicators
    d['volume_log'] = np.log(d['Volume'] + 1)
    d['volume_ratio'] = d['Volume'] / (d['Volume'].rolling(20).mean() + 1)
    direction = np.sign(d['close'].diff())
    d['obv'] = (direction * d['Volume']).cumsum()
    
    raw_money_flow = typical_price * d['Volume']
    pos_mf = raw_money_flow.where(d['close'] > d['close'].shift(1), 0)
    neg_mf = raw_money_flow.where(d['close'] < d['close'].shift(1), 0)
    mf_ratio = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + 1e-10)
    d['mfi_14'] = 100 - 100 / (1 + mf_ratio)
    
    # 6. MACD
    ema12 = d['close'].ewm(span=12, adjust=False).mean()
    ema26 = d['close'].ewm(span=26, adjust=False).mean()
    d['macd'] = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist'] = d['macd'] - d['macd_signal']
    d['macd_cross'] = np.sign(d['macd_hist'])
    
    # 7. Target Label: 1 if close_{t+1} > close_t else 0
    d['label'] = (d['close'].shift(-1) > d['close']).astype(int)
    
    return d

df_feat = feature_engineering(df_raw)
df_feat = df_feat.dropna().reset_index(drop=True)
print(f"Features created. Clean dataset shape: {df_feat.shape}")

FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'Volume',
    'daily_return', 'log_return', 'high_low_pct', 'open_close_pct',
    'sma_5', 'sma_10', 'sma_20', 'sma_50',
    'ema_5', 'ema_10', 'ema_20', 'ema_50',
    'price_sma20_ratio', 'price_ema20_ratio', 'ema5_ema20_ratio',
    'rsi_14', 'stoch_k', 'stoch_d', 'williams_r', 'roc_10', 'roc_20', 'cci_20',
    'bb_width', 'bb_pct', 'atr_14', 'atr_pct', 'hv_20',
    'macd', 'macd_signal', 'macd_hist', 'macd_cross',
    'volume_log', 'volume_ratio', 'obv', 'mfi_14'
]
TARGET_COL = 'label'
LOOKBACK = 30

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = df_feat.copy()
df_scaled[FEATURE_COLS] = scaler.fit_transform(df_feat[FEATURE_COLS])

# Save scaler and feature columns for Web App
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, 'feature_cols.pkl'))
print("Saved scaler.pkl and feature_cols.pkl")

# Prepare sliding sequences
X, y = [], []
feat_vals = df_scaled[FEATURE_COLS].values
target_vals = df_scaled[TARGET_COL].values

for i in range(LOOKBACK, len(feat_vals)):
    X.append(feat_vals[i - LOOKBACK:i])
    y.append(target_vals[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

N = len(X)
n_train = int(N * 0.70)
n_val = int(N * 0.15)
n_test = N - n_train - n_val

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
input_shape = (LOOKBACK, len(FEATURE_COLS))

# Build models
def build_vanilla_rnn():
    return Sequential([
        Input(shape=input_shape),
        SimpleRNN(64, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        SimpleRNN(32, return_sequences=False, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='VanillaRNN')

def build_lstm():
    return Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='LSTM')

def build_gru():
    return Sequential([
        Input(shape=input_shape),
        GRU(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        GRU(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='GRU')

def build_bilstm():
    return Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(32, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='BiLSTM')

def build_deep_lstm():
    """
    MÔ HÌNH CHÍNH: Deep Stacked LSTM với > 5 layer (7 layers: 4 LSTM + 3 Dense)
    HỢP CÁC HÀM: f_7 o f_6 o f_5 o f_4 o f_3 o f_2 o f_1 (X)
    """
    return Sequential([
        Input(shape=input_shape),
        # Layer 1: LSTM
        LSTM(256, return_sequences=True, kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.30),
        # Layer 2: LSTM
        LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.30),
        # Layer 3: LSTM
        LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.25),
        # Layer 4: LSTM
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.20),
        # Layer 5: Dense 1
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.20),
        # Layer 6: Dense 2
        Dense(32, activation='relu'),
        # Layer 7: Output Dense
        Dense(1, activation='sigmoid')
    ], name='DeepStackedLSTM')

model_builders = {
    'VanillaRNN': build_vanilla_rnn,
    'LSTM': build_lstm,
    'GRU': build_gru,
    'BiLSTM': build_bilstm,
    'DeepStackedLSTM': build_deep_lstm,
}

eval_results = {}
histories = {}
training_times = {}

EPOCHS = 35
BATCH_SIZE = 64

for name, builder in model_builders.items():
    print(f"\n==================== Training {name} ====================")
    model = builder()
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0),
    ]
    
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - t0
    
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_pred_prob))
    f1 = float(f1_score(y_test, y_pred))
    
    print(f"[{name}] Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | Time: {elapsed:.1f}s")
    
    # Save individual model
    model_path = os.path.join(MODELS_DIR, f"{name}.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    eval_results[name] = {
        'accuracy': acc,
        'auc': auc,
        'f1': f1,
        'y_pred_prob': y_pred_prob.tolist(),
        'y_pred': y_pred.tolist(),
        'train_time': elapsed
    }
    histories[name] = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'auc': [float(x) for x in history.history['auc']],
        'val_auc': [float(x) for x in history.history['val_auc']]
    }
    training_times[name] = elapsed

# Select best model (prioritizing DeepStackedLSTM or highest AUC)
best_model_name = max(eval_results, key=lambda k: eval_results[k]['auc'])
print(f"\nBest model by AUC: {best_model_name}")

# Also copy best model to models/best_model.keras
best_model = keras.models.load_model(os.path.join(MODELS_DIR, f"{best_model_name}.keras"))
best_model.save(os.path.join(MODELS_DIR, "best_model.keras"))
print(f"Saved models/best_model.keras (from {best_model_name})")

# Save DeepStackedLSTM as default deep model if needed
deep_model = keras.models.load_model(os.path.join(MODELS_DIR, "DeepStackedLSTM.keras"))
deep_model.save(os.path.join(MODELS_DIR, "deep_model.keras"))

# Metadata JSON
metadata = {
    'best_model': best_model_name,
    'feature_cols': FEATURE_COLS,
    'lookback': LOOKBACK,
    'eval_results': {
        k: {
            'accuracy': v['accuracy'],
            'auc': v['auc'],
            'f1': v['f1'],
            'train_time': v['train_time']
        } for k, v in eval_results.items()
    },
    'y_test': y_test.tolist(),
    'y_preds': {k: v['y_pred'] for k, v in eval_results.items()},
    'y_probs': {k: v['y_pred_prob'] for k, v in eval_results.items()},
    'histories': histories
}

with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print("Saved models/metadata.json")

# Export comparison plots
print("Generating comparison plots...")
plt.style.use('dark_background')

# 1. Bar Chart Metrics Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#0d1117')
metrics = ['accuracy', 'auc', 'f1']
titles = ['Accuracy', 'ROC-AUC', 'F1-Score']
colors = ['#58a6ff', '#3fb950', '#d29922']

for idx, (metric, title, col) in enumerate(zip(metrics, titles, colors)):
    ax = axes[idx]
    ax.set_facecolor('#161b22')
    names = list(eval_results.keys())
    values = [eval_results[n][metric] for n in names]
    bars = ax.bar(names, values, color=col, alpha=0.85, edgecolor='none')
    ax.set_title(f'Comparison: {title}', fontsize=12, fontweight='bold', color='white')
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=9, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'compare_algorithms.png'), dpi=150, facecolor='#0d1117')
plt.close()
print("Saved compare_algorithms.png")

# 2. ROC Curves Comparison
plt.figure(figsize=(8, 6), facecolor='#0d1117')
plt.gca().set_facecolor('#161b22')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random (AUC = 0.50)')

curve_colors = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff']
for (name, col) in zip(eval_results.keys(), curve_colors):
    probs = np.array(eval_results[name]['y_pred_prob'])
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC = {eval_results[name]['auc']:.3f})")

plt.xlabel('False Positive Rate', color='white')
plt.ylabel('True Positive Rate', color='white')
plt.title('ROC Curves - So Sánh Các Thuật Toán RNN', fontsize=14, color='white', fontweight='bold')
plt.legend(loc='lower right', facecolor='#161b22', edgecolor='gray')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'roc_comparison.png'), dpi=150, facecolor='#0d1117')
plt.close()
print("Saved roc_comparison.png")

# Also save sample last 60 records for Web App fast simulation/real-time testing
df_sample = df_raw.tail(120).copy()
df_sample.to_csv(os.path.join(MODELS_DIR, 'sample_recent_data.csv'), index=False)
print("Saved sample_recent_data.csv")

print("All models trained and artifacts generated successfully!")
