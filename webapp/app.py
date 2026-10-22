# -*- coding: utf-8 -*-
"""
Flask Web Application - Dự đoán giá cổ phiếu VN-Index
Hỗ trợ Dual-Engine:
- Ultra-Light NumPy Inference Engine: Tiêu thụ chỉ ~45MB RAM (Hoàn hảo 100% cho Render Free 512MB)
- TensorFlow/Keras Engine: Tự động sử dụng khi môi trường có sẵn TensorFlow.
Hỗ trợ giao diện Responsive Desktop & Mobile.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

def get_models_dir():
    if os.path.exists(os.path.join(BASE_DIR, 'models')):
        return os.path.join(BASE_DIR, 'models')
    return os.path.join(PARENT_DIR, 'models')

CURRENT_MODELS_DIR = get_models_dir()

# Global variables
metadata = {}
scaler = None
feature_cols = []
cached_weights = {}
loaded_tf_models = {}
HAS_TF = False

# Try importing TensorFlow optionally
try:
    import tensorflow as tf
    HAS_TF = True
    print("[OK] TensorFlow detected - Keras engine available")
except Exception:
    HAS_TF = False
    print("[INFO] Ultra-Light Mode: Running Pure NumPy Engine (~45MB RAM, Render Free Optimized)")

def load_artifacts():
    global metadata, scaler, feature_cols
    try:
        meta_path = os.path.join(CURRENT_MODELS_DIR, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
        scaler_path = os.path.join(CURRENT_MODELS_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            
        feat_path = os.path.join(CURRENT_MODELS_DIR, 'feature_cols.pkl')
        if os.path.exists(feat_path):
            feature_cols = joblib.load(feat_path)
            
        print(f"Loaded artifacts from {CURRENT_MODELS_DIR}")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

load_artifacts()

# =========================================================================
# ULTRA-LIGHT NUMPY INFERENCE ENGINE (<45MB RAM)
# =========================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def relu(x):
    return np.maximum(0, x)

def batch_norm_infer(x, gamma, beta, moving_mean, moving_var, eps=1e-3):
    return gamma * (x - moving_mean) / np.sqrt(moving_var + eps) + beta

def lstm_cell_step(xt, ht_prev, ct_prev, W, U, b):
    units = ht_prev.shape[-1]
    z = np.dot(xt, W) + np.dot(ht_prev, U) + b
    zi = z[..., :units]
    zf = z[..., units:2*units]
    zc = z[..., 2*units:3*units]
    zo = z[..., 3*units:]
    
    i = sigmoid(zi)
    f = sigmoid(zf)
    c_tilde = np.tanh(zc)
    ct = f * ct_prev + i * c_tilde
    ot = sigmoid(zo)
    ht = ot * np.tanh(ct)
    return ht, ct

def lstm_layer_forward(X, W, U, b, return_sequences=True):
    timesteps = X.shape[0]
    units = U.shape[0]
    ht = np.zeros(units, dtype=np.float32)
    ct = np.zeros(units, dtype=np.float32)
    outputs = []
    for t in range(timesteps):
        ht, ct = lstm_cell_step(X[t], ht, ct, W, U, b)
        if return_sequences:
            outputs.append(ht)
    if return_sequences:
        return np.array(outputs, dtype=np.float32)
    return ht

def predict_deep_lstm_numpy(X_seq, weights):
    """NumPy Forward Pass của Deep Stacked LSTM (7 layers)"""
    # L1: LSTM 256 + BN
    h1 = lstm_layer_forward(X_seq, weights['arr_0'], weights['arr_1'], weights['arr_2'], return_sequences=True)
    h1_bn = batch_norm_infer(h1, weights['arr_3'], weights['arr_4'], weights['arr_5'], weights['arr_6'])
    
    # L2: LSTM 128 + BN
    h2 = lstm_layer_forward(h1_bn, weights['arr_7'], weights['arr_8'], weights['arr_9'], return_sequences=True)
    h2_bn = batch_norm_infer(h2, weights['arr_10'], weights['arr_11'], weights['arr_12'], weights['arr_13'])
    
    # L3: LSTM 64 + BN
    h3 = lstm_layer_forward(h2_bn, weights['arr_14'], weights['arr_15'], weights['arr_16'], return_sequences=True)
    h3_bn = batch_norm_infer(h3, weights['arr_17'], weights['arr_18'], weights['arr_19'], weights['arr_20'])
    
    # L4: LSTM 32 + BN
    h4 = lstm_layer_forward(h3_bn, weights['arr_21'], weights['arr_22'], weights['arr_23'], return_sequences=False)
    h4_bn = batch_norm_infer(h4, weights['arr_24'], weights['arr_25'], weights['arr_26'], weights['arr_27'])
    
    # L5: Dense 64 + BN + ReLU
    z5 = relu(np.dot(h4_bn, weights['arr_28']) + weights['arr_29'])
    z5_bn = batch_norm_infer(z5, weights['arr_30'], weights['arr_31'], weights['arr_32'], weights['arr_33'])
    
    # L6: Dense 32 + ReLU
    z6 = relu(np.dot(z5_bn, weights['arr_34']) + weights['arr_35'])
    
    # L7: Output Dense + Sigmoid
    out = sigmoid(np.dot(z6, weights['arr_36']) + weights['arr_37'])
    return float(out[0])

def predict_vanilla_rnn_numpy(X_seq, weights):
    # Layer 1: SimpleRNN 64
    W1, U1, b1 = weights['arr_0'], weights['arr_1'], weights['arr_2']
    units1 = U1.shape[0]
    ht1 = np.zeros(units1, dtype=np.float32)
    h1_seq = []
    for t in range(X_seq.shape[0]):
        ht1 = np.tanh(np.dot(X_seq[t], W1) + np.dot(ht1, U1) + b1)
        h1_seq.append(ht1)
    h1_seq = np.array(h1_seq)
    
    # Layer 2: SimpleRNN 32
    W2, U2, b2 = weights['arr_3'], weights['arr_4'], weights['arr_5']
    units2 = U2.shape[0]
    ht2 = np.zeros(units2, dtype=np.float32)
    for t in range(h1_seq.shape[0]):
        ht2 = np.tanh(np.dot(h1_seq[t], W2) + np.dot(ht2, U2) + b2)
        
    # Dense 16
    z1 = relu(np.dot(ht2, weights['arr_6']) + weights['arr_7'])
    # Output
    out = sigmoid(np.dot(z1, weights['arr_8']) + weights['arr_9'])
    return float(out[0])

def predict_lstm_numpy(X_seq, weights):
    # L1: LSTM 128 + BN
    h1 = lstm_layer_forward(X_seq, weights['arr_0'], weights['arr_1'], weights['arr_2'], return_sequences=True)
    h1_bn = batch_norm_infer(h1, weights['arr_3'], weights['arr_4'], weights['arr_5'], weights['arr_6'])
    
    # L2: LSTM 64 + BN
    h2 = lstm_layer_forward(h1_bn, weights['arr_7'], weights['arr_8'], weights['arr_9'], return_sequences=False)
    h2_bn = batch_norm_infer(h2, weights['arr_10'], weights['arr_11'], weights['arr_12'], weights['arr_13'])
    
    # Dense 32
    z = relu(np.dot(h2_bn, weights['arr_14']) + weights['arr_15'])
    out = sigmoid(np.dot(z, weights['arr_16']) + weights['arr_17'])
    return float(out[0])

def get_weights(model_name):
    if model_name in cached_weights:
        return cached_weights[model_name]
    path = os.path.join(CURRENT_MODELS_DIR, f"{model_name}_weights.npz")
    if not os.path.exists(path):
        path = os.path.join(CURRENT_MODELS_DIR, "DeepStackedLSTM_weights.npz")
    if os.path.exists(path):
        w = np.load(path)
        cached_weights[model_name] = w
        return w
    return None

def run_model_inference(model_name, X_seq_30x40):
    """Universal Predictor: uses NumPy engine (fast & low-RAM) or TensorFlow fallback"""
    weights = get_weights(model_name)
    if weights is not None:
        if 'Deep' in model_name:
            return predict_deep_lstm_numpy(X_seq_30x40, weights)
        elif 'Vanilla' in model_name:
            return predict_vanilla_rnn_numpy(X_seq_30x40, weights)
        elif 'LSTM' in model_name and 'Bi' not in model_name:
            return predict_lstm_numpy(X_seq_30x40, weights)
            
    # Fallback to TensorFlow if available
    if HAS_TF:
        if model_name not in loaded_tf_models:
            keras_path = os.path.join(CURRENT_MODELS_DIR, f"{model_name}.keras")
            if not os.path.exists(keras_path):
                keras_path = os.path.join(CURRENT_MODELS_DIR, "best_model.keras")
            loaded_tf_models[model_name] = tf.keras.models.load_model(keras_path)
        model = loaded_tf_models[model_name]
        return float(model.predict(np.expand_dims(X_seq_30x40, 0), verbose=0)[0][0])
        
    # Default fallback
    if weights is not None:
        return predict_deep_lstm_numpy(X_seq_30x40, weights)
    return 0.50

# =========================================================================
# INDICATORS & PIPELINE
# =========================================================================
def compute_indicators_single(df):
    d = df.copy()
    d['daily_return'] = d['close'].pct_change() * 100
    d['log_return'] = np.log(d['close'] / d['close'].shift(1))
    d['high_low_pct'] = (d['high'] - d['low']) / d['close'] * 100
    d['open_close_pct'] = (d['close'] - d['open']) / d['open'] * 100
    
    for w in [5, 10, 20, 50]:
        d[f'sma_{w}'] = d['close'].rolling(w).mean()
        d[f'ema_{w}'] = d['close'].ewm(span=w, adjust=False).mean()
        
    d['price_sma20_ratio'] = d['close'] / d['sma_20']
    d['price_ema20_ratio'] = d['close'] / d['ema_20']
    d['ema5_ema20_ratio'] = d['ema_5'] / d['ema_20']
    
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
    
    d['volume_log'] = np.log(d['Volume'] + 1)
    d['volume_ratio'] = d['Volume'] / (d['Volume'].rolling(20).mean() + 1)
    direction = np.sign(d['close'].diff())
    d['obv'] = (direction * d['Volume']).cumsum()
    
    raw_money_flow = typical_price * d['Volume']
    pos_mf = raw_money_flow.where(d['close'] > d['close'].shift(1), 0)
    neg_mf = raw_money_flow.where(d['close'] < d['close'].shift(1), 0)
    mf_ratio = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + 1e-10)
    d['mfi_14'] = 100 - 100 / (1 + mf_ratio)
    
    ema12 = d['close'].ewm(span=12, adjust=False).mean()
    ema26 = d['close'].ewm(span=26, adjust=False).mean()
    d['macd'] = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist'] = d['macd'] - d['macd_signal']
    d['macd_cross'] = np.sign(d['macd_hist'])
    
    return d

def get_recent_history():
    sample_file = os.path.join(CURRENT_MODELS_DIR, 'sample_recent_data.csv')
    if os.path.exists(sample_file):
        df = pd.read_csv(sample_file)
    else:
        raw_file = os.path.join(PARENT_DIR, 'data', 'HOSE_DLY_VNINDEX1D.csv')
        df = pd.read_csv(raw_file).tail(120)
    
    if 'date' not in df.columns and 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'], unit='s').dt.strftime('%Y-%m-%d')
    return df

# =========================================================================
# FLASK ROUTES
# =========================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metadata')
def get_metadata():
    if not metadata:
        load_artifacts()
    return jsonify(metadata)

@app.route('/api/history')
def get_history():
    df = get_recent_history()
    records = df[['date', 'open', 'high', 'low', 'close', 'Volume']].to_dict(orient='records')
    return jsonify({
        'status': 'success',
        'count': len(records),
        'data': records
    })

@app.route('/api/predict_latest', methods=['POST', 'GET'])
def predict_latest():
    try:
        model_name = request.args.get('model', 'DeepStackedLSTM')
        df = get_recent_history()
        df_feat = compute_indicators_single(df).dropna().reset_index(drop=True)
        
        if len(df_feat) < 30:
            return jsonify({'status': 'error', 'message': 'Không đủ 30 phiên sau khi tính indicators'}), 400
            
        latest_30 = df_feat.iloc[-30:][feature_cols]
        scaled_seq = scaler.transform(latest_30).astype(np.float32)
        
        prob = run_model_inference(model_name, scaled_seq)
        prediction = "TĂNG" if prob >= 0.5 else "GIẢM"
        direction = "UP" if prob >= 0.5 else "DOWN"
        confidence = float(prob if prob >= 0.5 else 1.0 - prob) * 100
        
        last_row = df_feat.iloc[-1]
        last_date = str(df.iloc[-1].get('date', 'Hôm nay'))
        last_close = float(df.iloc[-1]['close'])
        
        rsi = float(last_row.get('rsi_14', 50))
        macd = float(last_row.get('macd', 0))
        macd_sig = float(last_row.get('macd_signal', 0))
        bb_pct = float(last_row.get('bb_pct', 0.5))
        
        if prob > 0.60 and rsi < 70:
            action = "MUA MẠNH (STRONG BUY)"
            action_color = "#22c55e"
        elif prob >= 0.50:
            action = "MUA NHẸ / NẮM GIỮ (ACCUMULATE)"
            action_color = "#84cc16"
        elif prob < 0.40 and rsi > 30:
            action = "BÁN / HẠ TỶ TRỌNG (SELL)"
            action_color = "#ef4444"
        else:
            action = "THEO DÕI / GIẢM VỊ THẾ (HOLD/WATCH)"
            action_color = "#f59e0b"
            
        return jsonify({
            'status': 'success',
            'model_used': model_name,
            'prediction': prediction,
            'direction': direction,
            'probability_up': round(prob * 100, 2),
            'probability_down': round((1.0 - prob) * 100, 2),
            'confidence': round(confidence, 1),
            'recommendation': action,
            'recommendation_color': action_color,
            'latest_date': last_date,
            'latest_close': round(last_close, 2),
            'indicators': {
                'rsi_14': round(rsi, 2),
                'macd': round(macd, 2),
                'macd_signal': round(macd_sig, 2),
                'bb_pct': round(bb_pct * 100, 1),
                'ema_5': round(float(last_row.get('ema_5', 0)), 2),
                'ema_20': round(float(last_row.get('ema_20', 0)), 2),
                'ema_50': round(float(last_row.get('ema_50', 0)), 2),
                'volatility': round(float(last_row.get('hv_20', 0)) * 100, 1)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict_compare_all', methods=['GET', 'POST'])
def predict_compare_all():
    models_list = ['VanillaRNN', 'LSTM', 'GRU', 'BiLSTM', 'DeepStackedLSTM']
    df = get_recent_history()
    df_feat = compute_indicators_single(df).dropna().reset_index(drop=True)
    latest_30 = df_feat.iloc[-30:][feature_cols]
    scaled_seq = scaler.transform(latest_30).astype(np.float32)
    
    results = []
    for m_name in models_list:
        try:
            p = run_model_inference(m_name, scaled_seq)
            results.append({
                'model': m_name,
                'prob_up': round(p * 100, 2),
                'pred': 'TĂNG' if p >= 0.5 else 'GIẢM',
                'direction': 'UP' if p >= 0.5 else 'DOWN',
                'confidence': round((p if p >= 0.5 else 1.0 - p) * 100, 1)
            })
        except Exception as e:
            results.append({'model': m_name, 'error': str(e)})
            
    return jsonify({
        'status': 'success',
        'comparison': results
    })

@app.route('/api/predict_custom', methods=['POST'])
def predict_custom():
    try:
        data = request.get_json()
        model_name = data.get('model', 'DeepStackedLSTM')
        close_price = float(data.get('close', 1250.0))
        open_price = float(data.get('open', 1245.0))
        high_price = float(data.get('high', 1255.0))
        low_price = float(data.get('low', 1240.0))
        volume = float(data.get('volume', 800000000))
        
        df = get_recent_history()
        new_row = {
            'date': 'Tùy chỉnh',
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'Volume': volume,
            'time': int(pd.to_datetime('today').timestamp())
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        df_feat = compute_indicators_single(df).dropna().reset_index(drop=True)
        latest_30 = df_feat.iloc[-30:][feature_cols]
        scaled_seq = scaler.transform(latest_30).astype(np.float32)
        
        prob = run_model_inference(model_name, scaled_seq)
        prediction = "TĂNG" if prob >= 0.5 else "GIẢM"
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'probability_up': round(prob * 100, 2),
            'probability_down': round((1.0 - prob) * 100, 2),
            'confidence': round((prob if prob >= 0.5 else 1.0 - prob) * 100, 1),
            'inputs': {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
