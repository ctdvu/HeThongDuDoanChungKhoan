import numpy as np
import tensorflow as tf

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def relu(x):
    return np.maximum(0, x)

def batch_norm_infer(x, gamma, beta, moving_mean, moving_var, eps=1e-3):
    return gamma * (x - moving_mean) / np.sqrt(moving_var + eps) + beta

def lstm_cell_step(xt, ht_prev, ct_prev, W, U, b):
    # Keras gates order: i, f, c, o
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
    # X shape: (timesteps, features)
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

def deep_stacked_lstm_numpy_forward(X_seq, weights):
    # Layer 1: LSTM 256 + BN
    W1, U1, b1 = weights['arr_0'], weights['arr_1'], weights['arr_2']
    h1 = lstm_layer_forward(X_seq, W1, U1, b1, return_sequences=True)
    # BN 1
    g1, b_b1, m1, v1 = weights['arr_3'], weights['arr_4'], weights['arr_5'], weights['arr_6']
    h1_bn = batch_norm_infer(h1, g1, b_b1, m1, v1)
    
    # Layer 2: LSTM 128 + BN
    W2, U2, b2 = weights['arr_7'], weights['arr_8'], weights['arr_9']
    h2 = lstm_layer_forward(h1_bn, W2, U2, b2, return_sequences=True)
    g2, b_b2, m2, v2 = weights['arr_10'], weights['arr_11'], weights['arr_12'], weights['arr_13']
    h2_bn = batch_norm_infer(h2, g2, b_b2, m2, v2)
    
    # Layer 3: LSTM 64 + BN
    W3, U3, b3 = weights['arr_14'], weights['arr_15'], weights['arr_16']
    h3 = lstm_layer_forward(h2_bn, W3, U3, b3, return_sequences=True)
    g3, b_b3, m3, v3 = weights['arr_17'], weights['arr_18'], weights['arr_19'], weights['arr_20']
    h3_bn = batch_norm_infer(h3, g3, b_b3, m3, v3)
    
    # Layer 4: LSTM 32 + BN (return_sequences=False)
    W4, U4, b4 = weights['arr_21'], weights['arr_22'], weights['arr_23']
    h4 = lstm_layer_forward(h3_bn, W4, U4, b4, return_sequences=False)
    g4, b_b4, m4, v4 = weights['arr_24'], weights['arr_25'], weights['arr_26'], weights['arr_27']
    h4_bn = batch_norm_infer(h4, g4, b_b4, m4, v4)
    
    # Layer 5: Dense 64 + BN + ReLU
    W5, b5 = weights['arr_28'], weights['arr_29']
    z5 = relu(np.dot(h4_bn, W5) + b5)
    g5, b_b5, m5, v5 = weights['arr_30'], weights['arr_31'], weights['arr_32'], weights['arr_33']
    z5_bn = batch_norm_infer(z5, g5, b_b5, m5, v5)
    
    # Layer 6: Dense 32 + ReLU
    W6, b6 = weights['arr_34'], weights['arr_35']
    z6 = relu(np.dot(z5_bn, W6) + b6)
    
    # Layer 7: Dense 1 + Sigmoid
    W7, b7 = weights['arr_36'], weights['arr_37']
    out = sigmoid(np.dot(z6, W7) + b7)
    return float(out[0])

# Test equivalence
keras_model = tf.keras.models.load_model('D:/N4_K1/HTTM/Code/TieuLuan/RNN_CoPhieu/models/DeepStackedLSTM.keras')
weights_npz = np.load('D:/N4_K1/HTTM/Code/TieuLuan/RNN_CoPhieu/models/deep_weights.npz')

# Create random input (30, 40)
np.random.seed(42)
sample_X = np.random.uniform(0, 1, size=(30, 40)).astype(np.float32)

keras_pred = float(keras_model.predict(np.expand_dims(sample_X, 0), verbose=0)[0][0])
numpy_pred = deep_stacked_lstm_numpy_forward(sample_X, weights_npz)

print(f'Keras Prediction: {keras_pred:.6f}')
print(f'NumPy Prediction: {numpy_pred:.6f}')
print(f'Difference:       {abs(keras_pred - numpy_pred):.8e}')
assert abs(keras_pred - numpy_pred) < 1e-4, 'Mismatch!'
print('EXACT MATCH CONFIRMED!')
