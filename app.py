import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU to avoid CUDA errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Keep for compatibility

import numpy as np
import pickle
import re
import requests
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load tokenizer
tokenizer_path = 'tokenizer.pkl'
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer file {tokenizer_path} not found!")

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_len = 200

print(f"[INFO] Tokenizer loaded successfully. Vocabulary size: {vocab_size}")

# Load Keras models
try:
    cnn_model = tf.keras.models.load_model('best_cnn.keras')
    lstm_model = tf.keras.models.load_model('best_lstm.keras')
    rnn_model = tf.keras.models.load_model('best_rnn.keras')
    print("[INFO] Keras models loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load Keras models: {e}")

# VirusTotal API configuration
VT_API_KEY = 'ee6e9534c372447877b0c5cf1c3983fc54970250e3815403baf5db606aa53576'
headers = {'x-apikey': VT_API_KEY}

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Check URLs using VirusTotal
def check_virustotal(content):
    urls = re.findall(r'https?://\S+', content)
    results = {}
    for url in urls:
        url_id = requests.utils.quote(url, safe='')
        response = requests.get(
            f'https://www.virustotal.com/api/v3/urls/{url_id}',
            headers=headers
        )
        if response.status_code == 200:
            results[url] = response.json().get('data', {}).get('attributes', {}).get('last_analysis_stats')
        else:
            results[url] = f"Error: {response.status_code}"
    return results

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in request.form:
        return jsonify({'error': 'No email provided'}), 400

    email_content = request.form['email']
    processed_text = preprocess_text(email_content)

    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

    # Make predictions
    cnn_pred = cnn_model.predict(padded, verbose=0)[0][0]
    lstm_pred = lstm_model.predict(padded, verbose=0)[0][0]
    rnn_pred = rnn_model.predict(padded, verbose=0)[0][0]

    # Compute ensemble score
    weights = [0.33, 0.33, 0.34]
    ensemble_score = weights[0] * cnn_pred + weights[1] * lstm_pred + weights[2] * rnn_pred
    threshold = 0.5

    # Prepare predictions
    predictions = {
        'cnn': {
            'prediction': 'phishing' if cnn_pred > threshold else 'legitimate',
            'confidence': round(float(cnn_pred), 4)
        },
        'lstm': {
            'prediction': 'phishing' if lstm_pred > threshold else 'legitimate',
            'confidence': round(float(lstm_pred), 4)
        },
        'rnn': {
            'prediction': 'phishing' if rnn_pred > threshold else 'legitimate',
            'confidence': round(float(rnn_pred), 4)
        },
        'ensemble': {
            'prediction': 'phishing' if ensemble_score > threshold else 'legitimate',
            'confidence': round(float(ensemble_score), 4)
        }
    }

    # Get VirusTotal results
    vt_results = check_virustotal(email_content)

    return jsonify({
        'model_predictions': predictions,
        'virustotal_results': vt_results
    })

if __name__ == '__main__':
    app.run(debug=True)  # Disable reloader to avoid TensorFlow conflicts