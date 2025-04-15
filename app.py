from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import tensorflow as tf
import re
import requests
import base64

# Check for CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Define maximum length for inputs
max_length = 1000  # Consistent with the original Flask app

# VirusTotal API key
VIRUSTOTAL_API_KEY = 'ee6e9534c372447877b0c5cf1c3983fc54970250e3815403baf5db606aa53576'  # Replace with your VirusTotal API key

# Define CNN Model using PyTorch
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(100 * (max_length // 2 - 1), 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Change to (batch_size, embedding_dim, seq_length)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        output_size = x.size(1)
        if output_size != self.fc.in_features:
            self.fc = nn.Linear(output_size, 1)
        x = torch.sigmoid(self.fc(x))
        return x

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = torch.sigmoid(self.fc(lstm_out))
        return x

# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]
        x = torch.sigmoid(self.fc(rnn_out))
        return x

# Load and prepare the models
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
hidden_dim = 64

cnn_model = CNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
cnn_model.load_state_dict(torch.load('cnn_model.keras', map_location=device, weights_only=True))
cnn_model.eval()

lstm_model = LSTMModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
lstm_model.load_state_dict(torch.load('lstm_model.keras', map_location=device, weights_only=True))
lstm_model.eval()

rnn_model = RNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
rnn_model.load_state_dict(torch.load('rnn_model.keras', map_location=device, weights_only=True))
rnn_model.eval()

# Updated prediction function adapted for PyTorch
def predict_email(email_text, cnn_model, lstm_model, rnn_model, tokenizer, max_len=1000, threshold=0.5):
    # Preprocess the input
    sequence = tokenizer.texts_to_sequences([email_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len, padding='post')
    email_tensor = torch.tensor(padded_sequence, dtype=torch.long).to(device)

    # Get predictions from each model
    with torch.no_grad():
        cnn_pred = cnn_model(email_tensor).item()
        lstm_pred = lstm_model(email_tensor).item()
        rnn_pred = rnn_model(email_tensor).item()

    # Compute ensemble score with weights
    weights = [0.33, 0.33, 0.34]  # Adjustable based on model performance
    ensemble_score = sum(w * p for w, p in zip(weights, [cnn_pred, lstm_pred, rnn_pred]))

    # Determine labels based on threshold
    cnn_label = "Phishing" if cnn_pred > threshold else "Legitimate"
    lstm_label = "Phishing" if lstm_pred > threshold else "Legitimate"
    rnn_label = "Phishing" if rnn_pred > threshold else "Legitimate"
    ensemble_label = "Phishing" if ensemble_score > threshold else "Legitimate"

    return {
        "CNN": {"label": cnn_label, "score": cnn_pred},
        "LSTM": {"label": lstm_label, "score": lstm_pred},
        "RNN": {"label": rnn_label, "score": rnn_pred},
        "Ensemble": {"label": ensemble_label, "score": ensemble_score}
    }

# VirusTotal check function
def check_virustotal_links(email_text):
    links = re.findall(r'(https?://\S+)', email_text)
    for link in links:
        encoded_link = base64.urlsafe_b64encode(link.encode()).decode().strip('=')
        url = f"https://www.virustotal.com/api/v3/urls/{encoded_link}"
        headers = {'x-apikey': VIRUSTOTAL_API_KEY}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            analysis_results = data.get("data", {}).get("attributes", {}).get("last_analysis_results", {})
            for result in analysis_results.values():
                if result.get("category") in ["malicious", "phishing"]:
                    return "VirusTotal: Malicious/Phishing"
    return "VirusTotal: Clean"

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']

    # Check VirusTotal
    virustotal_result = check_virustotal_links(email_text)

    # If VirusTotal flags as malicious, override predictions
    if virustotal_result == "VirusTotal: Malicious/Phishing":
        final_verdict = "Phishing"
        results = {
            "VirusTotal": virustotal_result,
            "CNN": {"label": "Phishing", "score": 1.0},
            "LSTM": {"label": "Phishing", "score": 1.0},
            "RNN": {"label": "Phishing", "score": 1.0},
            "Ensemble": {"label": "Phishing", "score": 1.0},
            "Final Verdict": final_verdict
        }
    else:
        # Get model predictions and ensemble result
        prediction_results = predict_email(email_text, cnn_model, lstm_model, rnn_model, tokenizer, max_len=max_length, threshold=0.5)
        final_verdict = prediction_results["Ensemble"]["label"]
        results = {
            "VirusTotal": virustotal_result,
            **prediction_results,
            "Final Verdict": final_verdict
        }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)