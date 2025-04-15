import os
import torch
import numpy as np
import pickle
import re
import requests
from torch import nn
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Add this import

# Disable OneDNN optimization (for TensorFlow compatibility)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Load tokenizer
tokenizer_path = 'tokenizer.pkl'
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer file {tokenizer_path} not found!")

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_len = 100  # Manually set max length if not available in pickle

print(f"[INFO] Tokenizer loaded successfully. Vocabulary size: {vocab_size}")

# Define model architectures (should match training setup)
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        conved = [nn.functional.relu(conv(x)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        return self.fc(torch.cat(pooled, dim=1))

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.lstm.bidirectional else hidden[-1]
        return self.fc(hidden)

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.rnn.bidirectional else hidden[-1]
        return self.fc(hidden)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters (should match training settings)
embed_dim = 100
num_filters = 100
filter_sizes = [3, 4, 5]
hidden_dim = 128
bidirectional = True

models = {
    'cnn': TextCNN(vocab_size, embed_dim, num_filters, filter_sizes).to(device),
    'lstm': TextLSTM(vocab_size, embed_dim, hidden_dim, bidirectional).to(device),
    'rnn': TextRNN(vocab_size, embed_dim, hidden_dim, bidirectional).to(device)
}

# Load trained weights with error handling
for model_name in models:
    model_path = f'best_{model_name}.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        try:
            models[model_name].load_state_dict(checkpoint['model_state_dict'])
            models[model_name].eval()
            print(f"[INFO] Model '{model_name}' loaded successfully.")
        except RuntimeError as e:
            print(f"[ERROR] Could not load model '{model_name}'. Mismatch in state_dict: {e}")
    else:
        print(f"[WARNING] Model checkpoint '{model_path}' not found!")

# VirusTotal API configuration
VT_API_KEY = 'ee6e9534c372447877b0c5cf1c3983fc54970250e3815403baf5db606aa53576'
headers = {'x-apikey': VT_API_KEY}

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
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
    tensor = torch.LongTensor(padded).to(device)

    # Get predictions
    predictions = {}
    with torch.no_grad():
        for model_name, model in models.items():
            output = model(tensor).squeeze()
            prob = torch.sigmoid(output).item()
            predictions[model_name] = {
                'prediction': 'phishing' if prob > 0.5 else 'legitimate',
                'confidence': round(prob, 4)
            }

    # Get VirusTotal results
    vt_results = check_virustotal(email_content)

    return jsonify({
        'model_predictions': predictions,
        'virustotal_results': vt_results
    })

if __name__ == '__main__':
    app.run(debug=True)
