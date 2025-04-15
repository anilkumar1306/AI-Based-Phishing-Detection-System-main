from flask import Flask, request, jsonify, render_template
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import base64

app = Flask(__name__)

# Load models and tokenizer
cnn_model = tf.keras.models.load_model('best_cnn.keras')
lstm_model = tf.keras.models.load_model('best_lstm.keras')
rnn_model = tf.keras.models.load_model('best_rnn.keras')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# VirusTotal API key
VIRUSTOTAL_API_KEY = 'ee6e9534c372447877b0c5cf1c3983fc54970250e3815403baf5db606aa53576'
max_length = 200  # Match your model's training sequence length
threshold = 0.5   # Classification threshold

def preprocess_text(text):
    """Clean email text before tokenization"""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)          # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)      # Remove special chars/numbers
    return text.strip()

def check_virustotal_links(email_text):
    """Check links in email against VirusTotal"""
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
                    return "Malicious/Phishing Link Detected"
    return "No Malicious Links Found"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']
    virustotal_status = check_virustotal_links(email_text)
    
    # If malicious links found, override models
    if "Malicious" in virustotal_status:
        return jsonify({
            "VirusTotal": virustotal_status,
            "CNN": {"label": "Phishing", "score": 1.0},
            "LSTM": {"label": "Phishing", "score": 1.0},
            "RNN": {"label": "Phishing", "score": 1.0},
            "Ensemble": {"label": "Phishing", "score": 1.0},
            "FinalVerdict": "Phishing"
        })

    # Preprocess and predict
    processed_text = preprocess_text(email_text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Get model predictions
    cnn_pred = cnn_model.predict(padded_sequence, verbose=0)[0][0]
    lstm_pred = lstm_model.predict(padded_sequence, verbose=0)[0][0]
    rnn_pred = rnn_model.predict(padded_sequence, verbose=0)[0][0]

    # Calculate weighted ensemble
    weights = [0.33, 0.33, 0.34]
    ensemble_score = (weights[0] * cnn_pred + 
                     weights[1] * lstm_pred + 
                     weights[2] * rnn_pred)
    
    # Generate results
    results = {
        "VirusTotal": virustotal_status,
        "CNN": {"label": "Phishing" if cnn_pred > threshold else "Legitimate",
                "score": float(cnn_pred)},
        "LSTM": {"label": "Phishing" if lstm_pred > threshold else "Legitimate",
                 "score": float(lstm_pred)},
        "RNN": {"label": "Phishing" if rnn_pred > threshold else "Legitimate",
                "score": float(rnn_pred)},
        "Ensemble": {"label": "Phishing" if ensemble_score > threshold else "Legitimate",
                    "score": float(ensemble_score)},
        "FinalVerdict": "Phishing" if ensemble_score > threshold else "Legitimate"
    }

    return jsonify(results)

# ... keep your existing about/team routes ...

if __name__ == '__main__':
    app.run(debug=True)