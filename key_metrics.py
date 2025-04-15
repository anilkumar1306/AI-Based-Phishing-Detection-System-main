import torch

# Load and print keys for best_cnn.pth
cnn_checkpoint = torch.load('best_cnn.pth')
print("CNN Model keys:", cnn_checkpoint.keys())

# Load and print keys for best_lstm.pth
lstm_checkpoint = torch.load('best_lstm.pth')
print("LSTM Model keys:", lstm_checkpoint.keys())

# Load and print keys for best_rnn.pth
rnn_checkpoint = torch.load('best_rnn.pth')
print("RNN Model keys:", rnn_checkpoint.keys())

