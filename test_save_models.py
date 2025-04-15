import unittest
import torch
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from save_models import CNNModel, LSTMModel, RNNModel, EmailDataset, tokenizer, X_val, y_val

# FILE: test_save_models.py


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load models
        cls.cnn_model = CNNModel(len(tokenizer.word_index) + 1, 128)
        cls.cnn_model.load_state_dict(torch.load('best_CNNModel.pth'))
        cls.cnn_model.eval()

        cls.lstm_model = LSTMModel(len(tokenizer.word_index) + 1, 128, 64)
        cls.lstm_model.load_state_dict(torch.load('best_LSTMModel.pth'))
        cls.lstm_model.eval()

        cls.rnn_model = RNNModel(len(tokenizer.word_index) + 1, 128, 64)
        cls.rnn_model.load_state_dict(torch.load('best_RNNModel.pth'))
        cls.rnn_model.eval()

        # Load tokenizer
        with open('tokenizer.pkl', 'rb') as tokenizer_file:
            cls.tokenizer = pickle.load(tokenizer_file)

        # Prepare validation data
        cls.val_loader = DataLoader(EmailDataset(X_val, y_val), batch_size=32, shuffle=False)

    def evaluate_model(self, model):
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for texts, labels in self.val_loader:
                outputs = model(texts).squeeze()
                preds = (outputs > 0.5).float()
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        return accuracy, precision, recall, f1

    def test_cnn_model(self):
        accuracy, precision, recall, f1 = self.evaluate_model(self.cnn_model)
        print(f"CNN Model - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

    def test_lstm_model(self):
        accuracy, precision, recall, f1 = self.evaluate_model(self.lstm_model)
        print(f"LSTM Model - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

    def test_rnn_model(self):
        accuracy, precision, recall, f1 = self.evaluate_model(self.rnn_model)
        print(f"RNN Model - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

if __name__ == '__main__':
    unittest.main()