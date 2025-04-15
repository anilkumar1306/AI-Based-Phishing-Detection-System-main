import tensorflow as tf
print(tf.__version__)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, LSTM, SimpleRNN, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Upgraded CNN Model (unchanged)
def create_cnn_model(vocab_size, max_len):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        Dropout(0.3),
        Conv1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

# Upgraded LSTM Model (unchanged)
def create_lstm_model(vocab_size, max_len):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))),
        BatchNormalization(),
        Bidirectional(LSTM(64, kernel_regularizer=l2(0.01))),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

# Upgraded RNN Model (unchanged)
def create_rnn_model(vocab_size, max_len):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        Dropout(0.3),
        Bidirectional(SimpleRNN(128, return_sequences=True, kernel_regularizer=l2(0.01))),
        BatchNormalization(),
        Bidirectional(SimpleRNN(64, kernel_regularizer=l2(0.01))),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

# Model parameters
vocab_size = 10000
max_len = 200
batch_size = 64
epochs = 20

# Create models
cnn_model = create_cnn_model(vocab_size, max_len)
lstm_model = create_lstm_model(vocab_size, max_len)
rnn_model = create_rnn_model(vocab_size, max_len)

# Enhanced callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Compute class weights to balance the dataset
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

# Train CNN
cnn_checkpoint = ModelCheckpoint('best_cnn.keras', monitor='val_loss', save_best_only=True)
cnn_history = cnn_model.fit(X_train_padded, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            class_weight=class_weight_dict,  # Added class weights
                            callbacks=[early_stopping, cnn_checkpoint, reduce_lr])

# Train LSTM
lstm_checkpoint = ModelCheckpoint('best_lstm.keras', monitor='val_loss', save_best_only=True)
lstm_history = lstm_model.fit(X_train_padded, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.2,
                              class_weight=class_weight_dict,  # Added class weights
                              callbacks=[early_stopping, lstm_checkpoint, reduce_lr])

# Train RNN
rnn_checkpoint = ModelCheckpoint('best_rnn.keras', monitor='val_loss', save_best_only=True)
rnn_history = rnn_model.fit(X_train_padded, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            class_weight=class_weight_dict,  # Added class weights
                            callbacks=[early_stopping, rnn_checkpoint, reduce_lr])