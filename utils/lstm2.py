import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def to_sequences(data, seq_length):
    d = []
    for index in range(len(data) - seq_length):
        d.append(data[index: index + seq_length])
    return np.array(d, dtype=np.float32)  # float32 provides a good balance between accuracy and memory consumption

def create_model(sequence_length):
    model = Sequential()
    model.add(LSTM(16, activation='tanh', input_shape=(sequence_length, 1), dtype=tf.float32))  # Update input_shape
    model.add(Dense(1, dtype=tf.float32))

    # Define an Adam optimizer with learning rate 0.001 and gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(optimizer=optimizer, loss='mse')
    return model

sequence_length = 50

# Load the CSV file
data = pd.read_csv('../data/historical_data/cleaned.csv')

# Initialize a MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Iterating over each column (each stock) in the dataframe
for column in tqdm(data.columns[1:], desc='Processing Stocks', unit='stock'):
    close_prices = data[column].values.reshape(-1, 1).astype(np.float32)

    # Normalize close_prices
    close_prices = scaler.fit_transform(close_prices)

    # Create the directory if it doesn't exist
    scaler_dir = '../scalers/'
    os.makedirs(scaler_dir, exist_ok=True)

    # Save the scaler for future use
    scaler_filename = f'{scaler_dir}scaler_{column}.pkl'
    joblib.dump(scaler, scaler_filename)

    data_sequences = to_sequences(close_prices, sequence_length+1)

    print(f'Shape of data_sequences for stock {column}: {data_sequences.shape}')

    train, test = train_test_split(data_sequences, test_size=0.2, random_state=42)

    X_train = train[:, :-1]
    y_train = train[:, -1]

    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)

    X_test = test[:, :-1]
    y_test = test[:, -1]

    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = create_model(sequence_length)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')

    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

    model_filename = f'../models/lstm_model_{column}.h5'
    model.save(model_filename)
