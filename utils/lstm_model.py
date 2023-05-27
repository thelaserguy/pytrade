import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

directory = '../data/historical_data/'


def to_sequences(data, seq_length):
    d = []

    for index in range(len(data) - seq_length):
        d.append(data[index: index + seq_length])
    return np.array(d, dtype=np.float16)


def create_model(sequence_length):
        model = Sequential()
        model.add(LSTM(50, activation='tanh', input_shape=(sequence_length, 1), dtype=tf.float16))
        model.add(Dropout(0.2, dtype=tf.float16))
        model.add(Dense(1, dtype=tf.float16))
        model.compile(optimizer='adam', loss='mse')

        return model

sequence_length = 500 

processed_data = []

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        data = pd.read_csv(directory + filename)
        close_prices = data['Close'].values.reshape(-1, 1).astype(np.float16)
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices = scaler.fit_transform(close_prices)
        processed_data.append(close_prices)

all_data = np.concatenate(processed_data)

data_sequences = to_sequences(all_data, sequence_length)

train, test = train_test_split(data_sequences, test_size=0.2)
print("Train shape:", train.shape)  # Add this line to check the shape of the train array
X_train = train[:, :sequence_length]
y_train = train[:, sequence_length:]

X_test = test[:, :-1]
y_test = test[:, -1]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = create_model(sequence_length)

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

model.save('lstm_model.h5')

for filename in tqdm(os.listdir(directory), desc='Processing Files', unit='file'):
    if filename.endswith('.csv'):
        data = pd.read_csv(directory + filename)
        close_prices = data['Close'].values.reshape(-1, 1).astype(np.float16)
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices = scaler.fit_transform(close_prices)
        processed_data.append(close_prices)

