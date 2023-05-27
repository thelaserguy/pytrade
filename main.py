from flask import Flask, render_template, request
import os
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf
import quandl
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

app = Flask(__name__)
quandl.ApiConfig.api_key = 'N-9udvz63Yt73U64Q7QG'
num_historical_days = 20  # Used for LSTM prediction
gan_noise_dim = 300  # Noise dimension for GAN

def scrape_data():
    # Function to scrape data
    pass

def preprocess_data(df, symbol):
    df.fillna(method='ffill', inplace=True)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[features]
    scaler_path = f'scalers/scaler_{symbol}.pkl'
    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(df)
    sequence_length = 50 
    seq_data = to_sequences(scaled_data, sequence_length)
    return seq_data

def to_sequences(data, seq_length):
    d = []
    for index in range(len(data) - seq_length):
        d.append(data[index: index + seq_length])
    return np.array(d, dtype=np.float32)  

def predict_with_lstm(lstm_model, data):
    lstm_data = data.values.reshape((-1, num_historical_days, 1))
    lstm_predictions = lstm_model.predict(lstm_data)
    return lstm_predictions

def generate_synthetic_data(gan_model, num_samples=1000):
    noise = tf.random.normal([num_samples, gan_noise_dim])
    synthetic_data = gan_model(noise, training=False)
    return synthetic_data

@app.route('/', methods=['GET'])
def index():
    stocks = scrape_data()
    top_picks = []
    for symbol in stocks['Symbol'].unique():
        df = stocks[stocks['Symbol'] == symbol]
        lstm_model_path = f'models/lstm_model_{symbol}.h5'
        lstm_model = load_model(lstm_model_path)
        lstm_predictions = predict_with_lstm(lstm_model, df)
        top_picks.append((symbol, lstm_predictions[-1]))  

    top_picks = sorted(top_picks, key=lambda x: x[1], reverse=True)[:10]
    return render_template('index.html', top_picks=top_picks)

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol'].upper()
    if not symbol or not symbol.isalpha():
        return render_template('error.html', error='Invalid symbol.')
    try:
        stocks = scrape_data()
        df = stocks[stocks['Symbol'] == symbol]
        if df.empty:
            return render_template('error.html', error='No data available for the specified symbol.')
        lstm_model_path = f'models/lstm_model_{symbol}.h5'
        lstm_model = load_model(lstm_model_path)
        lstm_predictions = predict_with_lstm(lstm_model, df)
        return render_template('index.html', symbol=symbol, lstm_predictions=lstm_predictions)
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)