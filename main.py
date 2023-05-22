from flask import Flask, render_template, request
import os
import pandas as pd
import xgboost as xgb
import joblib
from gan import GAN
import tensorflow as tf
import quandl


app = Flask(__name__)

quandl.ApiConfig.api_key = 'N-9udvz63Yt73U64Q7QG'

def scrape_data():
    try:
        data_folder = os.path.join(os.getcwd(), 'data')
        files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

        stocks = pd.DataFrame()
        for file in files:
            symbol = os.path.splitext(file)[0]
            file_path = os.path.join(data_folder, file)
            df = pd.read_csv(file_path)
            df['Symbol'] = symbol
            stocks = stocks.append(df)

        return stocks

    except Exception as e:
        print("Error occurred while retrieving stock data:", e)
        return pd.DataFrame()

def analyze_data(df):
    # Perform your analysis using XGBoost and GAN models
    num_historical_days = 20
    xgb_model_path = './models/clf.pkl'
    gan_model_path = './models/gan.ckpt'

    # Load XGBoost model
    xgb_model = joblib.load(xgb_model_path)

    # Load GAN model
    gan = GAN(num_historical_days=num_historical_days, is_train=False)
    gan_input_size = 200
    gan_features = gan.generator(gan_input_size)

    # Process the data for prediction
    processed_data = preprocess_data(df)

    # Generate XGBoost predictions
    xgb_data = processed_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    xgb_predictions = xgb_model.predict(xgb.DMatrix(xgb_data))

    # Generate GAN predictions
    gan_data = processed_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    gan_predictions = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, gan_model_path)
        for i in range(num_historical_days, len(gan_data), num_historical_days):
            features = sess.run(gan_features, feed_dict={gan.X: [gan_data[i - num_historical_days:i]]})
            gan_predictions.append(features[0])

    # Return the predictions
    return xgb_predictions, gan_predictions

@app.route('data/historical_data/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()

        try:
            df = scrape_data()
            df = df[df['Symbol'] == symbol]
            if df.empty:
                return render_template('error.html', error='No data available for the specified symbol.')

            xgb_predictions, gan_predictions = analyze_data(df)

            # Render the predictions and other information in the HTML template
            return render_template('index.html', symbol=symbol, xgb_predictions=xgb_predictions,
                                   gan_predictions=gan_predictions)
        except Exception as e:
            return render_template('error.html', error=str(e))
    else:
        return render_template('index.html')

