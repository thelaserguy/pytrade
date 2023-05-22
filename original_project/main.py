import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import statsmodels.api as sm
from datetime import datetime, timedelta
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)

def scrape_data(stocks):
    try:
        # Retrieve a list of most active stocks
        top_stocks = web.get_top_stock_holders('mostactive')

        # Select the top 10 stocks with the highest volume traded
        top_stocks = top_stocks.nlargest(10, 'volume')

        # Retrieve data for each top stock
        stocks = pd.DataFrame()
        for symbol in top_stocks.index:
            stock_data = web.get_data_yahoo(symbol)
            stock_data = stock_data[['Date', 'Close', 'Volume']]
            stocks = stocks.append(stock_data)

        return stocks

    except Exception as e:
        print("Error occurred while retrieving stock data:", e)
        return pd.DataFrame()  # Return an empty DataFrame in case of an error


def analyze_data(df):
    stocks = None
    predictions = None
    outlook = None
    r2 = None
    mse = None

    if 'Close' not in df:
        outlook = 'Error: DataFrame does not contain Close column'
    else:
        # Calculate daily return
        df['Daily Return'] = df['Close'].pct_change()

        # Calculate moving average
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()

        # Create a scatter plot to show the relationship between daily return and volume
        sns.scatterplot(x='Volume', y='Daily Return', data=df)
        plt.title('Daily Return vs. Volume')
        plt.savefig('scatter_plot.png')
        plt.close()

        # Create lagged variables for the closing price and volume
        df['Close_Lag1'] = df['Close'].shift(1)
        df['Volume_Lag1'] = df['Volume'].shift(1)

        # Create arrays for the predictor variables and response variable
        X = df[['Close_Lag1', 'Volume_Lag1']].dropna().values
        y = df['Close'][1:].values

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions with the trained model
        df = df.dropna()  # remove any rows with missing values
        X = df[['MA50', 'MA200']]
        y = df['Close']
        predictions = model.predict(X)

        # Calculate R-squared and mean squared error
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)

        # Determine the outlook based on the average daily return for the past month and the current position relative to the moving averages
        avg_return = df['Daily Return'][-30:].mean()
        if avg_return > 0:
            if df['Close'][-1] > df['MA50'][-1] > df['MA200'][-1]:
                outlook = 'Bullish'
            else:
                outlook = 'Neutral'
        else:
            if df['Close'][-1] < df['MA50'][-1] < df['MA200'][-1]:
                outlook = 'Bearish'
            else:
                outlook = 'Neutral'

    # Select the relevant columns to display in the HTML table
    stocks = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail().round(2)

    return stocks, predictions, outlook, r2, mse


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve the form input data
        symbol = request.form['symbol'].upper()
        start_date = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')

        try:
            # Retrieve stock data using pandas_datareader
            df = web.get_data_yahoo(symbol, start_date, end_date)
            df.reset_index(inplace=True)
            df = df[['Date', 'Close', 'Volume']]

            if df.empty:
                return render_template('index.html', error=True)

            stocks, predictions, outlook, r2, mse = analyze_data(df)

            return render_template('index.html', stocks=stocks.to_dict('records'), predictions=predictions,
                                   outlook=outlook, r2=r2, mse=mse)

        except Exception as e:
            return render_template('index.html', error=str(e))

    else:
        # Retrieve the most popular stocks data
        stocks = scrape_data(pd.DataFrame())

        return render_template('index.html', stocks=stocks.to_dict('records'))


if __name__ == '__main__':
    app.run(debug=True)
