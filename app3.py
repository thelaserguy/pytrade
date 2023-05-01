import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import statsmodels.api as sm
from bs4 import BeautifulSoup
from datetime import datetime
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import yfinance as yf
import threading
from datetime import timedelta

app = Flask(__name__)

stocks = pd.DataFrame(columns=['Symbol', 'Name', 'Price (Intraday)', 'Change', '% Change', 'Volume', 'Avg Vol (3 month)', 'Market Cap', 'r2', 'mse', 'Outlook'])
    
def scrape_data():
    global stocks
    
    # retrieve a list of most active stocks
    most_active = yf.Tickers('mostactive')

    # filter stocks based on volume traded
    top_stocks = most_active.tickers[most_active.tickers['volume'] > most_active.tickers['volume'].mean()]

    # select the top 10 stocks with highest volume traded
    top_stocks = top_stocks.nlargest(10, 'volume')

    # retrieve data for each top stock
    for symbol in top_stocks.index:
        try:
            stock = yf.Ticker(symbol)
            stock_data = stock.history(period='365d')
            if not stock_data.empty:
                stock_data = stock_data[['Symbol', 'Name', 'Price (Intraday)', 'Change', '% Change', 'Volume', 'Avg Vol (3 month)', 'Market Cap']]
                stocks = stocks.append(stock_data)
        except Exception as e:
            print("Error occurred while retrieving data from yfinance API: ", e)

    return stocks


if stocks.empty:
    stocks = pd.DataFrame(columns=['Symbol', 'Name', 'Price (Intraday)', 'Change', '% Change', 'Volume', 'Avg Vol (3 month)', 'Market Cap', 'r2', 'mse', 'Outlook'])

print(stocks)




def get_stock_outlook(stock, start, end):
    try:
        stock_data = yf.Ticker(stock).info
        stock_data = pd.DataFrame(stock_data.items())
        stock_data.columns = ['Variable', 'Value']
        stock_data = stock_data.set_index('Variable')
        stock_data = stock_data.T
        stock_data['Price'] = stock_data['RegularMarketPrice']
        stock_data['Change'] = stock_data['RegularMarketChange']
        stock_data['% Change'] = stock_data['RegularMarketChangePercent']
        stock_data['Volume'] = stock_data['RegularMarketVolume']
        stock_data['Avg Vol (3 month)'] = stock_data['averageDailyVolume3Month']
        stock_data['Market Cap'] = stock_data['marketCap']
        stock_data = stock_data[['Symbol', 'Name', 'Price', 'Change', '% Change', 'Volume', 'Avg Vol (3 month)', 'Market Cap']]
        
        start = datetime.strptime(start, '%Y-%m-%d') - timedelta(days=7)
        end = datetime.strptime(end, '%Y-%m-%d')
        stock_data['Date'] = stock_data.index
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data['Date'] = stock_data['Date'].apply(lambda x: x.date())
        stock_data['Date'] = stock_data['Date'].astype(str)
        stock_data = stock_data[(stock_data['Date'] >= str(start.date())) & (stock_data['Date'] <= str(end.date()))]
        
        x = np.array(range(len(stock_data)))
        x = x.reshape(-1,1)
        y = stock_data['Price'].to_numpy()
        model = LinearRegression().fit(x, y)
        stock_data['r2'] = model.score(x, y)
        stock_data['mse'] = np.mean((model.predict(x) - y)**2)
            
        if stock_data.empty:
            raise ValueError("Error: No data found for selected dates. Please try again with different dates.")
        else:
            return stock_data
        
    except Exception as e:
        raise ValueError(f"Error occurred while retrieving data: {e}")
        
    return None

def get_stock_data(symbols, interval, period):
    dfs = []
    for symbol in symbols:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&period1={period}&events=div%2Csplit"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            result = response.json()
            quotes = result["chart"]["result"][0]["indicators"]["quote"][0]
            data = {
                "timestamp": result["chart"]["result"][0]["timestamp"],
                "open": quotes["open"],
                "close": quotes["close"],
                "high": quotes["high"],
                "low": quotes["low"],
                "volume": quotes["volume"]
            }
            dfs.append(pd.DataFrame(data, index=data['timestamp']))
    return pd.concat(dfs)


	
def analyze_data(df):
    stocks = None
    predictions = None
    outlook = None

    if 'Adj Close' not in df:
        outlook = 'Error: DataFrame does not contain Adj Close column'
    else:
        # Calculate daily return
        df['Daily Return'] = df['Adj Close'].pct_change()

        # Calculate moving average
        df['MA50'] = df['Adj Close'].rolling(window=50).mean()
        df['MA200'] = df['Adj Close'].rolling(window=200).mean()

        # Create a scatter plot to show the relationship between daily return and volume
        sns.scatterplot(x='Volume', y='Daily Return', data=df)
        plt.title('Daily Return vs. Volume')
        plt.savefig('scatter_plot.png')

        # Create lagged variables for the closing price and volume
        df['Close_Lag1'] = df['Adj Close'].shift(1)
        df['Volume_Lag1'] = df['Volume'].shift(1)

        # Create arrays for the predictor variables and response variable
        X = df[['Close_Lag1', 'Volume_Lag1']].dropna().values
        y = df['Adj Close'][1:].values

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions with the trained model
        df = df.dropna()  # remove any rows with missing values
        X = df[['MA50', 'MA200']]
        y = df['Adj Close']
        model = LinearRegression().fit(X, y)
        df['Predicted'] = model.predict(X)

        # Calculate R-squared and mean squared error
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)

        # Display the coefficients and intercept of the linear regression model
        print('Coefficients:', model.coef_)
        print('Intercept:', model.intercept_)

        # Create a line plot to show the actual and predicted closing prices
        filename = 'scatter_plot.png'
        sns.scatterplot(x='Predicted', y='Actual', data=df)
        plt.title('Actual vs. Predicted Closing Prices')
        plt.savefig(filename)
        plt.close()


        # Display the summary statistics of the linear regression model
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        print(model.summary())

        # Calculate R-squared and mean squared error
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)

        # Determine the outlook based on the average daily return for the past month and the current position relative to the moving averages
        avg_return = df['Daily Return'][-30:].mean()
        if avg_return > 0:
            if df['Adj Close'][-1] > df['MA50'][-1] > df['MA200'][-1]:
                outlook = 'Bullish'
            else:
                outlook = 'Neutral'
        else:
            if df['Adj Close'][-1] < df['MA50'][-1] < df['MA200'][-1]:
                outlook = 'Bearish'
            else:
                outlook = 'Neutral'

    # Select the relevant columns to display in the HTML table
    stocks = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail().round(2)

    return stocks, predictions, outlook, r2, mse

    try:
        if 'Symbol' in df:
            symbol = df['Symbol'][0]
        else:
            print("No 'Symbol' column found")
            print(f"No data returned for {symbol}")
            return None, None, None, None, None

        stocks, predictions, outlook, r2, mse = analyze_data(df)

        return stocks, predictions[-1], outlook, r2, mse

    except TypeError:
        print(f"No data returned for {symbol}")
        return None, None, None, None, None

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None, None, None, None

def main():
    stocks = pd.DataFrame()
    stocks = scrape_data(stocks)
    print(stocks)

@app.route('/', methods=['GET', 'POST'])
def index():
    global stocks

    if request.method == 'POST':
        # get form input data
        symbol = request.form['symbol'].upper()

        # scrape stock data from Yahoo Finance API
        stock_data = scrape_data()

        # get data for the selected stock
        stock_data = stock_data[stock_data['Symbol'] == symbol]

        if stock_data.empty:
            # if no data is found for the selected stock, display an error message
            return render_template('index.html', error=True)
        else:
            # perform linear regression analysis
            X = np.array(range(len(stock_data)))
            X = X.reshape(-1, 1)
            y = np.array(stock_data['Price (Intraday)'])
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            # get stock outlook
            start_date = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = datetime.today().strftime('%Y-%m-%d')
            outlook = get_stock_outlook(symbol, start_date, end_date)

            # update stock data with regression analysis results and outlook
            stock_data['r2'] = r2
            stock_data['mse'] = mse
            stock_data['Outlook'] = outlook

            return render_template('index.html', stocks=stock_data.to_dict('records'))

    else:
        # if the request method is GET, display the homepage
        return render_template('index.html')

if __name__ == 'main':
    app.run(debug=True)

if __name__ == '__main__':
    app.run()

if __name__ == '__main__':
    main()
    
    
