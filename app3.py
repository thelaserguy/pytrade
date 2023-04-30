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
import tkinter as tk

app = Flask(__name__)
app.secret_key = 'secret_key'

def update_gui():
    # Generate the Flask URL for the home route
    flask_url = request.host_url
    
    # Update the GUI label with the Flask URL
    url_label.config(text=flask_url)
    
    # Schedule the next GUI update in 100ms
    app_gui.after(100, update_gui)

@app.route('/')
def home():
    return render_template('index.html')

def start_flask():
    # Create the tkinter window
    app_gui = tk.Tk()
    # Add the new function as a route
    app.add_url_rule('/', 'home', home)
    # Start the Flask server in a separate thread
    flask_thread = threading.Thread(target=app.run)
    flask_thread.start()
    # Start the Tkinter main loop
    app_gui.mainloop()

def scrape_data(symbol, stocks):
    start = datetime(2010, 1, 1)
    end = datetime.now()
    try:
        stock_data = get_stock_data(symbol, '1d', start.strftime('%s'), end.strftime('%s'))
        if stock_data.empty:
            return pd.DataFrame()
        else:
            return stock_data
    except Exception as e:
        print("Error occurred while retrieving data: ", e)
    try:
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period='365d')
        if stock_data.empty:
            return pd.DataFrame()
        else:
            return stock_data
    except Exception as e:
        print("Error occurred while retrieving data: ", e)
    # Initialize the stocks variable as an empty DataFrame
    if stocks is None:
        stocks = pd.DataFrame()   
    # Append the stock_data to the stocks DataFrame
    stocks = stocks.append(stock_data[['Symbol', 'Name', 'Price', 'Change', '% Change', 'Volume', 'Avg Vol (3 month)', 'Market Cap', 'r2', 'mse', 'Outlook']])
    try:
        stocks = pd.DataFrame(stocks, columns=['Symbol', 'Name', 'Price (Intraday)', 'Change', '% Change', 'Volume', 'Avg Vol (3 month)', 'Market Cap', 'r2', 'mse', 'Outlook'])
    except TypeError as e:
        print("Error occurred while creating the dataframe: ", e)
        stocks = None

def get_stock_outlook(stock, start, end):
    try:
        stocks = pd.DataFrame(columns=['Symbol', 'Name', 'Price (Intraday)', 'Change', '% Change', 'Volume', 'Avg Vol (3 month)', 'Market Cap', 'r2', 'mse', 'Outlook'])
        
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
            print("Error: No data found for selected dates. Please try again with different dates.")
            return None
        else:
            return stock_data
    except Exception as e:
        print("Error occurred while retrieving data: ", e)
        
    try:
        stocks = stocks.append(stock_data[['Symbol', 'Name', 'Price', 'Change', '% Change', 'Volume', 'Avg Vol (3 month)', 'Market Cap', 'r2', 'mse']])
    except TypeError as e:
        print("Error occurred while appending data to the dataframe: ", e)
        stocks = None
    
    try:
        yf.pdr_override()
        df = web.get_data_yahoo(stock, start=start, end=end)
        return df
    except:
        print("Error retrieving data from Yahoo Finance API")

    try:
        yf.pdr_override()
        df = web.get_data_yahoo(stock, start=start, end=end)
        return df
    except:
        print("Error retrieving data from Alpha Vantage API")

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

if __name__ == '__main__':
    start_flask()

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
        return filename
        

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

@app.route('/', methods=['GET', 'POST'])
def index():
    # Retrieve most-active stocks from Yahoo Finance
    url = 'https://finance.yahoo.com/most-active/'
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    html = requests.get(url, headers=header).text
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    df = pd.read_html(str(table))[0]
    df['Price (Intraday)'] = df['Price (Intraday)'].apply(lambda x: str(x).replace(',', '')).astype(float)
    df['Change'] = df['Change'].apply(lambda x: str(x).replace('%', '')).astype(float)

    # Analyze data for each stock
    stocks = []
    for symbol in df['Symbol']:
        stock_data = scrape_data(symbol, stocks)
        if not stock_data.empty:
            analyzed_data = analyze_data(stock_data)
            outlook = "Bullish" if analyzed_data['MA50'] > analyzed_data['MA200'] else "Bearish"
            stocks.append([symbol, df[df['Symbol'] == symbol]['Name'].values[0], df[df['Symbol'] == symbol]['Price (Intraday)'].values[0], df[df['Symbol'] == symbol]['Change'].values[0], df[df['Symbol'] == symbol]['% Change'].values[0], df[df['Symbol'] == symbol]['Volume'].values[0], df[df['Symbol'] == symbol]['Avg Vol (3 month)'].values[0], df[df['Symbol'] == symbol]['Market Cap'].values[0], analyzed_data['r2'], analyzed_data['mse'], outlook])

    # Convert the stocks list to a dataframe
    stocks = pd.DataFrame(stocks, columns=['Symbol', 'Name', 'Price (Intraday)', 'Change', '% Change', 'Volume', 'Avg Vol (3 month)', 'Market Cap', 'r2', 'mse', 'Outlook'])
    
    return render_template('index.html', stocks=stocks.to_dict('records'))

if __name__ == 'main':
    app.run(debug=True)
    
url = url_for('index', _external=True)
print(url)
