import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np



url = 'https://finance.yahoo.com/most-active/'
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}


def scrape_data():
    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    stocks_table = soup.find_all('table')[0]
   
    stocks_df = pd.read_html(str(stocks_table))[0]

    return stocks_df
    
    
def analyze_data(stocks_df):
    stocks_df['Decision'] = ['Buy' if x >= 0.5 else 'Sell' for x in np.random.rand(len(stocks_df))]
    return stocks_df
    
    
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    stocks_df = scrape_data()
    stocks_df = analyze_data(stocks_df)
    return render_template('index1.html', stocks=stocks_df)

if __name__ == '__main__':
    app.run(debug=True)
