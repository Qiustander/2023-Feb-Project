'''
Use public data from Yahoo Finance
'''
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode

hang_seng_idx = '^HSI' # Hang Seng Index in Yahoo
euro_stoxx_idx = '^STOXX50E' # Lyxor EURO STOXX 50 Index in Yahoo
dji_idx = '^DJI' # Dow Jones Industrial Average Index in Yahoo


def get_page(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:101.0) Gecko/20100101 Firefox/101.0',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        #  Send GET request
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        #  Get the symbols table
        tables = soup.find_all('table')
        #  #  Convert table to dataframe
        df = pd.read_html(str(tables))[7]

        return df
    except:
        print('Error loading data')
        return None

def download(url: str, idx_name: str):
    payload = get_page(url)
    stock_symbols = payload['Ticker'].values.tolist()
    stock_symbols = [unidecode(symbol).replace('SEHK: ', '')
                     for symbol in stock_symbols]

    stock_symbols = '.HK '.join([symbol.zfill(4) for symbol in stock_symbols]) + '.HK'

    raw_data = yf.download(tickers=stock_symbols, period='10y', interval="1d")

    payload.to_csv('{}_component.csv'.format(idx_name))
    raw_data.to_csv('{}.csv'.format(idx_name))
    return

'''
Download Hang Seng Stocks
'''
url = 'https://en.wikipedia.org/wiki/Hang_Seng_Index'
download(url, 'HSI')
