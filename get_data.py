# get_data.py
import yfinance as yf
import pandas as pd
import requests
from fredapi import Fred

# --- CONFIGURATION ---
COMPANIES = {'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google'}
NEWS_API_KEY = "2591dae659db4d3ca198b2fe2bc712ce" # Get a free key from newsapi.org

def fetch_all_data():
    """Fetches stock and news data and saves to CSV files."""
    
    # 1. Fetch Stock Data using yfinance
    print("Fetching stock prices...")
    stock_data = yf.download(list(COMPANIES.keys()), period="1y", interval="1d")
    stock_data.to_csv('stock_prices.csv')
    print("-> Saved to stock_prices.csv")

    # 2. Fetch News Data using NewsAPI
    print("\nFetching news headlines...")
    all_news = []
    for ticker, name in COMPANIES.items():
        # Search for news about the company name
        url = f"https://newsapi.org/v2/everything?q={name}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        articles = response.json().get('articles', [])
        
        for article in articles:
            all_news.append({
                'ticker': ticker,
                'headline': article['title'],
                'date': article['publishedAt'].split('T')[0] # Get just the date part
            })
            
    news_df = pd.DataFrame(all_news)
    news_df.to_csv('news_headlines.csv', index=False)
    print("-> Saved to news_headlines.csv")

def fetch_macro_data():
    print("Fetching macro data from FRED...")
    fred = Fred(api_key="db1687250d584d3805106cf2fa22e616")
    # Get 10-year treasury yield
    treasury_yield = fred.get_series('DGS10')
    treasury_yield.to_csv('macro_data.csv')
    print("-> Saved to macro_data.csv")

if __name__ == "__main__":
    fetch_all_data()# get_data.py
    fetch_macro_data()
