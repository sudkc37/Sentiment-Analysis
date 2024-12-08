from yahoo_fin import news
import pandas as pd

def news_scraper(ticker):
    text_df = news.get_yf_rss(ticker)
    df = pd.DataFrame(text_df)
    return df

