from scraper import news_scraper
from sentimentScore import sentiment_score
from languageModel import embedding
import warnings
warnings.filterwarnings("ignore")
from detail import ticker_detail

if __name__ == "__main__":
    ticker = "ticker" 
    try:
        df = news_scraper(ticker)
    except Exception as e:
        print(f"Error in scraping news data: {e}")
        df = None
    
    try:
        details = ticker_detail(ticker)
    except Exception as e:
        print(f"Error in fetching ticker details: {e}")
        details = None
    
    try:
        tokenizer, model = embedding()
    except Exception as e:
        print(f"Error in loading embedding model: {e}")
        tokenizer, model = None, None
    
    if df is not None and 'summary' in df.columns:
        try:   
            market_sentiment = sentiment_score(df, tokenizer, model)
            print(f"Market Sentiment: {market_sentiment}")
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
    else:
        print("Error: 'summary' column not found in the scraped data or no data available.")
    
    if details is not None:
        print(details)
    else:
        print("No ticker details available.")

