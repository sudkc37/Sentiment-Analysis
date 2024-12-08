from flask import Flask, render_template, request
from scraper import news_scraper
from sentimentScore import sentiment_score
from languageModel import embedding
from detail import ticker_detail
import warnings
warnings.filterwarnings("ignore")
from graph import plot_price
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    ticker = request.args.get('ticker) 
    market_sentiment = "N/A"
    sentiment_color = "gray"
    details = None
    plot_img = None  

    if request.method == 'POST':
        ticker = request.form.get('ticker').strip()  

    try:
        df = news_scraper(ticker)
        tokenizer, model = embedding()
        
        if 'summary' in df.columns:
            market_sentiment = sentiment_score(df, tokenizer, model)
            print(f"Market Sentiment: {market_sentiment}")
            
            if market_sentiment == "Positive":
                sentiment_color = "green"
            elif market_sentiment == "Neutral":
                sentiment_color = "yellow"
            else:
                sentiment_color = "red"
        else:
            print("Error: 'summary' column not found in the scraped data.")
            market_sentiment = "Error"
            sentiment_color = "gray"
    except Exception as e:
        print(f"Error in news or sentiment analysis: {e}")
        market_sentiment = "Error"
        sentiment_color = "gray"

    try:
        details = ticker_detail(ticker)
    except Exception as e:
        print(f"Error in fetching ticker details: {e}")
        details = None
    
    try:
        plot_img = plot_price(ticker)
    except Exception as e:
        print(f"Error in fetching plot: {e}")
        plot_img = None
    
    return render_template(
        'temp.html',
        sentiment=market_sentiment,
        color=sentiment_color,
        ticker_details=details,
        plot_img=plot_img  
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(debug=True, threaded=True, host="0.0.0.0", port=port)
