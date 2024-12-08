# Leveraging Large Language Model (LLM) for Sentiment Analysis

**Note: This article is for educational purposes only and is not intended to provide financial advice. It aims to demonstrate how language models can be utilized in financial applications.**

This article offers a comprehensive understanding of how fine-tuned language models can be leveraged with advanced data science tools, including web scraping, visualization, modeling, and APIs, to build real-world financial applications. These applications aim to facilitate effective and informed decision-making.

The resulting application features a dashboard with a user-friendly interface that includes:

- Key Real-Time Financial Indicators
- Market Sentiment Meter
- Price Chart of a Selected Portfolio Ticker

The article also provides a simple breakdown and architecture of the model, illustrating the workflow and integration of various components. 

**Scraper.py**
It uses a Yahoo Finance API with a built-in scraping method to extract the news feed from the Yahoo Finance webpage. The API returns financial news as an RSS feed in XML format, fetching the latest financial news published on Yahoo Finance in real-time. We are only using the summary content of each news feed. For simplicity, I have directly fed the summary content into our language model (FinBERT). However, to achieve more robust analysis, we could further leverage large language models (LLMs) like GPT to summarize the news before feeding it into our custom language model. That said, using GPT incurs additional costs.

**LanguageModel.py**
This module incorporates FinBERT, a specialized Natural Language Processing (NLP) model fine-tuned from the pre-trained BERT model. FinBERT is specifically designed for analyzing sentiment in financial texts. It has been further trained on a large corpus of financial data, making it highly effective for sentiment classification tasks within the financial domain.
The AutoTokenize, automatically loads the appropriate tokenizer for the FinBERT model. The tokenizer converts raw text into input tokens that the model can process. Then AutoModelForSequenceClassification loads the pre-trained FinBERT model tailored for sequence classification tasks such as sentiment analysis. And finally,The  embedding  function returns both the tokenizer and model, making them accessible for downstream tasks like processing financial text and performing sentiment analysis.

**SentimentScore.py**
It analyzes financial news summaries to determine overall market sentiment. It uses our  language model to classify each news summary into Positive, Neutral, or Negative categories based on the content. The classifications are stored in the dataframe, and the most frequent sentiment (mode) across all summaries is identified as the overall market sentiment. This provides a quick, automated way to gauge market sentiment from financial news.

**Detail.py**
The ticker_detail function retrieves detailed financial information about a specific stock ticker using the “yfinance” library. It fetches various key metrics such as current price, open price, bid/ask prices, market cap, and industry, organizing them into a dictionary. Leveraging “yfinance.Ticker.info”, we can further model other key indicators like moving averages (MA), stochastic oscillation, implied volatility, gamma, and vega for advanced financial analysis.

**Graph.py**
The plot price function generates a time-series plot of minute-by-minute adjusted closing prices for a given stock ticker over the past day. It calculates the time range by setting the end date to the current date and time and the start date to one day earlier, ensuring it fetches intraday data using an interval of one minute. Using “yfinance”, it retrieves this high-frequency data, and matplotlib is used to create and save the plot, providing a detailed visualization of short-term price movements.

**app2.py**
The application allows users to input a company's stock ticker to analyze recent news articles, assess market sentiment, and access detailed company information alongside a graph of the stock's price trends. By leveraging multiple modules, it provides a streamlined solution for stock market insights.

Under the hood, the app uses a news scraper to fetch the latest articles about the company. A language model then performs sentiment analysis on the scraped content, categorizing the sentiment as positive, neutral, or negative. Additional modules retrieve stock-specific details and generate a graph of the stock's recent price movements. This combination of tools ensures that the app delivers a comprehensive overview of the company's market position.
When users open the app, it displays analysis for a default stock ticker. They can enter a different ticker to dynamically update the sentiment analysis, company details, and stock graph. The results include a sentiment score with a color-coded indicator (green for positive, yellow for neutral, and red for negative), a summary of company-specific data, and a visual representation of the stock price trends.

The application includes robust error handling to manage scenarios like missing data or processing failures. In such cases, placeholders like "N/A" are displayed, ensuring the app remains functional and user-friendly. Additionally, the app dynamically processes user inputs without requiring page reloads, enhancing the overall interactivity.
Built with the Flask framework, the application is designed to run on a server, providing an accessible and efficient web-based interface for users. This architecture makes it well-suited for real-time stock market analysis and sentiment assessment.

https://market-sentiment-analysis.onrender.com
