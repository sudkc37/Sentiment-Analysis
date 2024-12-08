from languageModel import embedding
import torch
import pandas as pd


if __name__ == "__main__":
    tokenizer, model = embedding()  

def sentiment_score(df, tokenizer, model):  
    def calculate_score(news):
        tokens = tokenizer.encode(news, return_tensors='pt') 
        result = model(tokens)  
        score = int(torch.argmax(result.logits)) + 1  
        return 'Positive' if score == 3 else 'Neutral' if score == 2 else 'Negative'
    
  
    df['Market Sentiments'] = df['summary'].apply(calculate_score)
    market_sentiment = df['Market Sentiments'].mode()[0]
    return market_sentiment

