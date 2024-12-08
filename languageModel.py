from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def embedding():
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
    return tokenizer, model

