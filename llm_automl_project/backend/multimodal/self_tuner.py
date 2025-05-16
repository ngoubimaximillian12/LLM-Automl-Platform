from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased")

def process_text_column(text_column):
    return [classifier(text)[0] for text in text_column[:5]]  # Sample top 5
