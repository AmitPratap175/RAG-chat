# from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.model = None#pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    def analyze(self, text: str) -> str:
        result = {"label": "NEGATIVE"}#self.model(text)[0]
        return "positive" if result["label"] == "POSITIVE" else "negative"