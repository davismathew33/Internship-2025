from summarizer import Summarizer
from transformers import pipeline
from datasets import load_dataset
import evaluate  # pip install evaluate (if not already installed)

# Initialize models
extractive_model = Summarizer()  # BERT-based extractive summarizer
abstractive_model = pipeline("summarization", model="facebook/bart-large-cnn")

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Load ROUGE metric
rouge = evaluate.load("rouge")

# Function for summarization
def summarize_article(article_text):
    extractive_summary = extractive_model(article_text, ratio=0.3)
    abstractive_summary = abstractive_model(extractive_summary, max_length=100, min_length=30, do_sample=False)
    return abstractive_summary[0]['summary_text']

# Summarize and collect results for ROUGE evaluation
generated_summaries = []
reference_summaries = []

for i in range(3):  # Limiting to 3 articles for testing
    article = dataset['train'][i]
    generated = summarize_article(article['article'])
    reference = article['highlights']

    generated_summaries.append(generated)
    reference_summaries.append(reference)

    print(f"\nSummary {i+1}:\n", generated)
    print(f"Reference Summary {i+1}:\n", reference)

# Compute ROUGE scores
results = rouge.compute(predictions=generated_summaries, references=reference_summaries)
print("\n📊 ROUGE Evaluation:")
print(results)
