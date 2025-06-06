from PIL import Image
import pytesseract
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
from rouge_score import rouge_scorer
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Your OpenAI API Key
openai.api_key = "sk-157833290abssef1233355cdef4056215" # Replace with your key

# Load BLIP for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Extract caption from image using BLIP
def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Extract text from image using OCR
def extract_text(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image).strip()

# Generate a paragraph using GPT model
def generate_paragraph(caption, extracted_text):
    prompt = (
        "You are a journalist writing a short news article paragraph.\n\n"
        f"Image Description: {caption}\n"
        f"Extracted Text: {extracted_text}\n\n"
        "Write a coherent, informative paragraph for the article:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message["content"].strip()

# Calculate ROUGE scores
def calculate_rouge_scores(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        'rouge1': np.float64(scores['rouge1'].fmeasure),
        'rouge2': np.float64(scores['rouge2'].fmeasure),
        'rougeL': np.float64(scores['rougeL'].fmeasure),
        'rougeLsum': np.float64(scores['rougeL'].fmeasure)  # treating same as rougeL
    }

# Main
if __name__ == "__main__":
    image_path = "imgsum-input.png"  # Replace with your image path

    # Step 1: Generate content
    caption = generate_caption(image_path)
    extracted_text = extract_text(image_path)
    generated_paragraph = generate_paragraph(caption, extracted_text)

    # Step 2: Reference summary
    reference_summary = (
        "Artificial intelligence is increasingly becoming a part of our everyday lives. "
        "In the image, a man in a suit shakes hands with a robot, symbolizing the growing integration of AI "
        "into professional environments. As AI continues to revolutionize industries, there are mixed opinions "
        "about its impact. Some view it as an opportunity for innovation, while others see it as a threat to "
        "traditional jobs and human workers. The debate continues on whether AI will be a partner or a competitor "
        "in the workforce."
    )

    # Step 3: Calculate ROUGE
    rouge_scores = calculate_rouge_scores(reference_summary, generated_paragraph)

    # Step 4: Output
    print("\n📄 Generated Paragraph:\n", generated_paragraph)
    print("\n📝 Reference Summary using bertsum and bart:\n", reference_summary)
    print("\n📊 ROUGE Evaluation:")
    print(rouge_scores)
