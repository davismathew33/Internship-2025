from transformers import BlipForQuestionAnswering, BlipProcessor
from PIL import Image
import torch

# Load the model and processor
print("Loading model... (this may take a few seconds)")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
print("Model loaded successfully.")

# Load and show the image
image_path = "inputvqa.png"  # Replace with your image filename
try:
    image = Image.open(image_path).convert('RGB')
except FileNotFoundError:
    print(f"Error: Image file '{image_path}' not found.")
    exit()

# Tracking accuracy
total_questions = 0
correct_answers = 0

# Function to compare answers (case-insensitive, trimmed)
def is_correct(predicted, ground_truth):
    return predicted.strip().lower() == ground_truth.strip().lower()

# Interactive loop
print("\nYou can now ask questions about the image.")
print("After each answer, enter the correct answer to calculate accuracy.")
print("Type 'exit' to quit.\n")

while True:
    question = input("Your question: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Preprocess inputs
    inputs = processor(image, question, return_tensors="pt")

    # Generate answer
    with torch.no_grad():
        output = model.generate(**inputs)

    # Decode answer
    predicted_answer = processor.decode(output[0], skip_special_tokens=True)
    print("Model's Answer:", predicted_answer)

    # Get ground truth answer from user
    ground_truth = input("Enter the correct answer (or press Enter to skip evaluation): ")
    if ground_truth.strip():
        total_questions += 1
        if is_correct(predicted_answer, ground_truth):
            correct_answers += 1

        accuracy = (correct_answers / total_questions) * 100
        print(f"Current Accuracy: {accuracy:.2f}%\n")
    else:
        print("Skipped accuracy evaluation for this question.\n")
