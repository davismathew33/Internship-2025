import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForQuestionAnswering
from torchvision import transforms
from tqdm import tqdm

# Paths
DATASET_PATH = 'custom_vqa_dataset'
ANNOTATIONS_FILE = os.path.join(DATASET_PATH, 'annotations.json')
IMAGE_DIR = os.path.join(DATASET_PATH, 'images')

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Custom dataset
class CustomVQADataset(Dataset):
    def __init__(self, annotation_file, image_dir, processor):
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.image_dir, sample['image'])
        question = sample['question']
        answer = sample['answer']

        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(image, question, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": answer
        }

# Create dataset and dataloader
dataset = CustomVQADataset(ANNOTATIONS_FILE, IMAGE_DIR, processor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # Adjust as needed
    epoch_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        inputs = {
            "pixel_values": batch["pixel_values"].to(device),
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"]
        }

        # Convert text labels to token ids
        labels = processor.tokenizer(inputs["labels"], padding="max_length", max_length=20, truncation=True, return_tensors="pt").input_ids.to(device)
        outputs = model(pixel_values=inputs["pixel_values"],
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader)}")

# Save fine-tuned model
model.save_pretrained("blip-vqa-custom")
processor.save_pretrained("blip-vqa-custom")
