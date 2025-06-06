import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
import os

# ==== Step 1: Define your custom dataset ====
class ImageCaptionDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        caption = item["caption"]
        inputs = self.processor(images=image, text=caption, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

# ==== Step 2: Load model and processor ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# ==== Step 3: Load your dataset ====
dataset = ImageCaptionDataset("dataset.json", processor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ==== Step 4: Optimizer ====
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# ==== Step 5: Training loop ====
model.train()
for epoch in range(3):  # Number of epochs
    total_loss = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# ==== Step 6: Save model ====
model.save_pretrained("blip-finetuned")
processor.save_pretrained("blip-finetuned")

# ==== Step 7: Test generation ====
model.eval()
sample = dataset[0]
image = Image.open(dataset.data[0]["image_path"]).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)
output_ids = model.generate(**inputs)
caption = processor.decode(output_ids[0], skip_special_tokens=True)
print("Generated caption:", caption)
