import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from pytorchvideo.models.hub import slowfast_r50
from torchvision.transforms import Compose
from pytorchvideo.transforms import UniformTemporalSubsample, Normalize
from torchvision.transforms import Resize

# === Step 1: Dataset Loader ===
class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        for cls in self.class_to_idx:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".mp4"):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video, _, _ = read_video(video_path, pts_unit="sec")
        video = video.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]

        if self.transform:
            video = self.transform(video)

        return video, label

# === Step 2: Transformations ===
def slowfast_transform():
    return Compose([
        UniformTemporalSubsample(32),
        Resize((224, 224)),
        Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    ])

# === Step 3: Parameters ===
NUM_CLASSES = 4  # goal, foul, shot, no_event
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-4
ROOT_DIR = "custom_dataset/"

# === Step 4: Dataset and Loader ===
transform = slowfast_transform()
dataset = VideoDataset(ROOT_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Step 5: Load Pretrained SlowFast and Modify ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = slowfast_r50(pretrained=True)
model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, NUM_CLASSES)
model = model.to(device)

# === Step 6: Optionally Freeze Early Layers ===
for param in model.blocks[:-1].parameters():
    param.requires_grad = False

# === Step 7: Loss and Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Step 8: Training Loop ===
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for videos, labels in dataloader:
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss:.4f}")

# === Step 9: Save Trained Model ===
torch.save(model.state_dict(), "slowfast_football_finetuned.pth")
print("Model saved as 'slowfast_football_finetuned.pth'")
