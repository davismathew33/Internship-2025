# STEP 1: Install Ultralytics (YOLOv8 library)
#pip install ultralytics --upgrade

# STEP 2: Import YOLO and necessary modules
from ultralytics import YOLO
import os

# STEP 3: Define dataset YAML configuration
yaml_content = """
path: /content/umpire_dataset
train: images/train
val: images/val
nc: 3
names: ['umpire_six', 'umpire_four', 'umpire_out']
"""

# Save the config file
with open("umpire.yaml", "w") as f:
    f.write(yaml_content)

# STEP 4: (OPTIONAL) Check dataset structure
# Your folder should look like:
# /content/umpire_dataset/images/train/*.jpg
# /content/umpire_dataset/images/val/*.jpg
# /content/umpire_dataset/labels/train/*.txt
# /content/umpire_dataset/labels/val/*.txt

# STEP 5: Train the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use YOLOv8n for speed, change to yolov8m.pt or yolov8l.pt for accuracy

results = model.train(
    data="umpire.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolo_umpire_custom",
)

# STEP 6: Load trained model for inference
trained_model_path = "runs/detect/yolo_umpire_custom/weights/best.pt"
model = YOLO(trained_model_path)

# STEP 7: Test inference on one image
results = model("sample_test_image.jpg")  # Replace with your test image
results[0].show()  # Display result with bounding boxes
