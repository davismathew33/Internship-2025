#  Multimodal Summarization and Event Highlight Generation using Deep Learning

## 📘 Overview

This project implements a suite of multimodal AI systems for understanding and summarizing various media formats including text, images, and videos. It includes the following modules:

### 1. **Text Summarization**
A hybrid model combining:
- **BERTSUM** for extractive summarization (selects key sentences)
- **BART** for abstractive summarization (rewrites selected content into fluent text)

### 2. **Image-to-Story Generation**
- Uses **BLIP** to generate image captions and visual context.
- Uses **OCR** to extract embedded text from images.
- A **custom-trained GPT** model generates coherent, narrative stories from visual and textual content.

### 3. **Visual Question Answering (VQA)**
- Utilizes a fine-tuned **BLIP VQA** model to answer natural language questions based on image content.

### 4. **Video Summarization**
#### Cricket:
- Scene change detection (histogram-based)
- Audio spike detection for crowd/commentary reactions
- **YOLOv5** for umpire gesture detection (wicket, six, no-ball)
- OCR on scoreboard to track runs/wickets

#### Football:
- Scene changes and audio spike analysis
- OCR on scoreboard for score monitoring
- **SlowFast** model fine-tuned for action recognition (goals, fouls, saves)
- Event clustering for seamless highlights

#### Speech Video Summarization:
- Uses **Whisper** to transcribe audio from long-form videos.
- Generates summaries using the **BERTSUM + BART** hybrid pipeline.
- Ideal for summarizing lectures, podcasts, and interviews.

---

## ⚙️ Environment Setup

### ✅ Requirements

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Make sure to install ffmpeg separately for video/audio processing.

 Main Libraries Used:
transformers, torch, opencv-python
whisper, pytesseract, easyocr
scikit-learn, matplotlib, numpy, moviepy

Running the Models:

The following command is used to run the six tasks :
python corresponding_file_name.py

Preprocessing Steps:
Text:
Clean HTML tags, remove stopwords
Sentence segmentation before input to BERTSUM

Image:
Resize to 224x224 for BLIP/GPT input
Use pytesseract or easyocr for embedded text extraction

 Video:
Convert videos to .mp4
Extract frames (for YOLO or SlowFast) using OpenCV
Extract audio for Whisper using ffmpeg:
ffmpeg -i input_video.mp4 -q:a 0 -map a audio.wav

 Evaluation:
ROUGE-1, ROUGE-2, ROUGE-L used for summarization accuracy.

Highlight accuracy manually verified against ground truth.

Qualitative evaluation of image stories and VQA.

Dataset:
CNN/Daily Mail dataset was used for text summarization . Images related to AI robots were used for image to story generation. Images related to parks, gardens were used for visual question answering. Cricket match(England vs India , 3rd T20I, 2022), football match(Spain vs Portugal,FIFA World Cup, 2018), speech video (from YouTube) were used as input for video summarization.



