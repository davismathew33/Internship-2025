import os
import cv2
import ffmpeg
import whisper
import textwrap
import numpy as np
from transformers import pipeline
from summarizer import Summarizer  # BERTSUM extractive summarizer
from rouge_score import rouge_scorer

# 🔊 Extract audio using ffmpeg
def extract_audio(video_path, audio_path="extracted_audio.wav"):
    try:
        ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run(quiet=True, overwrite_output=True)
        return audio_path
    except ffmpeg.Error as e:
        print("❌ FFmpeg error:", e)
        return None

# 🗣️ Transcribe audio with Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # You can use "small" or "medium" for better accuracy
    result = model.transcribe(audio_path)
    return result['text']

# 📝 Summarize text with combined BERTSUM + BART
def summarize_text(text, max_length=130, min_length=30):
    # Initialize extractive summarizer (BERTSUM)
    bert_summarizer = Summarizer()

    # Initialize abstractive summarizer (BART)
    bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Split text into manageable chunks (~1000 characters)
    chunks = textwrap.wrap(text, width=1000, break_long_words=False, break_on_hyphens=False)
    extracted_texts = []

    for i, chunk in enumerate(chunks):
        print(f"🧠 Extracting summary sentences from chunk {i+1}/{len(chunks)} using BERTSUM...")
        extracted = bert_summarizer(chunk)
        extracted_texts.append(extracted)

    # Combine extracted sentences into one text
    combined_extracted = "\n".join(extracted_texts)

    # Step 2: Use BART to generate abstractive summary from extracted text
    print("🧠 Generating abstractive summary from extracted text using BART...")
    bart_summary = bart_summarizer(
        combined_extracted, max_length=max_length, min_length=min_length, do_sample=False
    )

    return bart_summary[0]['summary_text']

# 🎞️ (Optional) Scene detection
def detect_scenes(video_path, threshold=30.0):
    cap = cv2.VideoCapture(video_path)
    last_frame = None
    scene_frames = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if last_frame is not None:
            diff = cv2.absdiff(gray, last_frame)
            non_zero_count = cv2.countNonZero(diff)
            if non_zero_count > threshold * gray.size:
                scene_frames.append(frame_number)

        last_frame = gray
        frame_number += 1

    cap.release()
    return scene_frames

# 🎯 Full pipeline: video -> summary
def video_to_text_summary(video_path):
    print("🔍 Detecting scenes (optional)...")
    scenes = detect_scenes(video_path)
    print(f"✅ Detected {len(scenes)} scene changes.\n")

    print("🔊 Extracting audio...")
    audio_path = extract_audio(video_path)
    if not audio_path:
        return "Audio extraction failed."

    print("🗣️ Transcribing speech...")
    transcript = transcribe_audio(audio_path)
    os.remove(audio_path)

    print("📝 Generating summary from transcript...")
    summary = summarize_text(transcript)

    return summary

# 🧪 ROUGE score evaluation
def evaluate_rouge(generated_summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    rouge_eval = {
        'rouge1': np.float64(scores['rouge1'].fmeasure),
        'rouge2': np.float64(scores['rouge2'].fmeasure),
        'rougeL': np.float64(scores['rougeL'].fmeasure),
        'rougeLsum': np.float64(scores['rougeL'].fmeasure)  # same as rougeL for now
    }
    return rouge_eval

# 📁 Example usage
if __name__ == "__main__":
    video_file = "inputts45min.webm"  # Replace with your video filename
    print("📂 Processing:", video_file)
    generated_summary = video_to_text_summary(video_file)
    print("\n📄 Final Summary:\n", generated_summary)

    # ✏️ Reference summary for comparison
    reference_summary = """
    Stop letting doubt and excuses control your life — now is your moment of truth. Success starts with discipline, hunger, and falling in love with the grind, not just the reward. Champions don’t make excuses; they embrace discomfort and adversity to grow. Life will knock you down, but if you can look up, you can get up. The mind is your strongest weapon — control it, or it will control you. Your struggles, losses, and pain are not in vain; they shape you. True transformation happens when you stop running from challenges and face them head-on. Surround yourself with people who push you forward, not ones who hold you back. The journey is lonely, and it’s not easy, but greatness requires sacrifice. Speak your desires louder than your fears, stay committed, and never settle for less than your potential.
    """

    print("\n🔍 Calculating ROUGE scores...")
    rouge_scores = evaluate_rouge(generated_summary, reference_summary)
    print("\n📊 ROUGE Evaluation:")
    print(rouge_scores)
