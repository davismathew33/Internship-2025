import os
import cv2
import numpy as np
import easyocr
from pydub import AudioSegment
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import subprocess

# Load models
yolo_model = YOLO("yolov8n.pt")  # Replace with custom model if available
ocr_reader = easyocr.Reader(['en'], gpu=False)

# Paths
TEMP_AUDIO_PATH = "temp_audio.wav"

def extract_audio(video_path, audio_path=TEMP_AUDIO_PATH):
    command = f'ffmpeg -y -i "{video_path}" -q:a 0 -map a "{audio_path}"'
    subprocess.call(command, shell=True)
    return audio_path

def detect_scenes(video_path, threshold=30.0):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

def detect_audio_spikes(audio_path, chunk_ms=500, threshold_db=5):
    audio = AudioSegment.from_file(audio_path)
    volumes = [audio[i:i+chunk_ms].dBFS for i in range(0, len(audio), chunk_ms)]
    mean_volume = np.mean([v for v in volumes if v != float('-inf')])
    spike_times = [i for i, v in enumerate(volumes) if v > mean_volume + threshold_db]
    return set([t * chunk_ms // 1000 for t in spike_times])

def merge_scenes_with_audio(scene_times, audio_spike_times):
    highlights = []
    for start, end in scene_times:
        for sec in range(int(start), int(end)):
            if sec in audio_spike_times:
                highlights.append((start, end))
                break
    return highlights

def detect_umpire_signal(frame, confidence_threshold=0.4):
    results = yolo_model(frame, conf=confidence_threshold)
    labels = results[0].names
    detected_labels = [labels[int(cls)] for cls in results[0].boxes.cls]
    for label in detected_labels:
        if label in ['umpire_six', 'umpire_out', 'umpire_four']:  # Custom class names
            return label
    return None

def extract_score_from_frame(frame):
    h, w, _ = frame.shape
    scoreboard_roi = frame[0:int(h*0.25), int(w*0.70):w]  # Top-right corner
    result = ocr_reader.readtext(scoreboard_roi, detail=0)
    text = " ".join(result)
    return text

def detect_score_changes(video_path, highlight_candidates):
    confirmed = []
    cap = cv2.VideoCapture(video_path)

    prev_runs = prev_wickets = None
    for (start, end) in highlight_candidates:
        mid_time = (start + end) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        text = extract_score_from_frame(frame)
        digits = [int(s) for s in text.split() if s.isdigit()]
        if len(digits) >= 2:
            runs, wickets = digits[0], digits[1]

            if (prev_runs is not None and runs > prev_runs) or \
               (prev_wickets is not None and wickets > prev_wickets):
                confirmed.append((start, end))

            prev_runs, prev_wickets = runs, wickets

    cap.release()
    return confirmed

def confirm_highlights(video_path, highlight_candidates):
    confirmed = []
    cap = cv2.VideoCapture(video_path)

    for (start, end) in highlight_candidates:
        mid_time = (start + end) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        signal = detect_umpire_signal(frame)
        if signal:
            print(f"Confirmed by gesture: {signal} at {mid_time:.1f}s")
            confirmed.append((start, end))
            continue

        text = extract_score_from_frame(frame)
        digits = [int(s) for s in text.split() if s.isdigit()]
        if len(digits) >= 2:
            print(f"Confirmed by OCR: {text} at {mid_time:.1f}s")
            confirmed.append((start, end))

    cap.release()
    return confirmed

def save_highlights_with_audio_ffmpeg(video_path, highlights, output_path="summary.mp4"):
    with open("highlights_list.txt", "w") as f:
        for i, (start, end) in enumerate(highlights):
            temp_clip = f"clip_{i}.mp4"
            cmd = f'ffmpeg -y -i "{video_path}" -ss {start} -to {end} -c:v libx264 -c:a aac "{temp_clip}"'
            subprocess.call(cmd, shell=True)
            f.write(f"file '{temp_clip}'\n")

    cmd_merge = f'ffmpeg -y -f concat -safe 0 -i highlights_list.txt -c copy "{output_path}"'
    subprocess.call(cmd_merge, shell=True)

    for i in range(len(highlights)):
        os.remove(f"clip_{i}.mp4")
    os.remove("highlights_list.txt")

def summarize_video(video_path, output_path="summary.mp4"):
    print("Extracting audio...")
    audio_path = extract_audio(video_path)

    print("Detecting scenes...")
    scenes = detect_scenes(video_path)

    print("Detecting audio spikes...")
    audio_spikes = detect_audio_spikes(audio_path)

    print("Merging scene and audio spikes...")
    audio_highlights = merge_scenes_with_audio(scenes, audio_spikes)

    print("Confirming highlights via gestures and scoreboard OCR...")
    confirmed = confirm_highlights(video_path, audio_highlights)

    if confirmed:
        print(f"Saving {len(confirmed)} highlights...")
        save_highlights_with_audio_ffmpeg(video_path, confirmed, output_path)
    else:
        print("No confirmed highlights found.")

    os.remove(audio_path)

# Example usage
video_path = "input48min.webm"  # Replace with your cricket video
summarize_video(video_path, output_path="cricket_summary.mp4")
