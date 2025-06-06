import os
import subprocess
import cv2
import torch
from torch.nn import functional as F
from torchvision.transforms import Compose, Lambda
from pytorchvideo.models.hub import slowfast_r50
from pytorchvideo.data.encoded_video import EncodedVideo
from pydub import AudioSegment
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import easyocr

# --- CONFIGURATION ---
VIDEO_PATH = 'inputfootball48min.mp4'  # Full match video
OUTPUT_DIR = 'highlight_clips'
FINAL_OUTPUT = 'highlights_output.mp4'
AUDIO_SPIKE_THRESHOLD = -20  # dBFS threshold for audio spikes
MIN_HIGHLIGHT_DURATION = 3   # seconds

NUM_CLASSES = 3  # e.g. 0=no_event,1=goal,2=foul
SLOWFAST_CHECKPOINT = 'slowfast_football_finetuned.pth'  # Fine-tuned model path

# Scoreboard cropping params (adjust these to your video)
SCOREBOARD_X, SCOREBOARD_Y, SCOREBOARD_W, SCOREBOARD_H = 50, 20, 300, 80

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. Scene Detection ===
def detect_scenes(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    return [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]

# === 2. Audio Spike Detection ===
def detect_audio_spikes(video_path, threshold_db=AUDIO_SPIKE_THRESHOLD):
    audio = AudioSegment.from_file(video_path)
    spikes = []
    for i in range(0, len(audio), 1000):
        chunk = audio[i:i + 1000]
        if chunk.dBFS > threshold_db:
            spikes.append(i / 1000)  # seconds
    return spikes

# === 3. Load SlowFast model ===
def load_action_model(num_classes, checkpoint_path):
    model = slowfast_r50(pretrained=False)
    model.blocks[-1].proj = torch.nn.Linear(model.blocks[-1].proj.in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# === 4. Prepare video clip tensor for SlowFast ===
def preprocess_video_clip(video_path, start_sec, end_sec):
    video = EncodedVideo.from_path(video_path)
    clip = video.get_clip(start_sec, end_sec)['video']  # T,C,H,W
    clip = clip / 255.0
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0)  # [B,C,T,H,W]
    return clip

# === 5. Run action recognition on clip ===
def run_action_recognition(model, clip_tensor):
    with torch.no_grad():
        clip_tensor = clip_tensor.to(device)
        preds = model(clip_tensor)
        probs = F.softmax(preds, dim=1)
        conf, pred_class = torch.max(probs, dim=1)
    return pred_class.item(), conf.item()

# === 6. Extract scoreboard text from frame using EasyOCR ===
reader = easyocr.Reader(['en'])

def extract_scoreboard_text(video_path, time_sec):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    scoreboard_region = frame[SCOREBOARD_Y:SCOREBOARD_Y+SCOREBOARD_H, SCOREBOARD_X:SCOREBOARD_X+SCOREBOARD_W]
    gray = cv2.cvtColor(scoreboard_region, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    # Combine all recognized text parts
    text = ' '.join([res[1] for res in results])
    return text

# === 7. Detect score changes between scenes ===
def detect_score_changes(scenes, video_path):
    prev_score = None
    score_change_scenes = []
    for start, end in scenes:
        score_text = extract_scoreboard_text(video_path, start)
        if score_text is None:
            continue
        # Simple heuristic: detect if score text changed
        if prev_score is not None and score_text != prev_score:
            score_change_scenes.append((start, end))
        prev_score = score_text
    return score_change_scenes

# === 8. Combine signals and select highlights ===
def match_highlights(scenes, audio_spikes, score_change_scenes, model, video_path, min_duration=MIN_HIGHLIGHT_DURATION):
    highlights = []
    score_change_set = set(score_change_scenes)
    for start, end in scenes:
        if (end - start) < min_duration:
            continue
        # Conditions
        spike_in_scene = any(start <= spike <= end for spike in audio_spikes)
        score_changed = (start, end) in score_change_set

        if not (spike_in_scene or score_changed):
            continue

        clip_tensor = preprocess_video_clip(video_path, start, end)
        pred_class, conf = run_action_recognition(model, clip_tensor)

        CONF_THRESHOLD = 0.6
        if conf >= CONF_THRESHOLD and pred_class != 0:
            highlights.append((start, end, pred_class, conf))
    return highlights

# === 9. Extract clips with ffmpeg ===
def generate_clips(video_path, highlights, output_dir):
    clip_paths = []
    for i, (start, end, event_class, conf) in enumerate(highlights):
        output_file = os.path.join(output_dir, f"clip_{i:03d}_class{event_class}.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", str(start), "-to", str(end),
            "-c:v", "libx264", "-c:a", "aac",
            output_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clip_paths.append(output_file)
    return clip_paths

# === 10. Concatenate clips ===
def concatenate_clips(clip_paths, output_path):
    list_file = "clips_list.txt"
    with open(list_file, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_path
    ]
    subprocess.run(cmd)
    os.remove(list_file)

# === 11. Event class mapping (customize) ===
EVENT_CLASS_MAP = {
    0: 'no_event',
    1: 'goal',
    2: 'foul',
    3: 'shot'
}

def main():
    print("[1] Detecting scenes...")
    scenes = detect_scenes(VIDEO_PATH)
    print(f"Detected {len(scenes)} scenes.")

    print("[2] Detecting audio spikes...")
    spikes = detect_audio_spikes(VIDEO_PATH)
    print(f"Detected {len(spikes)} audio spikes.")

    print("[3] Detecting scoreboard changes...")
    score_changes = detect_score_changes(scenes, VIDEO_PATH)
    print(f"Detected {len(score_changes)} scenes with scoreboard changes.")

    print("[4] Loading action recognition model...")
    action_model = load_action_model(NUM_CLASSES, SLOWFAST_CHECKPOINT)

    print("[5] Matching highlights with combined cues...")
    highlights = match_highlights(scenes, spikes, score_changes, action_model, VIDEO_PATH)
    print(f"Matched {len(highlights)} highlights.")

    if not highlights:
        print("No highlights found.")
        return

    print("[6] Extracting highlight clips...")
    clips = generate_clips(VIDEO_PATH, highlights, OUTPUT_DIR)

    print("[7] Concatenating clips...")
    concatenate_clips(clips, FINAL_OUTPUT)

    print(f"[✓] Highlight video saved as: {FINAL_OUTPUT}")

if __name__ == "__main__":
    main()
