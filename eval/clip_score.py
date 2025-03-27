"""Compute CLIP similarity from video frames to text prompt"""
import clip
import torch
import argparse
import cv2, imageio
from PIL import Image
from pathlib import Path

def load_video(video_path):
    if video_path.endswith('.gif'):
        return imageio.mimread(video_path)
    elif video_path.endswith('.mp4'):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

@torch.no_grad()
def calculate_clip_score(video, text, model, preprocess):
    score_acc = 0.
    for image in video:
        image_input = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(text).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        
        # normalize features
        image_features /= image_features.norm(dim=1, keepdim=True).to(torch.float32)
        text_features /= text_features.norm(dim=1, keepdim=True).to(torch.float32)
        
        score = (text_features * image_features).sum() #* logit_scale
        score_acc += score
    
    return score_acc / len(video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./results")
    opt = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    video = imageio.mimread(opt.video_path)
    text = opt.prompt

    clip_score = calculate_clip_score(video, text, model, preprocess)

    results = clip_score.item()
    print("CLIP score:", results)
    
    # Save results in clip_score object
    Path(opt.output_path).mkdir(parents=True, exist_ok=True)
    torch.save(results, f"{opt.output_path}/clip_score.pt")

