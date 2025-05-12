import os
import torch
import pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
import scipy.io as sio

device = "cuda" if torch.cuda.is_available() else "cpu"

script_dir = os.path.dirname(os.path.abspath(__file__)) 
#frames_dir = os.path.join(script_dir, "frames")    

def initialize_clip_model():
    '''
    load clip and image preprocessor
    '''
    model_name = "openai/clip-vit-large-patch14"#tokenzier
    processor = CLIPProcessor.from_pretrained(model_name) #image preprocessor
    vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device) #vision encoder + projection head
    vision_model.eval()
    return processor, vision_model

def extract_clip_features(frames_dir, frame_indices, processor, vision_model):
    '''
    clip embeddings extraction
    '''
    features = []
    for frame_num in tqdm(frame_indices, desc="Extracting CLIP features"):
        frame_path = os.path.join(frames_dir, f"frame_{int(frame_num):04d}.jpg")#construct frame
        if not os.path.exists(frame_path):
            continue

        image = Image.open(frame_path).convert("RGB")#convert to rgb
        inputs = processor(images=image, return_tensors="pt").to(device)#preprocess image

        with torch.no_grad():
            outputs = vision_model(**inputs)
        image_features = outputs.image_embeds
        features.append(image_features.cpu().numpy())
    features = np.concatenate(features, axis=0)
    print(f"Final CLIP feature shape: {features.shape}")
    return features

def save_clip_features(features, output_path="vision_clip_features.mat"):
    sio.savemat(output_path, {"features": features})
    print(f"Saved to {output_path}")
