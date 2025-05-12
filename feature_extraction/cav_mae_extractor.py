import torch
import torchaudio
import torchvision
import numpy as np
import sys
import os
import librosa
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

script_dir = os.path.dirname(os.path.abspath(__file__)) 
frames_dir = os.path.join(script_dir, "frames")     

sys.path.append(os.path.abspath('./cav-mae/src/models'))
from cav_mae import CAVMAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#referenced https://github.com/YuanGongND/cav-mae
def initialize_model():
    '''
    load the pretrained cavmae model
    '''

    model = CAVMAE(audio_length=1024, modality_specific_depth=11, norm_pix_loss=True, tr_pos=False)
    model = torch.nn.DataParallel(model)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "audio_model.pth")
    ckpt = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print("Missing keys:", missing, "\nUnexpected keys:", unexpected)
    model.to(device)
    model.eval()
    return model

def get_audio_duration(audio_path):
    '''
    total time in seconds of audio
    '''
    y, sr = librosa.load(audio_path, sr=None)
    duration = int(librosa.get_duration(y=y, sr=sr))
    return y, sr, duration

def extract_av_embeddings(y, sr, duration, model, frames_folder="frames"):
    '''
    extraction of 1 audiovisual embedding for each audio second
    '''
    #waveform -> mel spectogram (cavmae input)
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128
    ).to(device)

    #imagenet transformation  -> cavmae input
    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []

    for t in tqdm(range(1, duration + 1)):
        start_sample = (t - 1) * sr
        end_sample = t * sr#1 second duration chunk
        audio_seg = y[start_sample:end_sample]
        if len(audio_seg) < sr:
            audio_seg = np.pad(audio_seg, (0, sr - len(audio_seg)))

        audio_tensor = torch.from_numpy(audio_seg).float().to(device)
        mel_spec = mel_spec_transform(audio_tensor).squeeze() #time-frequency
        if mel_spec.ndim == 2:
            mel_spec = mel_spec.unsqueeze(0)
        T_cur = mel_spec.shape[-1]
        if T_cur < 1024: #padding or cropping to 1024 frames -> cavmae input
            mel_spec = torch.nn.functional.pad(mel_spec, (0, 1024 - T_cur))
        else:
            mel_spec = mel_spec[:, :, :1024]
        mel_spec = mel_spec.to(device)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        frames_dir = os.path.join(script_dir, "frames")

        frame_path = os.path.join(frames_dir, f"frame_{t:04d}.jpg")
        fallback_path = os.path.join(frames_dir, f"frame_{t-1:04d}.jpg")



        if os.path.exists(frame_path):
            frame_img = Image.open(frame_path).convert("RGB")
        elif os.path.exists(fallback_path):
            frame_img = Image.open(fallback_path).convert("RGB")
        else:
            print(f"Warning: missing frame at t={t}s")
            continue
        img_tensor = img_transform(frame_img).unsqueeze(0).to(device)

        with torch.no_grad():
            audio_feat, video_feat = model.module.forward_feat(mel_spec, img_tensor) #spectogram + image
            audio_emb = audio_feat.mean(dim=1).squeeze(0)#per frame
            video_emb = video_feat.mean(dim=1).squeeze(0)
            av_emb = (audio_emb + video_emb) / 2.0
            embeddings.append(av_emb.cpu().numpy()) #(768,)

        if t % 1000 == 0:
            print(f"Processed {t} / {duration} seconds")

    embeddings = np.stack(embeddings)
    print("Final embeddings shape:", embeddings.shape)
    return embeddings
