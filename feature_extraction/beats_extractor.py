import os
import sys
import torch
import torchaudio
from tqdm import tqdm


script_dir = os.path.dirname(os.path.abspath(__file__))
beats_src = os.path.join(script_dir, "beats_model", "unilm", "beats")
sys.path.append(beats_src)

#referencing https://github.com/microsoft/unilm/blob/master/beats/README.md
from Tokenizers import Tokenizers, TokenizersConfig
from BEATs import BEATs, BEATsConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
'''
load the pretrained beats tokenizer and encoder 
'''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(script_dir, "beats_model")

    tokenizer_ckpt_path = os.path.join(ckpt_dir, "Tokenizer_iter3_plus_AS2M.pt")
    model_ckpt_path = os.path.join(ckpt_dir, "BEATs_iter3_plus_AS2M.pt")
    #get saved weights from beats model tokenizer
    tokenizer_ckpt = torch.load(tokenizer_ckpt_path, map_location=device)
    model_ckpt = torch.load(model_ckpt_path, map_location=device)

    cfg_tok = TokenizersConfig(tokenizer_ckpt["cfg"])#instance of tokenizer
    BEATs_tokenizer = Tokenizers(cfg_tok)#pass tokenizer into beats model
    BEATs_tokenizer.load_state_dict(tokenizer_ckpt["model"])  #load weights
    BEATs_tokenizer.eval().to(device)

    cfg_model = BEATsConfig(model_ckpt["cfg"])  #model architecutre
    BEATs_model = BEATs(cfg_model)
    BEATs_model.load_state_dict(model_ckpt["model"])  #load model weights
    BEATs_model.eval().to(device)

    return BEATs_tokenizer, BEATs_model

def preprocess_audio(audio_path, target_sample_rate=16000):
    '''
    process audio of movie.mp4, load and resample and convert to waveform
    '''
    waveform, sample_rate = torchaudio.load(audio_path)

    #sampling rate must be 16khz -> resample
    if sample_rate != target_sample_rate:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample_transform(waveform)
    # mono
    if waveform.shape[0] > 1:#waveform can have multiple channels -> make 1 channel 
        waveform = waveform.mean(dim=0, keepdim=True)  #1 channel

    return waveform, target_sample_rate

def extract_beats_features(waveform, BEATs_tokenizer, BEATs_model, sample_rate=16000, chunk_duration=10):
    '''
    get beats extracted features for each second of audio
    '''
    chunk_samples = chunk_duration*sample_rate  #10 second chunks -> run beats on each chunk individually to not crash -> 160k samples
    labels = []
    features = []

    BEATs_tokenizer.eval()
    BEATs_model.eval()

    with torch.no_grad():
        for start in tqdm(range(0, waveform.shape[1], chunk_samples), total=waveform.shape[1] // chunk_samples):
            end = min(start + chunk_samples, waveform.shape[1])#end of current chunk
            chunk = waveform[:, start:end]

            if chunk.shape[1] < chunk_samples:
                #any chunk size < 10 sec -> zero padding
                padding = torch.zeros(1, chunk_samples - chunk.shape[1], device=chunk.device)
                chunk = torch.cat((chunk, padding), dim=1) 

            chunk = chunk.to(device)
            labels_chunk = BEATs_tokenizer.extract_labels(chunk)  #(token ids, )
            labels.append(labels_chunk)#tokenized chunk
            features_chunk = BEATs_model.extract_features(chunk)[0]  # (batch, num frames, frame dim)
            features.append(features_chunk.cpu().squeeze(0))  # remove batch dim -> beats model input take (number frames, frame dim)

    labels = torch.cat(labels, dim=0)  #(token ids, )
    features = torch.cat(features, dim=0)  # (total frames, frame dim)

    # Get features averaged per second
    total_audio_dur = waveform.shape[1] / sample_rate  #num samples / sample rate
    frame_rate = int(features.shape[0] / total_audio_dur)  #beats outputs 49 frames/sec

    features_per_sec = []
    for t in range(1, int(total_audio_dur) + 1):
        start = (t-1)*frame_rate  #t-1 to t features
        end = t*frame_rate
        if end <= features.shape[0]:
            window = features[start:end]
            features_per_sec.append(window.mean(dim=0))  #brain data 1 per sec -> 49 frames per sec feature vectors -> avg 49 feature vectors

    features_per_sec = torch.stack(features_per_sec)  #(time dim, feature dim)
    return features_per_sec.cpu().numpy(), labels.cpu().numpy()
