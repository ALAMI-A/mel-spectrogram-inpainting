import torch
from pathlib import Path
import numpy as np
from processors.audio_processor import AudioMelProcessor

from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
MUSIC_DIR = Path("dataset/musics")
MEL_DIR = Path("dataset/mel_spectres")
MEL_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = 'cuda'  # ou 'cpu'
N_MELS = 128

# Initialiser le processor
processor = AudioMelProcessor(n_mels=N_MELS, device=DEVICE)

# -------------------------
# PARCOURIR LES MP3
# -------------------------
mp3_files = list(MUSIC_DIR.rglob("*.mp3"))


print(f"Found {len(mp3_files)} audio files.")

for mp3_path in mp3_files:
    try:
        # 1️ Charger audio (torchaudio supporte MP3)
        waveform = processor.load_audio(str(mp3_path))
        if waveform is None:
            continue
        
        # 2️ Convertir en Mel-spectrogramme (log-scale)
        mel = processor.audio_to_mel(waveform, log_scale=True)  # shape [1, n_mels, time]

        # 3️ normaliser en [-1,1]
        mel = (mel.clamp(-80.0, 0.0) + 80.0) / 40.0 - 1.0  # [-1,1]
        mel_to_save = mel.cpu().numpy()

        # 4️ Sauvegarder en .npy
        mel_file = MEL_DIR / (mp3_path.stem + ".npy")
        np.save(mel_file, mel_to_save)
        
        print(f"Saved {mel_file}, shape={mel.shape}")
        
    except Exception as e:
        print(f"Error processing {mp3_path}: {e}")
