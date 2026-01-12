# Mel-Spectrogram Inpainting for Music Reconstruction 🎵🧠

This project explores **music reconstruction via inpainting of Mel-spectrograms** using vision-based deep learning models.  
The goal is to reconstruct missing audio segments by treating the problem as an **image inpainting task**.

---

## 📌 Motivation

Audio recordings may contain missing, corrupted, or poorly recorded segments due to noise, transmission errors, or editing issues.  
Rather than directly generating raw audio, we transform the problem into the **Mel-spectrogram domain**, enabling the use of powerful convolutional architectures originally designed for computer vision.

The objective is **not** to perfectly recover the original audio, but to generate a **coherent and perceptually plausible reconstruction**.

---

## 🧠 Core Idea

**Pipeline:**

Audio with gap  
↓ *(torchaudio)*  
Mel-spectrogram with gap  
↓ *(Mel-Inpainter – CNN / U-Net)*  
Reconstructed Mel-spectrogram  
↓ *(Griffin–Lim)*  
Reconstructed audio segment


This formulation allows us to leverage **image inpainting techniques** for audio processing.

---

## 🏗️ Model Architecture

- Input: Mel-spectrogram context (before & after the gap)
- Model: Convolutional neural network (U-Net–style)
- Output: Missing region of the Mel-spectrogram
- Post-processing: Griffin–Lim algorithm for waveform reconstruction

Baseline models (fully connected) are also implemented for comparison.

---

## 🎼 Dataset

We use the **CAL500 (Computer Audition Lab 500)** dataset:

- 500 songs from Western popular music
- 500 distinct artists
- Average duration ≈ 3 minutes
- Total duration ≈ 27 hours

> ⚠️ Audio files are **not included** in this repository due to size constraints.

---

## 🔊 Audio Preprocessing

- Sample rate: 22,050 Hz
- Mel bands: 128
- FFT size: 2048
- Hop length: 512  
  → 1 frame ≈ 23 ms

Fixed-size windows are used:
- `context_window`: audio context before & after the gap
- `missing_gap`: duration of the missing segment to reconstruct

---

## Audio Reconstruction

**Griffin–Lim** is used to reconstruct the waveform from the predicted Mel-spectrogram.

> Note: Griffin–Lim is an iterative phase estimation algorithm and does not perfectly recover the original signal.  
> Only the reconstructed missing segment is used; the rest comes from the original audio.

---

## 📊 Training & Evaluation

- Loss: pixel-wise loss on Mel-spectrograms
- Training curves show stable convergence
- Qualitative evaluation via spectrogram visualization and audio listening

---

## 📁 Repository Structure

```text
mel-spectrogram-inpainting/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── configs/
│   └── config.py
│
├── models/                 # Neural network architectures
│   ├── fc_baseline.py
│   ├── unet.py
│   └── inpainter.py
│
├── data/                   # Dataset (not versioned)
│   └── music/              # Audio files (local only)
│
├── datasets/               # PyTorch datasets
│   ├── mel_dataset.py
│   └── data_gen.py
│
├── processors/             # Audio & feature processing
│   └── audio_processor.py
│
├── training/
│   └── trainer.py
│
├── scripts/                # Executable scripts
│   ├── train.py
│   └── inference.py
│
├── checkpoints/            # Trained models
│   └── model_final.pth
│
└── results/                # Final results
    ├── audio/
    └── figures/
```


---

## Future Work

- Architecture optimization
- More flexible temporal modeling
- Alternative vocoders (e.g., neural vocoders)
- Quantitative perceptual evaluation

---

## 👤 Author

**Adnane Alami**  
**Anas Maillal**  
Project developed for academic purposes in audio & deep learning research.
