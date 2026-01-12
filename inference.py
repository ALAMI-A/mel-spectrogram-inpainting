import torch
import torchaudio
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Assumes project structure
from config import Config
from processors.audio_processor import AudioMelProcessor
from models.inpainter import MelSpectrogramInpainter

# --- Constants for Mel dB range ---
MIN_DB = -80.0
MAX_DB = 0.0

def denormalize(x):
    """Convert from [-1,1] normalized model output to actual dB scale"""
    x = (x + 1.0) / 2.0         # [-1,1] -> [0,1]
    x = x * (MAX_DB - MIN_DB) + MIN_DB
    return torch.clamp(x, MIN_DB, MAX_DB)

def to_numpy(x):
    """Convert tensor to numpy for plotting"""
    return x.squeeze().cpu().numpy()

def save_plot(original, masked, inpainted, save_path, start_frame, end_frame,
              context_before=20, context_after=20):
    """Visualize Original, Masked, and Inpainted Mel spectrograms with extra frames around the gap."""
    
    # Compute display range (clip around start/end with context)
    total_frames = original.shape[-1]
    display_start = max(0, start_frame - context_before)
    display_end = min(total_frames, end_frame + context_after)
    
    # Slice for display
    original_disp = original[..., display_start:display_end]
    masked_disp = masked[..., display_start:display_end]
    inpainted_disp = inpainted[..., display_start:display_end]
    
    all_mels = [to_numpy(original_disp), to_numpy(masked_disp), to_numpy(inpainted_disp)]
    global_min = min(m.min() for m in all_mels if m.size > 0)
    global_max = max(m.max() for m in all_mels if m.size > 0)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    cmap = 'viridis'
    n_mels = original.shape[-2]
    
    # Adjust boundary relative to display slice
    boundary_params = {
        'xy': (start_frame - display_start, 0),
        'width': end_frame - start_frame,
        'height': n_mels,
        'fill': False,
        'edgecolor': 'red',
        'linewidth': 2,
        'linestyle': '--',
        'zorder': 10
    }
    
    titles = ["1. Original (Ground Truth)", "2. Masked Input (Gap region)", "3. Inpainted Result"]
    for i, ax in enumerate(axes):
        ax.imshow(all_mels[i], origin='lower', aspect='auto', cmap=cmap,
                  vmin=global_min, vmax=global_max)
        ax.set_title(titles[i])
        ax.add_patch(plt.Rectangle(**boundary_params))
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def run_inference(
    checkpoint_path,
    audio_file,
    output_dir="results",
    missing_duration=2.0,
    start_time=None
):
    import torch
    import torchaudio
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # =========================
    # 1. Processor & model
    # =========================
    processor = AudioMelProcessor(**Config.get_processor_config())
    inpainter = MelSpectrogramInpainter(**Config.get_model_config())

    checkpoint = torch.load(checkpoint_path, map_location=device)
    inpainter.model.load_state_dict(checkpoint["model"])
    inpainter.model.to(device).eval()

    # =========================
    # 2. Load audio & Mel
    # =========================
    waveform = processor.load_audio(audio_file)  # shape [1, T]
    if waveform is None:
        print("X Failed to load audio.")
        return

    waveform = waveform.to(device)
    full_mel = processor.audio_to_mel(waveform)  # [n_mels, frames]

    # =========================
    # 3. Time → frames
    # =========================
    seconds_per_frame = processor.hop_length / processor.sr
    missing_width = int(missing_duration / seconds_per_frame)

    total_frames = full_mel.shape[-1]

    if start_time is None:
        start_frame = (total_frames - missing_width) // 2
    else:
        start_frame = int(start_time / seconds_per_frame)

    end_frame = start_frame + missing_width

    # =========================
    # 4. Context extraction
    # =========================
    before_raw = full_mel[..., :start_frame]
    after_raw = full_mel[..., end_frame:]

    max_context = Config.MAX_CONTEXT_WIDTH

    def pad_or_crop(x, left=True):
        if x.shape[-1] >= max_context:
            return x[..., -max_context:] if left else x[..., :max_context]
        pad = max_context - x.shape[-1]
        return torch.nn.functional.pad(x, (pad, 0) if left else (0, pad))

    before_in = pad_or_crop(before_raw, left=True)
    after_in = pad_or_crop(after_raw, left=False)

    def norm_for_model(x):
        return 2.0 * (x - MIN_DB) / (MAX_DB - MIN_DB) - 1.0

    before_tensor = norm_for_model(before_in).unsqueeze(0).to(device)
    after_tensor = norm_for_model(after_in).unsqueeze(0).to(device)

    # =========================
    # 5. Inference
    # =========================
    with torch.no_grad():
        prediction_norm = inpainter.predict(
            before_tensor,
            after_tensor
        )

    prediction_db = denormalize(prediction_norm.squeeze(0))

    # =========================
    # 6. Full Mel reconstruction
    # =========================
    reconstructed_mel = torch.cat(
        [before_raw, prediction_db, after_raw],
        dim=-1
    )

    # =========================
    # 7. Mel → audio (full)
    # =========================
    print("🔊 Converting Mel to waveform...")
    reconstructed_audio = processor.mel_to_audio(
        reconstructed_mel.unsqueeze(0).to(device)
    ).squeeze(0)

    #original_audio = waveform.squeeze(0)
    # Force shape [channels, T]
    if waveform.dim() == 1:
        original_audio = waveform.unsqueeze(0)
    else:
        original_audio = waveform

    if reconstructed_audio.dim() == 1:
        reconstructed_audio = reconstructed_audio.unsqueeze(0)


    # =========================
    # 8. Splicing audio (ROBUST)
    # =========================

    # Force shape [C, T]
    if original_audio.dim() == 1:
        original_audio = original_audio.unsqueeze(0)
    if reconstructed_audio.dim() == 1:
        reconstructed_audio = reconstructed_audio.unsqueeze(0)

    # Align global length
    min_len = min(
        original_audio.shape[-1],
        reconstructed_audio.shape[-1]
    )

    original_audio = original_audio[..., :min_len]
    reconstructed_audio = reconstructed_audio[..., :min_len]

    # Compute approximate boundaries
    samples_per_frame = processor.hop_length
    start_sample = start_frame * samples_per_frame
    end_sample = start_sample + reconstructed_audio.shape[-1] - original_audio.shape[-1]

    # Clamp (VERY IMPORTANT)
    start_sample = max(0, min(start_sample, min_len))
    end_sample = max(start_sample, min(end_sample, min_len))

    # Gap length taken from reconstructed audio
    gap_len = end_sample - start_sample

    # =========================
    # Crossfade (safe)
    # =========================
    fade_len = int(0.02 * processor.sr)  # 20 ms
    fade_len = min(fade_len, start_sample, min_len - end_sample)

    if fade_len > 0:
        fade_in = torch.linspace(0, 1, fade_len, device=original_audio.device)
        fade_out = 1.0 - fade_in

        reconstructed_audio[..., start_sample:start_sample + fade_len] *= fade_in
        original_audio[..., start_sample:start_sample + fade_len] *= fade_out

        reconstructed_audio[..., end_sample - fade_len:end_sample] *= fade_out
        original_audio[..., end_sample - fade_len:end_sample] *= fade_in
    else:
        print("⚠️ Crossfade skipped (gap too close to boundaries)")

    # =========================
    # Final assembly (SAFE)
    # =========================
    final_audio = torch.cat([
        original_audio[..., :start_sample],
        reconstructed_audio[..., start_sample:end_sample],
        original_audio[..., end_sample:]
    ], dim=-1)


    # =========================
    # 9. Save results
    # =========================
    base_name = Path(audio_file).stem
    out_audio = output_dir / f"{base_name}_inpainted.wav"

    # ------------------------------
    # Ensure 2D tensor for torchaudio.save
    # ------------------------------
    if final_audio.dim() == 1:
        # mono
        final_audio_to_save = final_audio.unsqueeze(0)
    elif final_audio.dim() == 2:
        # stereo or multi-channel
        final_audio_to_save = final_audio
    else:
        # edge case: extra dimension
        final_audio_to_save = final_audio.squeeze(0)

    torchaudio.save(
        out_audio,
        final_audio_to_save.cpu(),
        processor.sr
    )

    # 9. Save results
    # Exemple avec 50 frames avant et 50 frames après pour mieux voir le gap
    zeros = torch.full_like(prediction_db, MIN_DB)
    masked_mel_vis = torch.cat([before_raw.cpu(), zeros.cpu(), after_raw.cpu()], dim=-1)
    save_plot(full_mel.cpu(), masked_mel_vis, reconstructed_mel, 
            output_dir / f"{base_name}_comparison.png",
            start_frame, end_frame,
            context_before=3000, context_after=3000)

    # =========================
    # 10. Save Mel spectrograms
    # =========================

    # --- Original full Mel ---
    torch.save(
        full_mel.cpu(),
        f"{base_name}_mel_original.pt"
    )

    # --- Reconstructed full Mel ---
    torch.save(
        reconstructed_mel.cpu(),
        f"{base_name}_mel_reconstructed.pt"
    )

    # --- Masked Mel (optional, debug/visualisation) ---
    torch.save(
        masked_mel_vis,
        f"{base_name}_mel_masked.pt"
    )

    print("💾 Mel spectrograms saved (.pt)")
    print(f"! Inpainting completed: {out_audio}")


import argparse
import os

def ask(prompt, default=None, cast=str):
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    while True:
        user_input = input(prompt).strip()
        if user_input == "":
            return default
        try:
            return cast(user_input)
        except ValueError:
            print("❌ Valeur invalide, réessaie.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Inpainting Inference")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--start", type=float, default=None)

    args = parser.parse_args()

    # 🔹 Valeurs par défaut interactives
    checkpoint = ask(
        "📦 Checkpoint",
        default=args.checkpoint or "checkpoints/model_final.pth"
    )

    input_audio = ask(
        "🎵 Audio d'entrée",
        default=args.input or "ATTACK_extract.wav"
    )

    output_dir = ask(
        "📁 Dossier de sortie",
        default=args.output_dir or "results"
    )

    duration = ask(
        "⏱️ Durée de l'inpainting (sec)",
        default=args.duration or 2.0,
        cast=float
    )

    start = ask(
        "▶️ Début de la zone manquante (sec)",
        default=args.start,
        cast=float
    )

    print("\n📌 Paramètres utilisés :")
    print(f"  checkpoint : {checkpoint}")
    print(f"  input      : {input_audio}")
    print(f"  output_dir : {output_dir}")
    print(f"  duration   : {duration}")
    print(f"  start      : {start}")

    run_inference(checkpoint, input_audio, output_dir, duration, start)
