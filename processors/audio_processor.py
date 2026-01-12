# import torch
# import torchaudio
# import torchaudio.transforms as T
# import numpy as np
# from pathlib import Path
# import warnings
# warnings.filterwarnings('ignore')

# class AudioMelProcessor:
#     """Handles audio to Mel-spectrogram conversion and processing"""
    
#     def __init__(self, sr=22050, n_mels=128, n_fft=2048, 
#                  hop_length=512, f_min=20, f_max=8000, device='cuda'):
#         self.sr = sr
#         self.n_mels = n_mels
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.f_min = f_min
#         self.f_max = f_max
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
#         # Initialize transforms
#         self.mel_transform = T.MelSpectrogram(
#             sample_rate=sr, n_fft=n_fft, hop_length=hop_length,\
#             n_mels=n_mels, f_min=f_min, f_max=f_max, normalized=False
#         ).to(self.device)
        
#         self.inverse_mel = T.InverseMelScale(
#             sample_rate=sr, n_stft=n_fft//2+1, n_mels=n_mels,\
#             f_min=f_min, f_max=f_max
#         ).to(self.device)
        
#         self.griffin_lim = T.GriffinLim(
#             n_fft=n_fft, n_iter=32, win_length=n_fft, hop_length=hop_length,\
#             power=2.0
#         ).to(self.device)
    
#     def load_audio(self, file_path):
#         """Load audio file and resample to target SR"""
#         try:
#             # torchaudio loads audio as [channel, time]
#             waveform, sr = torchaudio.load(file_path)
            
#             # Resample if needed
#             if sr != self.sr:
#                 resampler = T.Resample(sr, self.sr).to(self.device)
#                 waveform = resampler(waveform.to(self.device))
#             else:
#                 waveform = waveform.to(self.device)
            
#             # Convert to mono if necessary
#             if waveform.shape[0] > 1:
#                 waveform = torch.mean(waveform, dim=0, keepdim=True)
            
#             return waveform
            
#         except Exception as e:
#             print(f"Error loading audio file {file_path}: {e}")
#             return None

#     def audio_to_mel(self, waveform, log_scale=True):
#         mel_spec = self.mel_transform(waveform)  # power mel-spectrogram

#         if log_scale:
#             mel_spec = torchaudio.functional.power_to_db(
#                 mel_spec,
#                 ref=torch.max,
#                 amin=1e-10,
#                 top_db=80.0
#             )

#         if mel_spec.dim() == 2:
#             mel_spec = mel_spec.unsqueeze(0)

#         return mel_spec


#     def mel_to_audio(self, mel_spec, is_log_scale=True):
#         """Convert Mel-spectrogram back to audio using Griffin-Lim"""
#         # Ensure input is 3D: [1, n_mels, time]
#         if mel_spec.dim() == 2:
#             mel_spec = mel_spec.unsqueeze(0)
        
#         # 1. Convert from DB back to amplitude
#         if is_log_scale:
#             # We use 0.5 for power because we are converting to amplitude scale (sqrt of power)
#             mel_spec = torchaudio.functional.DB_to_amplitude(mel_spec, ref=1.0, power=0.5)
        
#         # 2. Inverse Mel Scale (Mel frequency -> Linear STFT magnitude)
#         linear_spec = self.inverse_mel(mel_spec)
        
#         # 3. Griffin-Lim (Linear STFT magnitude -> Audio waveform)
#         waveform = self.griffin_lim(linear_spec)
        
#         return waveform
    
#     def create_inpainting_example(self, mel_spec, missing_duration=2.0, random_position=True):
#         """Create inpainting training example"""
#         total_frames = mel_spec.shape[-1]
#         seconds_per_frame = self.hop_length / self.sr
#         missing_frames = int(missing_duration / seconds_per_frame)
        
#         if missing_frames >= total_frames:
#             raise ValueError(f"Missing duration {missing_duration}s is too long")
        
#         if random_position:
#             max_start = total_frames - missing_frames
#             missing_start = torch.randint(0, max_start, (1,)).item()
#         else:
#             missing_start = total_frames // 3
        
#         missing_end = missing_start + missing_frames
        
#         before_part = mel_spec[..., :missing_start]
#         missing_part = mel_spec[..., missing_start:missing_end]
#         after_part = mel_spec[..., missing_end:]
        
#         # Pad to consistent size
#         target_width = max(before_part.shape[-1], after_part.shape[-1])
        
#         before_padded = torch.nn.functional.pad(before_part, (target_width - before_part.shape[-1], 0))
#         after_padded = torch.nn.functional.pad(after_part, (0, target_width - after_part.shape[-1]))
        
#         missing_info = {
#             'missing_start_frame': missing_start,
#             'missing_end_frame': missing_end,
#             'missing_frames': missing_frames,
#             'missing_duration': missing_duration,
#             'seconds_per_frame': seconds_per_frame
#         }
        
#         return before_padded, after_padded, missing_part, missing_info
    
#     def load_mel_spectrogram(self, file_path):
#         """Load Mel-spectrogram from file"""
#         file_path = Path(file_path)
        
#         if file_path.suffix == '.npy':
#             mel_np = np.load(file_path, allow_pickle=True)
#         else:
#             # Load as text/csv
#             mel_np = np.loadtxt(file_path)
        
#         # Convert to tensor and ensure correct shape [1, n_mels, time]
#         mel_tensor = torch.from_numpy(mel_np).float()
        
#         if mel_tensor.dim() == 2:
#             if mel_tensor.shape[0] == self.n_mels:
#                 mel_tensor = mel_tensor.unsqueeze(0)  # [1, n_mels, time]
#             elif mel_tensor.shape[1] == self.n_mels:
#                 mel_tensor = mel_tensor.T.unsqueeze(0)
        
#         if mel_tensor.dim() == 3 and mel_tensor.shape[1] != self.n_mels:
#             mel_tensor = mel_tensor.transpose(1, 2)
        
#         return mel_tensor.to(self.device)


import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import numpy as np


class AudioMelProcessor:
    """
    Coherent audio <-> mel pipeline.
    Mel spectrogram values are in dB and strictly in [-80, 0].
    """

    def __init__(
        self,
        sr=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=20,
        f_max=8000,
        device="cuda"
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # --- Mel Spectrogram (POWER, not normalized) ---
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            normalized=False
        ).to(self.device)

        # --- Inverse transforms ---
        self.inverse_mel = T.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sr,
            f_min=f_min,
            f_max=f_max
        ).to(self.device)

        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_iter=32,
            power=2.0
        ).to(self.device)

    # ------------------------------------------------------------------
    # AUDIO I/O
    # ------------------------------------------------------------------
    def load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)

        if sr != self.sr:
            waveform = T.Resample(sr, self.sr)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform.to(self.device)

    @staticmethod
    def power_to_db(
        power,
        ref=None,
        amin=1e-10,
        top_db=80.0
    ):
        """
        Convert power spectrogram to decibel (dB).
        Output range: [-top_db, 0]
        """
        if ref is None:
            ref = power.max()

        ref = torch.clamp(ref, min=amin)
        power = torch.clamp(power, min=amin)

        log_spec = 10.0 * torch.log10(power)
        log_ref = 10.0 * torch.log10(ref)

        db = log_spec - log_ref

        if top_db is not None:
            db = torch.clamp(db, min=-top_db)

        return db

    # ------------------------------------------------------------------
    # AUDIO -> MEL (DB)
    # ------------------------------------------------------------------
    def audio_to_mel(self, waveform):
        """
        Output:
            mel_db : Tensor [1, n_mels, time] in dB, range [-80, 0]
        """
        mel_power = self.mel_transform(waveform)

        mel_db = self.power_to_db(
            mel_power,
            ref=mel_power.max(),
            top_db=80.0
        )


        if mel_db.dim() == 2:
            mel_db = mel_db.unsqueeze(0)

        return mel_db

    # ------------------------------------------------------------------
    # MEL (DB) -> AUDIO
    # ------------------------------------------------------------------
    def mel_to_audio(self, mel_db):
        """
        Approximate inversion (ML-safe).
        """
        if mel_db.dim() == 2:
            mel_db = mel_db.unsqueeze(0)

        # dB -> POWER
        mel_power = torchaudio.functional.DB_to_amplitude(
            mel_db,
            ref=1.0,
            power=1.0
        )

        # Mel -> Linear STFT
        linear_power = self.inverse_mel(mel_power)

        # Griffin-Lim
        waveform = self.griffin_lim(linear_power)

        return waveform

    # ------------------------------------------------------------------
    # NORMALIZATION FOR ML
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_db(mel_db):
        """
        [-80, 0] -> [0, 1]
        """
        return (mel_db + 80.0) / 80.0

    @staticmethod
    def denormalize_db(mel_norm):
        """
        [0, 1] -> [-80, 0]
        """
        mel_norm = torch.clamp(mel_norm, 0.0, 1.0)
        return mel_norm * 80.0 - 80.0
