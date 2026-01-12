# import torch
# from torch.utils.data import Dataset
# import numpy as np
# from pathlib import Path
# import torch.nn.functional as F

# def create_training_example(spectrogram, missing_start, missing_width, max_context_width=50):
#     """Create training example from complete spectrogram"""
#     # Extract parts
#     before_part = spectrogram[:, :, :missing_start]
#     missing_part = spectrogram[:, :, missing_start:missing_start+missing_width]
#     after_part = spectrogram[:, :, missing_start+missing_width:]
    
#     # Pad context parts
#     if before_part.shape[2] > max_context_width:
#         before_part = before_part[:, :, -max_context_width:]
#     else:
#         pad_width = max_context_width - before_part.shape[2]
#         before_part = F.pad(before_part, (pad_width, 0))
    
#     if after_part.shape[2] > max_context_width:
#         after_part = after_part[:, :, :max_context_width]
#     else:
#         pad_width = max_context_width - after_part.shape[2]
#         after_part = F.pad(after_part, (0, pad_width))
    
#     # Add channel dimension
#     before_part = before_part.unsqueeze(1)
#     after_part = after_part.unsqueeze(1)
#     missing_part = missing_part.unsqueeze(1)
    
#     return before_part, after_part, missing_part


# class MelInpaintingDataset(Dataset):
#     """Dataset for Mel-spectrogram inpainting"""
    
#     def __init__(self, processor, mel_folder, missing_durations=[1.0, 2.0, 3.0], 
#                  max_files=10, max_context_width=50, test_mode=False, aug_db=2):
#         self.processor = processor
#         self.missing_durations = missing_durations
#         self.max_context_width = max_context_width
#         self.test_mode = test_mode
#         self.examples = []
        
#         # Find Mel files
#         mel_folder = Path(mel_folder)
#         mel_files = list(mel_folder.glob("*.mel")) + list(mel_folder.glob("*.npy"))
        
#         if max_files:
#             mel_files = mel_files[:max_files]
            
#         print(f"Processing {len(mel_files)} files...")
#         self._create_examples(mel_files, aug_db)
    
#     def _create_examples(self, mel_files, aug_db=2):
#         """Create examples using correct context window logic"""
#         for mel_file in mel_files:
#             try:
#                 # Load full spectrogram [1, n_mels, time]
#                 mel_spec = self.processor.load_mel_spectrogram(mel_file)
                
#                 if mel_spec is None or mel_spec.numel() == 0:
#                     continue
                    
#                 total_frames = mel_spec.shape[-1]
                
#                 for _ in range(aug_db):
#                     for duration in self.missing_durations:
#                         # Calculate missing width in frames
#                         seconds_per_frame = self.processor.hop_length / self.processor.sr
#                         missing_width = int(duration / seconds_per_frame)
                        
#                         if missing_width >= total_frames:
#                             continue
                            
#                         # Randomly position the missing part
#                         max_start = total_frames - missing_width
#                         if max_start <= 0: continue
                        
#                         missing_start = torch.randint(0, max_start, (1,)).item()
                        
#                         # 1. Extract Raw Parts
#                         # Before: [1, n_mels, start]
#                         before_raw = mel_spec[..., :missing_start]
#                         #print(before_raw.shape[-1])
#                         # Target: [1, n_mels, width]
#                         target_raw = mel_spec[..., missing_start : missing_start + missing_width]
#                         #print(target_raw.shape[-1])
#                         # After:  [1, n_mels, end:]
#                         after_raw = mel_spec[..., missing_start + missing_width:]
#                         #print(after_raw.shape[-1])
                        
#                         # 2. Handle "Before" Context (Pad LEFT if too short)
#                         if before_raw.shape[-1] > self.max_context_width:
#                             # Take only the part adjacent to the gap (the end of 'before')
#                             before_part = before_raw[..., -self.max_context_width:]
#                         else:
#                             # Pad LEFT
#                             pad_amount = self.max_context_width - before_raw.shape[-1]
#                             before_part = F.pad(before_raw, (pad_amount, 0))
                            
#                         # 3. Handle "After" Context (Pad RIGHT if too short)
#                         if after_raw.shape[-1] > self.max_context_width:
#                             # Take only the part adjacent to the gap (the start of 'after')
#                             after_part = after_raw[..., :self.max_context_width]
#                         else:
#                             # Pad RIGHT
#                             pad_amount = self.max_context_width - after_raw.shape[-1]
#                             after_part = F.pad(after_raw, (0, pad_amount))

#                         # 4. Save to list (IMPORTANT: Convert to CPU to avoid pinning error and save VRAM)
#                         self.examples.append({
#                             'before': before_part.squeeze(0).cpu(),
#                             'after': after_part.squeeze(0).cpu(),
#                             'target': target_raw.squeeze(0).cpu(),
#                             'mel_file': str(mel_file),
#                             'duration': duration
#                         })
                        
#                         if self.test_mode:
#                             break
                        
#             except Exception as e:
#                 print(f"Error processing {mel_file}: {e}")
#                 continue
    
#     def __len__(self):
#         return len(self.examples)
    
#     def __getitem__(self, idx):
#         item = self.examples[idx]
#         return item['before'], item['after'], item['target']


import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torch.nn.functional as F


class MelInpaintingDataset(Dataset):
    """
    Dataset for Mel-spectrogram inpainting.
    Assumes mel-spectrograms are already normalized in [-1, 1].
    """

    def __init__(
        self,
        processor,              # conservé pour compatibilité, mais non utilisé
        mel_folder,
        missing_durations=(1.0, 2.0, 3.0),
        max_files=None,
        max_context_width=50,
        test_mode=False,
        aug_db=2
    ):
        self.missing_durations = missing_durations
        self.max_context_width = max_context_width
        self.test_mode = test_mode
        self.examples = []

        mel_folder = Path(mel_folder)

        # Charger UNIQUEMENT les .npy
        mel_files = sorted(mel_folder.glob("*.npy"))

        if max_files is not None:
            mel_files = mel_files[:max_files]

        print(f"Processing {len(mel_files)} mel files (.npy)")
        self._create_examples(mel_files, aug_db)

    # --------------------------------------------------
    # Create training examples
    # --------------------------------------------------
    def _create_examples(self, mel_files, aug_db):
        for mel_file in mel_files:
            try:
                mel = np.load(mel_file)
                mel = torch.from_numpy(mel).float()

                # Accept [n_mels, T] or [1, n_mels, T]
                if mel.dim() == 2:
                    mel = mel.unsqueeze(0)

                if mel.dim() != 3:
                    print(f"Skipping {mel_file}, invalid shape {mel.shape}")
                    continue

                _, n_mels, total_frames = mel.shape
                if total_frames < 2:
                    continue

                # Safety check (optional, but useful)
                if mel.min() < -1.01 or mel.max() > 1.01:
                    print(f"⚠️ Warning: {mel_file} out of [-1,1] range")

                for _ in range(aug_db):
                    for duration in self.missing_durations:

                        # Convert seconds → frames
                        # IMPORTANT: this must match how mel was generated
                        # If hop_length=512 and sr=22050 → ~0.023s/frame
                        seconds_per_frame = 512 / 22050
                        missing_width = int(duration / seconds_per_frame)

                        if missing_width <= 0 or missing_width >= total_frames:
                            continue

                        max_start = total_frames - missing_width
                        if max_start <= 0:
                            continue

                        missing_start = torch.randint(0, max_start, (1,)).item()

                        # ---------------------------
                        # Split spectrogram
                        # ---------------------------
                        before_raw = mel[..., :missing_start]
                        target_raw = mel[..., missing_start:missing_start + missing_width]
                        after_raw  = mel[..., missing_start + missing_width:]

                        # ---------------------------
                        # BEFORE context (pad LEFT)
                        # ---------------------------
                        if before_raw.shape[-1] > self.max_context_width:
                            before = before_raw[..., -self.max_context_width:]
                        else:
                            pad = self.max_context_width - before_raw.shape[-1]
                            before = F.pad(before_raw, (pad, 0))

                        # ---------------------------
                        # AFTER context (pad RIGHT)
                        # ---------------------------
                        if after_raw.shape[-1] > self.max_context_width:
                            after = after_raw[..., :self.max_context_width]
                        else:
                            pad = self.max_context_width - after_raw.shape[-1]
                            after = F.pad(after_raw, (0, pad))

                        # Remove batch dim → [n_mels, T]
                        self.examples.append({
                            "before": before.squeeze(0).cpu(),
                            "after": after.squeeze(0).cpu(),
                            "target": target_raw.squeeze(0).cpu(),
                            "mel_file": str(mel_file),
                            "duration": duration
                        })

                        if self.test_mode:
                            break

            except Exception as e:
                print(f"Error processing {mel_file}: {e}")

    # --------------------------------------------------
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return item["before"], item["after"], item["target"]
