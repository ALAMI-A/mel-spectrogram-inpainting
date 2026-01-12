import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from processors.audio_processor import AudioMelProcessor
from models.inpainter import MelSpectrogramInpainter
from datasets.mel_dataset import MelInpaintingDataset
from training.trainer import Trainer

def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("Audio Inpainting System")
    print("=" * 60)
    
    # 1. Initialize components
    print("\n1. Initializing components...")
    processor = AudioMelProcessor(**Config.get_processor_config())
    inpainter = MelSpectrogramInpainter(**Config.get_model_config())
    trainer = Trainer(inpainter, device=Config.DEVICE)
    
    # 2. Create datasets
    print("\n2. Creating datasets...")
    
    # Test mode first
    print("\nPhase 1: Test mode (checking data integrity)")
    test_dataset = MelInpaintingDataset(
        processor=processor,
        mel_folder=Config.DATASET_DIR + "/mel_spectres",
        missing_durations=[2.0],
        max_files=500,
        max_context_width=Config.MAX_CONTEXT_WIDTH,
        test_mode=True
    )
    
    if len(test_dataset) == 0:
        print("✗ No valid examples created. Check your data.")
        return
    
    print(f"✓ Test dataset created with {len(test_dataset)} examples")
    
    # Full training dataset
    print("\nPhase 2: Creating full training dataset")
    train_dataset = MelInpaintingDataset(
        processor=processor,
        mel_folder=Config.DATASET_DIR + "/mel_spectres",
        missing_durations=Config.MISSING_DURATIONS,
        max_files=Config.MAX_FILES,
        max_context_width=Config.MAX_CONTEXT_WIDTH,
        test_mode=False,
        aug_db=Config.AUG_DB
    )
    
    if len(train_dataset) == 0:
        print("✗ No training examples created.")
        return
    
    print(f"✓ Full dataset created with {len(train_dataset)} examples")
    
    # Split into train/validation
    train_size = int(0.80 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_split = torch.utils.data.Subset(train_dataset, range(train_size))
    val_split = torch.utils.data.Subset(train_dataset, range(train_size, train_size + val_size))
    
    print(f"  Training split: {len(train_split)} examples")
    print(f"  Validation split: {len(val_split)} examples")
    
    # 3. Train model
    print("\n3. Training model...")
    history = trainer.train(
        train_dataset=train_split,
        val_dataset=val_split,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        checkpoint_freq=Config.CHECKPOINT_FREQ
    )
    
    # 4. Test with a sample
    print("\n4. Testing trained model...")
    if val_split and len(val_split) > 0:
        # Get item (now returns 3 separate tensors)
        before, after, target = val_split[0]
        
        # Add batch and channel dims if missing (Dataset removed them)
        # Dataset returns [n_mels, T]
        before = before.unsqueeze(0).unsqueeze(0).to(Config.DEVICE) # [1, 1, 128, T]
        after = after.unsqueeze(0).unsqueeze(0).to(Config.DEVICE)   # [1, 1, 128, T]
        target = target.unsqueeze(0).unsqueeze(0).to(Config.DEVICE) # [1, 1, 128, T]
        
        loss, pred = trainer.model.validate_step(before, after, target)
        print(f"Sample test loss: {loss:.6f}")
        
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()