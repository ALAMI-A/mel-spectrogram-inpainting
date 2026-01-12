import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'learning_rates' in history:
        axes[1].plot(history['learning_rates'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_mel_comparison(original, inpainted, missing_info, sr=22050, 
                        hop_length=512, save_path=None):
    """Plot comparison between original and inpainted Mel spectrograms"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Convert to numpy for plotting
    if hasattr(original, 'cpu'):
        original = original.squeeze().cpu().numpy()
    if hasattr(inpainted, 'cpu'):
        inpainted = inpainted.squeeze().cpu().numpy()
    
    # Plot original
    img = librosa.display.specshow(
        original, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel', ax=axes[0]
    )
    axes[0].set_title('Original Mel Spectrogram')
    
    # Plot inpainted
    img = librosa.display.specshow(
        inpainted, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel', ax=axes[1]
    )
    axes[1].set_title('Inpainted Mel Spectrogram')
    
    plt.colorbar(img, ax=axes, format='%+2.0f dB')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig