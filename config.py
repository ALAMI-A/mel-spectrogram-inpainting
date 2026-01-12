import torch

class Config:
    """Central configuration for audio inpainting system"""
    
    # Audio processing
    SAMPLE_RATE = 22050
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    F_MIN = 20
    F_MAX = 8000
    
    # Model
    MODEL_TYPE = 'adaptive_unet' #'fc_baseline' # 'adaptive_unet'  # 'adaptive_unet', 'simple_unet', 'context'
    HIDDEN_CHANNELS = 64
    
    # Training
    BATCH_SIZE = 10
    LEARNING_RATE = 1e-4 # 1e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    CHECKPOINT_FREQ = 10
    AUG_DB = 5 #10
    
    # Dataset
    MISSING_DURATIONS = [3.0] #[1.0, 2.0, 3.0]
    MAX_CONTEXT_WIDTH = 504 #1024 #504
    MAX_FILES = 500
    
    SECONDS_PER_FRAME = HOP_LENGTH / SAMPLE_RATE  # ≈ 0.02322
    MAX_MISSING_WIDTH = int(MISSING_DURATIONS[0] / SECONDS_PER_FRAME)

    # Paths
    CHECKPOINT_DIR = 'checkpoints'
    OUTPUT_DIR = 'outputs'
    DATASET_DIR = 'dataset'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def get_processor_config(cls):
        """Get processor configuration"""
        return {
            'sr': cls.SAMPLE_RATE,
            'n_mels': cls.N_MELS,
            'n_fft': cls.N_FFT,
            'hop_length': cls.HOP_LENGTH,
            'f_min': cls.F_MIN,
            'f_max': cls.F_MAX,
            'device': cls.DEVICE
        }
    
    @classmethod
    def get_model_config(cls):
        """Get model configuration"""
        return {
            'model_type': cls.MODEL_TYPE,
            'n_mels': cls.N_MELS,
            'hidden_channels': cls.HIDDEN_CHANNELS,
            'max_context_width': cls.MAX_CONTEXT_WIDTH,
            'max_missing_width': cls.MAX_MISSING_WIDTH,
            'device': cls.DEVICE
        }