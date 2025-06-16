# ====================== Global Configuration ======================
import torch

DATASET_CHOICE = 'iot'  # 'kdd' or 'iot'
TASK_TYPE = 'multiclass'  # Only valid for IoT dataset: 'binary' or 'multiclass'

# Dataset paths
DATA_PATHS = {
    'kdd': './data/kddcup/kddcup.data',
    'iot': './data/IOTID20/IoT Network Intrusion Dataset.csv'
}

# Sequence parameters
WINDOW_SIZE = 10  # Time window size
STRIDE = 5  # Sliding step size

# Training parameters
BATCH_SIZE = 64  # Batch size
EPOCHS = 3  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate

# Model hyperparameters
D_MODEL = 128  # Model dimension
N_HEADS = 8  # Number of attention heads
E_LAYERS = 3  # Number of encoder layers
DROPOUT = 0.2  # Dropout probability

# Output settings
SAVE_MODEL = True  # Whether to save the full model
SAVE_PLOTS = False  # Whether to save plots

# Device settings (auto-detect)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon
else:
    DEVICE = "cpu"
