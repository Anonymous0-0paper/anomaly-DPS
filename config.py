# ====================== Global Configuration ======================
import torch

DATASET_CHOICE = 'smd'  # 'kdd' or 'iot'
TASK_TYPE = 'binary'  # Only valid for IoT dataset: 'binary' or 'multiclass'

# Add SMD machine ID configuration
SMD_MACHINE_ID = 'machine-1-1'  # Default machine ID for SMD dataset

# Dataset paths
DATA_PATHS = {
    'kdd': './data/kddcup/kddcup.data',
    'iot': './data/IOTID20/IoT Network Intrusion Dataset.csv',
    'msl': './data/msl_smap/initialData',
    'smap': './data/msl_smap/initialData',
    'smd': './data/smd/initialData'
}

# Add MSL/SMAP specific configuration
MSL_SMAP_CONFIG = {
    'window_size': 100,  # Larger window is more suitable for time series anomaly detection
    'stride': 20,
    'd_model': 64,       # Smaller model size
    'e_layers': 2        # Shallower encoder layers
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

# Time Series Specific Settings
TS_NORMALIZE = True  # Whether to normalize the time series
