import torch

class DatasetConfig:
    """Base class for dataset configuration"""

    def __init__(self, name, path, window_size, stride, **kwargs):
        self.name = name
        self.path = path
        self.window_size = window_size
        self.stride = stride
        self.params = kwargs

class TimeSeriesConfig(DatasetConfig):
    """Specialized configuration for time series datasets"""

    def __init__(self, name, path, window_size, stride, d_model, e_layers, **kwargs):
        super().__init__(name, path, window_size, stride, **kwargs)
        self.d_model = d_model
        self.e_layers = e_layers

class NetworkConfig(DatasetConfig):
    """Specialized configuration for network datasets"""

    def __init__(self, name, path, window_size, stride, task_type, d_model=None, e_layers=None, **kwargs):
        super().__init__(name, path, window_size, stride, **kwargs)
        self.task_type = task_type
        self.d_model = d_model
        self.e_layers = e_layers

# ====================== GLOBAL SETTINGS ======================
class GlobalConfig:
    # Dataset selection (choose from available datasets)
    DATASET_CHOICE = ('swat')

    # Sequence parameters (default values, will be overridden by specific datasets)
    WINDOW_SIZE = 10
    STRIDE = 5

    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 0.001

    # Model hyperparameters (default values, will be overridden by specific datasets)
    D_MODEL = 128
    N_HEADS = 8
    E_LAYERS = 3
    DROPOUT = 0.2

    # Output settings
    SAVE_MODEL = True
    SAVE_PLOTS = False

    # Time series settings
    TS_NORMALIZE = True

    # Device settings (automatically detected at runtime)
    @property
    def DEVICE(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        return "cpu"

# ====================== DATASET CONFIGURATIONS ======================
class DatasetRegistry:
    """Dataset configuration registry"""
    DATASETS = {
        # Time series datasets
        'msl': TimeSeriesConfig(
            name='msl',
            path='data/msl_smap/initialData',
            window_size=100,
            stride=20,
            d_model=64,
            e_layers=2,
            task_type='binary'
        ),
        'smap': TimeSeriesConfig(
            name='smap',
            path='data/msl_smap/initialData',
            window_size=100,
            stride=20,
            d_model=64,
            e_layers=2,
            task_type='binary',
        ),
        'smd': TimeSeriesConfig(
            name='smd',
            path='data/smd/initialData',
            window_size=100,
            stride=20,
            d_model=128,
            e_layers=3,
            machine_id='machine-1-1',
            task_type='binary',
        ),
        'swat': TimeSeriesConfig(
            name='swat',
            path='data/swat/initialData/SWaT_dataset_Jul 19 v2.xlsx',
            window_size=120,
            stride=30,
            d_model=128,
            e_layers=3,
            data_version='v2',
            task_type='binary',
        ),
        'wadi': TimeSeriesConfig(
            name='wadi',
            path='data/wadi/initialData/WADI.A2_19 Nov 2019',
            window_size=120,
            stride=30,
            d_model=128,
            e_layers=3,
            version='A2',
            task_type='binary',
        ),

        # Network datasets
        'kdd': NetworkConfig(
            name='kdd',
            path='data/kddcup/kddcup.data_loader',
            window_size=1,
            stride=1,
            task_type='multiclass',
            d_model=128,  # Add model parameters
            e_layers=3  # Add number of encoder layers
        ),
        'iot': NetworkConfig(
            name='iot',
            path='data/IOTID20/IoT Network Intrusion Dataset.csv',
            window_size=1,
            stride=1,
            task_type='multiclass',
            d_model=128,
            e_layers=3
        ),
        'unsw': NetworkConfig(
            name='unsw',
            path='data/unsw',
            window_size=1,
            stride=1,
            task_type='multiclass',
            d_model=128,
            e_layers=3
        ),
        'iiot': NetworkConfig(
            name='iiot',
            path='data/wustl/wustl_iiot_2021.csv',
            window_size=5,
            stride=1,
            task_type='multiclass',
            d_model=128,
            e_layers=3
        )
    }

    @classmethod
    def get_config(cls, dataset_name):
        """Get configuration for the specified dataset"""
        config = cls.DATASETS.get(dataset_name)
        if not config:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        return config

    @classmethod
    def get_current_config(cls):
        """Get configuration for the currently selected dataset"""
        return cls.get_config(GlobalConfig.DATASET_CHOICE)

# ====================== CONFIG ACCESSOR ======================
class Config:
    """Unified configuration access interface"""

    @property
    def global_config(self):
        return GlobalConfig()

    @property
    def dataset_config(self):
        return DatasetRegistry.get_current_config()

    def __getattr__(self, name):
        """Priority return dataset-specific configuration, then global configuration"""
        # Special handling for model parameters
        model_params = ['d_model', 'e_layers', 'n_heads', 'dropout']
        if name in model_params:
            # Priority from dataset configuration
            ds_config = self.dataset_config
            if hasattr(ds_config, name) and getattr(ds_config, name) is not None:
                return getattr(ds_config, name)
            # Then from global configuration
            elif hasattr(self.global_config, name):
                return getattr(self.global_config, name)
            # Finally from params
            elif hasattr(ds_config, 'params') and name in ds_config.params:
                return ds_config.params[name]

        # Regular attribute access
        ds_config = self.dataset_config
        if hasattr(ds_config, name):
            return getattr(ds_config, name)
        if hasattr(self.global_config, name):
            return getattr(self.global_config, name)
        if hasattr(ds_config, 'params') and name in ds_config.params:
            return ds_config.params[name]

        raise AttributeError(f"Config has no attribute '{name}'")

# Create configuration instance
CONFIG = Config()
