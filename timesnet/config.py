import torch

# Model configuration
MODEL_CONFIG = {
    'seq_len': 100,
    'stride': 1,
    'epochs': 1,
    'batch_size': 32,
    'inference_batch_size': 10000,  # New: Inference batch size
    'lr': 1e-4,
    'd_model': 64,
    'd_ff': 64,
    'dropout': 0.1,
    'top_k': 5,
    'num_kernels': 6
}

# Dataset configuration
DATA_CONFIG = [
    # NASA MSL dataset
    {
        'name': 'NASA_MSL',
        'type': 'nasa',
        'params': {
            'root_dir': '../data/msl_smap/initialData',
            'dataset': 'MSL',
            'process_all': True,
            'test_channels': None,
            'max_channels': None,
            'exclude_channels': []
        }
    },
    # NASA SMAP dataset
    {
        'name': 'NASA_SMAP',
        'type': 'nasa',
        'params': {
            'root_dir': '../data/msl_smap/initialData',
            'dataset': 'SMAP',
            'process_all': True,
            'test_channels': None,
            'max_channels': None,
            'exclude_channels': []
        }
    },
    # SMD dataset
    {
        'name': 'SMD',
        'type': 'smd',
        'params': {
            'root_dir': '../data/smd/initialData',
            'process_all': True,
            'test_machines': None,
            'max_machines': None,
            'exclude_machines': []
        }
    },
    # SWAT dataset
    {
        'name': 'SWAT',
        'type': 'swat',
        'params': {
            'root_dir': '../data/swat/SWaT_dataset_split',
            'train_file': 'SWaT_train.csv',
            'test_file': 'SWaT_test.csv',
            'label_column': 'Normal/Attack',
            'feature_columns': None,
            'normal_value': 0,
            'attack_value': 1
        }
    },
    # WADI dataset
    {
        'name': 'WADI',
        'type': 'wadi',
        'params': {
            'root_dir': '../data/wadi/processed_data',
            'train_file': 'train.csv',
            'test_file': 'test.csv',
            'label_column': 'attack_label'
        }
    }
]

# Experiment configuration
EXPERIMENT_CONFIG = {
    'save_plots': True,
    'plot_dir': 'results/plots',
    'results_file': 'results/summary.csv',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_plot_points': 2000,  # More conservative value
    'log_level': 'DEBUG',  # Add log level configuration

    # New data sampling configuration
    'data_sampling': {
        'enable': True,  # Whether to enable sampling
        'train_sample_ratio': 0.1,  # Training data sampling ratio
        'test_sample_ratio': 0.1,  # Test data sampling ratio
        'max_train_points': 5000,  # Maximum number of training data points
        'max_test_points': 10000  # Maximum number of test data points
    }
}
