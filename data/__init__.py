from .kddcup.data_loader import load_and_preprocess_kdd_data
from .IOTID20.data_loader import load_and_preprocess_iot_data
from .base_loader import AnomalyDetectionDatasets

def get_data_loader(dataset_name, file_path, task_type='multiclass'):
    """Get the loader for the specified dataset"""
    if dataset_name == 'kdd':
        return load_and_preprocess_kdd_data(file_path)
    elif dataset_name == 'iot':
        return load_and_preprocess_iot_data(file_path, task=task_type)
    else:
        raise ValueError(f"Unsupported database type: {dataset_name}")
