from .kddcup.data_loader import load_and_preprocess_kdd_data
from .IOTID20.data_loader import load_and_preprocess_iot_data
from .msl_smap.data_loader import load_msl_smap_data
from .smd.data_loader import load_smd_data
from .base_loader import AnomalyDetectionDatasets

def get_data_loader(dataset_name, file_path, task_type='multiclass', **kwargs):
    """Get the loader for the specified dataset"""
    if dataset_name == 'kdd':
        return load_and_preprocess_kdd_data(file_path)
    elif dataset_name == 'iot':
        return load_and_preprocess_iot_data(file_path, task=task_type)
    elif dataset_name == 'msl':
        return load_msl_smap_data(file_path, 'msl')
    elif dataset_name == 'smap':
        return load_msl_smap_data(file_path, 'smap')
    elif dataset_name == 'smd':
        # Optional parameter to specify machine ID
        machine_id = kwargs.get('machine_id', 'machine-1-1')
        return load_smd_data(file_path, machine_id)
    else:
        raise ValueError(f"Unsupported database type: {dataset_name}")
