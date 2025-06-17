from .kddcup.data_loader import load_and_preprocess_kdd_data
from .IOTID20.data_loader import load_and_preprocess_iot_data
from .msl_smap.data_loader import load_msl_smap_data
from .smd.data_loader import load_smd_data
from .swat.data_loader import load_swat_data
from .unsw.data_loader import load_unsw_data
from .wustl.data_loader import load_iiot_data
from .wadi.data_loader import load_wadi_data
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
    elif dataset_name == 'swat':
        version = kwargs.get('version', 'v2')
        return load_swat_data(file_path, version=version)
    elif dataset_name == 'unsw':
        return load_unsw_data(file_path)
    elif dataset_name == 'iiot':
        return load_iiot_data(file_path)
    elif dataset_name == 'wadi':
        version = kwargs.get('version', 'A2')
        return load_wadi_data(file_path, version=version)
    else:
        raise ValueError(f"Unsupported database type: {dataset_name}")
