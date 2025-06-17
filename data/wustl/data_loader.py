import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def load_iiot_data(file_path):
    """Load WUSTL-IIoT-2021 dataset"""
    df = pd.read_csv(file_path)

    # Key feature engineering
    df['FlowDirection'] = np.where(df['DstPkts'] == 0, 1, 0)  # Unidirectional traffic marking
    df['PortAnomaly'] = np.where(df['Dport'].isin([80, 443]), 1, 0)  # Anomalous port access

    # Time feature conversion
    df['StartTime'] = pd.to_datetime(df['StartTime'])
    df['Hour'] = df['StartTime'].dt.hour
    df['Minute'] = df['StartTime'].dt.minute

    # Drop redundant columns
    drop_cols = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'Traffic']
    df = df.drop(drop_cols, axis=1, errors='ignore')

    # Industrial protocol feature encoding
    protocol_map = {'tcp': 0, 'udp': 1, 'icmp': 2}
    df['Proto'] = df['Proto'].map(protocol_map).fillna(3)  # Unknown protocol = 3

    # Handle special values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Separate features and labels
    X = df.drop('Target', axis=1)
    y = df['Target'].values

    # Robust standardization (handling extreme values)
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    # Industrial scenarios do not require splitting into training and test sets (original data is already time-ordered)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, 2, ['Normal', 'Attack']
