import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_kdd_data(file_path):
    """Load and preprocess KDD99 dataset"""
    print("\n" + "=" * 80)
    print(f"{'KDD CUP 1999 Dataset Preprocessing':^80}")
    print("=" * 80)

    # Define column names
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]

    # Load data
    df = pd.read_csv(file_path, header=None, names=columns)
    print(f"Original dataset size: {df.shape}")

    # Simplify labels into 5 major categories
    attack_mapping = {
        'normal.': 0,
        'neptune.': 1, 'back.': 1, 'land.': 1, 'pod.': 1, 'smurf.': 1, 'teardrop.': 1,  # DoS
        'portsweep.': 2, 'satan.': 2, 'ipsweep.': 2, 'nmap.': 2,  # Probe
        'ftp_write.': 3, 'guess_passwd.': 3, 'imap.': 3, 'multihop.': 3, 'phf.': 3, 'spy.': 3, 'warezclient.': 3,
        'warezmaster.': 3,  # R2L
        'buffer_overflow.': 4, 'loadmodule.': 4, 'perl.': 4, 'rootkit.': 4  # U2R
    }

    # Map labels to category IDs
    df['target'] = df['label'].map(attack_mapping)

    # Ensure all labels are mapped
    if df['target'].isnull().any():
        unmapped_labels = df[df['target'].isnull()]['label'].unique()
        print(f"Warning: Unmapped labels: {unmapped_labels}")
        df['target'] = df['target'].fillna(5)  # Map unmapped labels to the 6th category

    # Extract features and labels
    X = df.drop(['label', 'target'], axis=1)
    y = df['target'].values

    # Separate numerical and categorical features
    numerical_features = [
        'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]

    categorical_features = ['protocol_type', 'service', 'flag']

    # Categorical feature encoding
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Numerical feature standardization
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Split training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_classes = len(np.unique(y))
    class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R', 'Other']
    print(f"Preprocessing completed - Feature dimension: {X_train.shape[1]}, Number of classes: {num_classes}")

    return X_train.values, X_test.values, y_train, y_test, num_classes, class_names
