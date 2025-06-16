import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_iot_data(file_path, task='multiclass'):
    """Load and preprocess IoT network intrusion dataset"""
    print("\n" + "=" * 80)
    print(f"{'IoT Network Intrusion Dataset Preprocessing':^80}")
    print("=" * 80)

    # Load data
    df = pd.read_csv(file_path)
    print(f"Original dataset size: {df.shape}")

    # Print column names for debugging
    print("Dataset columns:", df.columns.tolist())

    # 1. Handle labels first - before dropping columns
    # Find label column
    label_col = next((col for col in df.columns if 'label' in col.lower()), None)
    if label_col is None:
        raise KeyError("Label column not found in the dataset")

    # Find category column
    cat_col = next((col for col in df.columns if 'cat' in col.lower()), None)

    # Create target column
    if task == 'binary':
        # Binary classification: normal=0, anomaly=1
        df['target'] = df[label_col].map({'Normal': 0, 'Anomaly': 1})
        print("Task type: binary classification (normal vs anomaly)")
        class_names = ['Normal', 'Anomaly']
    else:
        if cat_col and cat_col in df.columns:
            # Multiclass: based on attack category
            all_cats = df[cat_col].unique()
            cat_to_id = {cat: idx for idx, cat in enumerate(all_cats)}
            df['target'] = df[cat_col].map(cat_to_id)
            print(f"Task type: Multiclass attack recognition (number of categories: {len(all_cats)})")
            class_names = list(all_cats)
        else:
            # Fall back to binary classification
            print("Warning: Category column (Cat) not found, using label column for binary classification")
            df['target'] = df[label_col].map({'Normal': 0, 'Anomaly': 1})
            print("Task type: binary classification (normal vs anomaly)")
            class_names = ['Normal', 'Anomaly']
            task = 'binary'

    # 2. Feature engineering - before dropping columns
    # Protocol type
    protocol_col = next((col for col in df.columns if 'protocol' in col.lower()), 'Protocol')
    if protocol_col in df.columns:
        df['Protocol_Type'] = df[protocol_col].apply(lambda x: 1 if x == 6 else 2 if x == 17 else 0)

    # Source port type
    src_port_col = next((col for col in df.columns if 'src_port' in col.lower()), 'Src_Port')
    if src_port_col in df.columns:
        df['Src_Port_Type'] = df[src_port_col].apply(lambda x: 1 if x < 1024 else 0)

    # Destination port type
    dst_port_col = next((col for col in df.columns if 'dst_port' in col.lower()), 'Dst_Port')
    if dst_port_col in df.columns:
        df['Dst_Port_Type'] = df[dst_port_col].apply(lambda x: 1 if x < 1024 else 0)

    # Forward/Backward ratio - add safe handling
    tot_fwd_pkts_col = next((col for col in df.columns if 'tot_fwd_pkts' in col.lower()), 'Tot_Fwd_Pkts')
    tot_bwd_pkts_col = next((col for col in df.columns if 'tot_bwd_pkts' in col.lower()), 'Tot_Bwd_Pkts')
    if tot_fwd_pkts_col in df.columns and tot_bwd_pkts_col in df.columns:
        # Avoid division by zero and infinite values
        df['Fwd_Bwd_Ratio'] = np.where(
            df[tot_bwd_pkts_col] > 0,
            df[tot_fwd_pkts_col] / df[tot_bwd_pkts_col],
            0.0
        )
        # Handle possible infinite values
        df['Fwd_Bwd_Ratio'] = df['Fwd_Bwd_Ratio'].replace([np.inf, -np.inf], 0.0)

    # Forward/Backward size ratio - add safe handling
    totlen_fwd_pkts_col = next((col for col in df.columns if 'totlen_fwd_pkts' in col.lower()), 'TotLen_Fwd_Pkts')
    totlen_bwd_pkts_col = next((col for col in df.columns if 'totlen_bwd_pkts' in col.lower()), 'TotLen_Bwd_Pkts')
    if totlen_fwd_pkts_col in df.columns and totlen_bwd_pkts_col in df.columns:
        # Avoid division by zero and infinite values
        df['Fwd_Bwd_Size_Ratio'] = np.where(
            df[totlen_bwd_pkts_col] > 0,
            df[totlen_fwd_pkts_col] / df[totlen_bwd_pkts_col],
            0.0
        )
        # Handle possible infinite values
        df['Fwd_Bwd_Size_Ratio'] = df['Fwd_Bwd_Size_Ratio'].replace([np.inf, -np.inf], 0.0)

    # Flag score - add safe handling
    fwd_psh_flags_col = next((col for col in df.columns if 'fwd_psh_flags' in col.lower()), 'Fwd_PSH_Flags')
    bwd_psh_flags_col = next((col for col in df.columns if 'bwd_psh_flags' in col.lower()), 'Bwd_PSH_Flags')
    fwd_urg_flags_col = next((col for col in df.columns if 'fwd_urg_flags' in col.lower()), 'Fwd_URG_Flags')
    if all(col in df.columns for col in [fwd_psh_flags_col, bwd_psh_flags_col, fwd_urg_flags_col]):
        df['Flag_Score'] = df[fwd_psh_flags_col] + df[bwd_psh_flags_col] * 2 + df[fwd_urg_flags_col] * 4

    # Handle timestamps
    timestamp_col = next((col for col in df.columns if 'timestamp' in col.lower()), 'Timestamp')
    if timestamp_col in df.columns:
        try:
            df['Timestamp'] = pd.to_datetime(df[timestamp_col])
            df['Hour'] = df['Timestamp'].dt.hour
            df['Minute'] = df['Timestamp'].dt.minute
        except Exception as e:
            print(f"Warning: Timestamp processing error - {e}")

    # 3. Drop unnecessary columns - after creating target and features
    drop_cols = []
    for pattern in ['flow_id', 'src_ip', 'dst_ip', 'timestamp', 'label', 'cat', 'sub_cat']:
        matches = [col for col in df.columns if pattern in col.lower() and col != 'target']
        drop_cols.extend(matches)

    # Ensure we don't drop target and feature columns
    drop_cols = [col for col in drop_cols if
                 col != 'target' and not col.startswith('Protocol_Type') and not col.startswith(
                     'Src_Port_Type') and not col.startswith('Dst_Port_Type')]

    # Print columns to be dropped
    print(f"Columns to be dropped: {drop_cols}")
    df = df.drop(drop_cols, axis=1, errors='ignore')

    # 4. Check and handle infinite values and NaN values
    print("\nChecking for infinite values and NaNs:")
    print(f"  Number of infinite values: {df.isin([np.inf, -np.inf]).sum().sum()}")
    print(f"  Number of NaN values: {df.isna().sum().sum()}")

    # Replace infinite values with 0
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Fill NaN values
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            # For numerical columns, use median to fill
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
        elif df[col].dtype == 'object':
            # For object columns, use mode to fill
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    print(f"  Number of infinite values after processing: {df.isin([np.inf, -np.inf]).sum().sum()}")
    print(f"  Number of NaN values after processing: {df.isna().sum().sum()}")

    # Extract features and labels
    if 'target' not in df.columns:
        raise KeyError("Target column 'target' was not created successfully")

    X = df.drop('target', axis=1)
    y = df['target'].values

    # Separate numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Determine categorical features
    categorical_features = []
    for col in ['Protocol_Type', 'Src_Port_Type', 'Dst_Port_Type']:
        if col in X.columns:
            categorical_features.append(col)

    # Categorical feature encoding
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Numerical feature standardization - add safe handling
    scaler = StandardScaler()
    if numerical_features:
        # Check numerical ranges
        print("\nNumerical features summary statistics:")
        print(X[numerical_features].describe())

        # Handle extreme values
        for col in numerical_features:
            # Calculate IQR
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Replace extreme values
            X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
            X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])

        # Standardize
        try:
            X[numerical_features] = scaler.fit_transform(X[numerical_features])
        except Exception as e:
            print(f"Standardization error: {e}")
            # Fall back to robust scaling
            robust_scaler = RobustScaler()
            X[numerical_features] = robust_scaler.fit_transform(X[numerical_features])

    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_classes = len(np.unique(y))
    print(f"Preprocessing completed - Feature dimension: {X_train.shape[1]}, Number of classes: {num_classes}")

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique, counts):
        cls_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
        print(f"  {cls_name}: {count} samples ({count / len(y) * 100:.2f}%)")

    return X_train.values, X_test.values, y_train, y_test, num_classes, class_names
