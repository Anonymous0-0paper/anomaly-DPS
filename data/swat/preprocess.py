import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
import pytz

def load_and_preprocess_data(file_path):
    print("Reading Excel file...")
    # Read the entire file to check the structure
    full_df = pd.read_excel(file_path, header=None)
    print(f"Total rows in file: {len(full_df)}")
    # Print the first 5 rows to check the structure
    print("Structure of the first 5 rows:")
    for i in range(min(5, len(full_df))):
        print(f"Row {i}: {list(full_df.iloc[i, :5])}...")

    # Identify the header row - find the row containing "timestamp"
    header_row = None
    for i in range(min(5, len(full_df))):
        if any("timestamp" in str(cell).lower() for cell in full_df.iloc[i]):
            header_row = i
            break
    if header_row is None:
        raise ValueError("No header row containing 'timestamp' found")
    print(f"Header found at row {header_row}")

    # Set column names - use the found header row and the row above it
    header_df = full_df.iloc[header_row - 1:header_row + 1, :]
    column_names = []
    for col in range(header_df.shape[1]):
        first_part = str(header_df.iloc[0, col]) if not pd.isna(header_df.iloc[0, col]) else ""
        second_part = str(header_df.iloc[1, col]) if not pd.isna(header_df.iloc[1, col]) else ""
        if first_part and second_part:
            col_name = f"{first_part} - {second_part}"
        elif first_part:
            col_name = first_part
        elif second_part:
            col_name = second_part
        else:
            col_name = f"Column_{col + 1}"
        column_names.append(col_name)

    # Read the actual data (skip header rows)
    df = full_df.iloc[header_row + 1:, :].copy()
    df.columns = column_names
    df = df.reset_index(drop=True)
    print(f"Number of data rows: {len(df)}")
    print("First 5 column names:", df.columns[:5].tolist())

    # Find timestamp column
    timestamp_cols = [col for col in df.columns if "time" in col.lower() or "gmt" in col.lower()]
    if not timestamp_cols:
        print("Warning: No explicit timestamp column found, trying to use the first column")
        timestamp_col = df.columns[0]
    else:
        timestamp_col = timestamp_cols[0]
    print(f"Using timestamp column: {timestamp_col}")
    df = df.rename(columns={timestamp_col: "timestamp"})

    # Print the timestamp of the first record
    first_timestamp = df.iloc[0]['timestamp']
    print(f"First record time: {first_timestamp} (type: {type(first_timestamp)})")

    # Convert time format
    try:
        # Try automatic conversion
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    except Exception as e:
        print(f"Automatic conversion failed: {e}")
        # Try ISO format
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
        except Exception as e2:
            print(f"ISO format conversion failed: {e2}")
            # Try custom format
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S%z', utc=True)
            except Exception as e3:
                print(f"Custom format conversion failed: {e3}")
                # Finally try string processing
                print("Attempting string processing...")
                try:
                    df['timestamp'] = pd.to_datetime(
                        df['timestamp'].str.replace('Z', '+00:00'),
                        utc=True
                    )
                except Exception as e4:
                    print(f"String processing failed: {e4}")
                    print("Using fallback method: direct conversion and adding timezone")
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    print(f"Converted time: {df.iloc[0]['timestamp']}")
    return df

def add_labels(df):
    print("\nAdding attack labels...")
    # Define attack periods (GMT+8)
    attacks_gmt8 = [
        ('2019-07-20 15:08:46', '2019-07-20 15:10:31', 'FIT401 Spoof'),
        ('2019-07-20 15:15:00', '2019-07-20 15:19:32', 'LIT301 Spoof'),
        ('2019-07-20 15:26:57', '2019-07-20 15:30:48', 'P601 Switch'),
        ('2019-07-20 15:38:50', '2019-07-20 15:46:20', 'Multi-point'),
        ('2019-07-20 15:54:00', '2019-07-20 15:56:00', 'MV501 Switch'),
        ('2019-07-20 16:02:56', '2019-07-20 16:16:18', 'P301 Switch')
    ]

    # Convert attack times to GMT+0 and add timezone information
    utc = pytz.UTC
    attacks_gmt0 = []
    for start, end, name in attacks_gmt8:
        start_dt = datetime.strptime(start, '%Y-%m-%d %H:%M:%S') - timedelta(hours=8)
        end_dt = datetime.strptime(end, '%Y-%m-%d %H:%M:%S') - timedelta(hours=8)
        # Add timezone information
        start_dt = utc.localize(start_dt)
        end_dt = utc.localize(end_dt)
        attacks_gmt0.append((start_dt, end_dt, name))

    # Create label column
    df['Normal/Attack'] = 'Normal'  # Default all data as normal
    # Mark attack periods
    for start, end, name in attacks_gmt0:
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        attack_count = mask.sum()
        print(f"Attack '{name}' period: {start} to {end} - {attack_count} data points")
        df.loc[mask, 'Normal/Attack'] = 'Attack'
    return df, attacks_gmt0

def create_train_set(df):
    print("\nCreating training set...")
    # According to the PDF document, extract data from the safe period
    # Normal operation time (GMT+8): 12:35 PM to 2:50 PM → GMT+0: 04:35 AM to 06:50 AM
    utc = pytz.UTC
    train_start = utc.localize(datetime(2019, 7, 20, 4, 35, 0))  # 04:35 GMT+0
    train_end = utc.localize(datetime(2019, 7, 20, 6, 50, 0))  # 06:50 GMT+0

    # Extract training set
    train_df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)]

    # Ensure training set contains only normal data
    if not train_df['Normal/Attack'].eq('Normal').all():
        attack_count = train_df[train_df['Normal/Attack'] != 'Normal'].shape[0]
        print(f"Warning: Training set contains {attack_count} attack data points!")
    else:
        print("Training set verification: All data is normal")

    # Remove label column (not needed for unsupervised learning)
    train_df = train_df.drop(columns=['Normal/Attack'])
    print(f"Training set size: {len(train_df)} rows")
    return train_df

def main():
    input_file = './initialData/SWaT_dataset_Jul 19 v2.xlsx'
    output_dir = 'SWaT_dataset_split'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 1. Load and process data
    df = load_and_preprocess_data(input_file)

    # 2. Add labels
    df, attacks_gmt0 = add_labels(df)

    # 3. Create training set (pure normal data)
    train_df = create_train_set(df)

    # 4. Create test set (complete data)
    test_df = df.copy()
    print(f"Test set size: {len(test_df)} rows")

    # 5. Save datasets
    train_file = os.path.join(output_dir, 'SWaT_train.csv')
    test_file = os.path.join(output_dir, 'SWaT_test.csv')
    print(f"\nSaving training set to: {train_file}")
    train_df.to_csv(train_file, index=False)
    print(f"Saving test set to: {test_file}")
    test_df.to_csv(test_file, index=False)

    # 6. Print statistics
    total_points = len(df)
    normal_points = df[df['Normal/Attack'] == 'Normal'].shape[0]
    attack_points = df[df['Normal/Attack'] == 'Attack'].shape[0]
    print("\n===== Dataset Statistics =====")
    print(f"Total data points: {total_points}")
    print(f"Training set size (pure normal data): {len(train_df)} ({len(train_df) / total_points:.1%})")
    print(f"Test set size (complete data): {len(test_df)}")
    print(f"Normal data proportion: {normal_points} ({normal_points / total_points:.1%})")
    print(f"Attack data proportion: {attack_points} ({attack_points / total_points:.1%})")
    print("\n===== Attack Periods (GMT+0) =====")
    for i, (start, end, name) in enumerate(attacks_gmt0, 1):
        attack_points = \
            df[(df['timestamp'] >= start) & (df['timestamp'] <= end) & (df['Normal/Attack'] == 'Attack')].shape[0]
        print(f"Attack {i}: {name}")
        print(f"  Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End: {end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {(end - start).seconds} seconds")
        print(f"  Number of data points: {attack_points}")
    print("\nProcessing completed!")

if __name__ == "__main__":
    main()
