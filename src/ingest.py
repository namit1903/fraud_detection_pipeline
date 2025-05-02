#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import pandas as pd
import numpy as np
import os


# In[3]:


RAW_DATA_PATH = '/content/drive/My Drive/fraud_detection_pipeline/data/raw/'


# In[4]:


# Function to ingest a dataset
def ingest_dataset(path: str, modality: str, expected_schema: dict) -> pd.DataFrame:
    """
    Ingest the dataset, validate schema, and return the cleaned dataframe.

    Args:
    - path (str): The file path of the dataset.
    - modality (str): The modality type ('CDR', 'IPDR', 'EDR').
    - expected_schema (dict): Expected column names and their types.

    Returns:
    - pd.DataFrame: Cleaned and validated dataframe.
    """
    # Load the dataset into a DataFrame
    df = pd.read_csv(path)

    # Step 1: Validate columns based on expected schema
    missing_cols = [col for col in expected_schema if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {modality}: {missing_cols}")

    # Step 2: Convert columns to correct data types (if necessary)
    for col, dtype in expected_schema.items():
        df[col] = df[col].astype(dtype, errors='ignore')  # Ignore errors if column type is already correct

    # Step 3: Normalize user_id field
    if modality == 'CDR':
        df['user_id'] = df['caller_id']
    elif modality == 'IPDR':
        df['user_id'] = df['user_id']
    elif modality == 'EDR':
        df['user_id'] = df['user_id']

    # Step 4: Add modality tag to distinguish datasets
    df['source_modality'] = modality

    return df


# In[5]:


# Define the expected schema for each modality
expected_cdr_schema = {
    'caller_id': str, 'callee_id': str, 'call_start': str, 'call_duration': int,
    'call_type': str, 'cell_id': str, 'location': str, 'imei': str, 'is_fraud': bool
}

expected_ipdr_schema = {
    'user_id': str, 'timestamp': str, 'domain': str, 'ip_dst': str, 'port': int,
    'protocol': str, 'duration': int, 'bytes_sent': int, 'bytes_received': int,
    'vpn_usage': bool, 'is_fraud': bool
}

expected_edr_schema = {
    'event_id': str, 'user_id': str, 'device_model': str, 'os_type': str,
    'roaming_status': bool, 'network_type': str, 'event_time': str, 'location': str,
    'event_type': str, 'is_fraud': bool
}


# In[6]:


# Load and preprocess each dataset
cdr_file_path = os.path.join(RAW_DATA_PATH, 'cdr.csv')
ipdr_file_path = os.path.join(RAW_DATA_PATH, 'ipdr.csv')
edr_file_path = os.path.join(RAW_DATA_PATH, 'edr.csv')

# Ingest each dataset
cdr_data = ingest_dataset(cdr_file_path, 'CDR', expected_cdr_schema)
ipdr_data = ingest_dataset(ipdr_file_path, 'IPDR', expected_ipdr_schema)
edr_data = ingest_dataset(edr_file_path, 'EDR', expected_edr_schema)

# Preview the data to ensure it's loaded correctly
# print(cdr_data.head())
# print(ipdr_data.head())
# print(edr_data.head())


# In[7]:


import shutil
shutil.copy("/content/drive/MyDrive/Colab Notebooks/ingest.ipynb", "/content/drive/MyDrive/fraud_detection_pipeline/src/ingest.ipynb")
# !cp /content/drive/MyDrive/Colab Notebooks/ingest.ipynb /content/drive/MyDrive/fraud_detection_pipeline/src/ingest.ipynb


# Instead of importing fucntion define here for preprocessing

# In[8]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_cdr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['call_start'] = pd.to_datetime(df['call_start'], errors='coerce')

    # Temporal patterns
    df['call_hour'] = df['call_start'].dt.hour
    df['call_day'] = df['call_start'].dt.dayofweek

    # Behavior encoding
    df['call_type'] = df['call_type'].astype(str)

    # Optional: Convert location string to lat/lon
    df[['latitude', 'longitude']] = df['location'].str.split(',', expand=True).astype(float)

    return df

def preprocess_ipdr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Temporal
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday

    # Convert domain to lowercase tokens (for future Transformer input)
    df['domain'] = df['domain'].astype(str).str.lower()

    return df

def preprocess_edr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    df['event_hour'] = df['event_time'].dt.hour
    df['event_day'] = df['event_time'].dt.weekday

    # Normalize categorical
    df['event_type'] = df['event_type'].astype(str).str.lower()
    df['network_type'] = df['network_type'].astype(str).str.lower()
    df['os_type'] = df['os_type'].astype(str).str.lower()

    return df

def standardize_numerics(dfs: list) -> list:
    scaler = StandardScaler()
    numeric_features = []

    for df in dfs:
        numeric_cols = df.select_dtypes(include=['int32', 'float32']).columns
        numeric_features.extend(numeric_cols)

    # Deduplicate features
    numeric_features = list(set(numeric_features))

    output_dfs = []
    for df in dfs:
        cols_to_scale = [col for col in numeric_features if col in df.columns]
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        output_dfs.append(df)

    return output_dfs


# In[ ]:


# !pip install import-ipynb



# In[9]:


# import sys
# sys.path.append('/content/drive/MyDrive/fraud_detection_pipeline/src')

# from preprocess import preprocess_cdr, preprocess_ipdr, preprocess_edr, standardize_numerics



# Apply preprocessing
# print(cdr_data)
cdr_processed = preprocess_cdr(cdr_data)
ipdr_processed = preprocess_ipdr(ipdr_data)
edr_processed = preprocess_edr(edr_data)
print("CDR \n",cdr_processed)
# print("IPDR \n",ipdr_processed.head())
# print("EDR \n",edr_processed.head())

# print(edr_processed.dtypes)

# Standardize
cdr_processed, ipdr_processed, edr_processed = standardize_numerics([cdr_processed, ipdr_processed, edr_processed])
# print("CDR \n",cdr_processed)
# print("IPDR \n",ipdr_processed.head())
# print("EDR \n",edr_processed.head())


# SAVE the processed dataframes
# 

# In[10]:


cdr_processed.to_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/processed/cdr_processed.csv", index=False)
ipdr_processed.to_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/processed/ipdr_processed.csv", index=False)
edr_processed.to_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/processed/edr_processed.csv", index=False)


# Merge all datasets on user_id
