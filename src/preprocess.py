#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[6]:


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



# In[7]:


def preprocess_ipdr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Temporal
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday

    # Convert domain to lowercase tokens (for future Transformer input)
    df['domain'] = df['domain'].astype(str).str.lower()

    return df



# In[8]:


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


# In[9]:


def standardize_numerics(dfs: list) -> list:
    """
    Standardize numerical columns across all datasets.
    """
    scaler = StandardScaler()
    output_dfs = []

    for df in dfs:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        output_dfs.append(df)

    return output_dfs


# In[10]:


get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/fraud_detection_pipeline/src/preprocess.ipynb"')


# In[ ]:


import shutil
shutil.copy("/content/drive/MyDrive/Colab Notebooks/preprocess.ipynb", "/content/drive/MyDrive/fraud_detection_pipeline/src/preprocess.ipynb")
# !cp /content/drive/MyDrive/Colab Notebooks/ingest.ipynb /content/drive/MyDrive/fraud_detection_pipeline/src/ingest.ipynb

