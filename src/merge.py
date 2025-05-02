#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd

def prepare_sequential_dataset(cdr_df, ipdr_df, edr_df) -> pd.DataFrame:
    # Tag modalities for tracking after merge
    cdr_df['modality'] = 'cdr'
    ipdr_df['modality'] = 'ipdr'
    edr_df['modality'] = 'edr'

    # Normalize time columns
    cdr_df['timestamp'] = pd.to_datetime(cdr_df['call_start'], errors='coerce')
    ipdr_df['timestamp'] = pd.to_datetime(ipdr_df['timestamp'], errors='coerce')
    edr_df['timestamp'] = pd.to_datetime(edr_df['event_time'], errors='coerce')

    # Align column names for stacking
    common_cols = ['user_id', 'timestamp', 'modality', 'is_fraud']
    cdr_cols = common_cols + [col for col in cdr_df.columns if col not in common_cols]
    ipdr_cols = common_cols + [col for col in ipdr_df.columns if col not in common_cols]
    edr_cols = common_cols + [col for col in edr_df.columns if col not in common_cols]

    cdr_df = cdr_df[cdr_cols]
    ipdr_df = ipdr_df[ipdr_cols]
    edr_df = edr_df[edr_cols]

    # Combine all into one large timeline of actions
    all_data = pd.concat([cdr_df, ipdr_df, edr_df], ignore_index=True)

    # Sort by user and timestamp
    all_data.sort_values(by=['user_id', 'timestamp'], inplace=True)

    return all_data


# IMPORT AND MERGE DATA
# 

# In[23]:


cdr_processed = pd.read_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/processed/cdr_processed.csv")
ipdr_processed = pd.read_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/processed/ipdr_processed.csv")
edr_processed = pd.read_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/processed/edr_processed.csv")
# Merge preprocessed and standardized data
# print(cdr_processed)
merged_df = prepare_sequential_dataset(cdr_processed, ipdr_processed, edr_processed)

# # See merged shape and a sample
# print(merged_df.shape)
merged_df.head()


# handling /preprocessing merged data

# In[31]:


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values by filling or dropping based on column importance.
    """
    # Fill non-critical missing values with 0 or a default value
    df.fillna({
        'call_hour': 0,
        'call_day': 0,
        'domain': 'unknown',
        'vpn_usage': False,
        'event_type': 'unknown',
        'caller_id': 'unknown',  # Fill missing caller_id
        'callee_id': 'unknown',  # Fill missing callee_id
        'call_duration': 0,      # Fill missing call_duration
        'call_type': 'unknown',  # Fill missing call_type
        'cell_id': 'unknown',    # Fill missing cell_id
        'event_type': 'unknown', # Fill missing event_type
        'device_model': 'unknown', # Fill missing device_model
        'os_type': 'unknown',    # Fill missing os_type
        'network_type': 'unknown' # Fill missing network_type
    }, inplace=True)

    # Drop rows with missing critical columns like 'is_fraud', 'user_id', or 'timestamp'
    df.dropna(subset=['is_fraud', 'user_id', 'timestamp'], inplace=True)

    return df

def sort_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that the dataframe is sorted by user_id and timestamp to maintain sequences.
    """
    df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    return df


# create user sequence

# In[32]:


merged_df=handle_missing_values(merged_df)
merged_df=sort_by_timestamp(merged_df)
# See merged shape and a sample
print(merged_df.shape)
merged_df.head()


# preprocessing

# In[ ]:





# In[33]:


# One-Hot Encoding for categorical columns
merged_df = pd.get_dummies(merged_df, columns=['call_type', 'protocol', 'event_type', 'network_type'], drop_first=True)

# Check the updated dataframe
print(merged_df.head())


# In[34]:


from sklearn.preprocessing import StandardScaler

# Define numerical columns to scale
numerical_columns = ['call_duration', 'bytes_sent', 'bytes_received', 'duration']

# Initialize the scaler
scaler = StandardScaler()

# Scale the numerical columns
merged_df[numerical_columns] = scaler.fit_transform(merged_df[numerical_columns])

# Check the scaled data
# print(merged_df[numerical_columns].head())
#


# 

# In[35]:


# Extract hour and weekday from timestamps
merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
merged_df['hour'] = merged_df['timestamp'].dt.hour
merged_df['weekday'] = merged_df['timestamp'].dt.weekday

# Aggregate features: Count of events per user
event_count_per_user = merged_df.groupby('user_id').size().reset_index(name='event_count')
merged_df = merged_df.merge(event_count_per_user, on='user_id', how='left')

# Check the aggregated features
# print(merged_df[['user_id', 'event_count', 'hour', 'weekday']].head())
# print(merged_df.head())



# In[36]:


# Check for missing values after the preprocessing steps
# print(merged_df.isnull().sum())

# Print the first few rows of the processed dataframe
print(merged_df.head())


# creating user sequence

# In[39]:


# Step 1: Convert the 'timestamp' column to datetime and sort the data by 'user_id' and 'timestamp'
merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
merged_df = merged_df.sort_values(by=['user_id', 'timestamp'])

# Step 2: Define the columns that represent actions or events for fraud detection
event_columns = ['timestamp', 'modality', 'is_fraud', 'protocol_UDP',
                 'event_type_network drop', 'event_type_reboot', 'event_type_roaming',
                 'event_type_sim switch', 'event_type_unknown', 'network_type_lte',
                 'network_type_wifi']

# Step 3: Create a sequence of actions/events for each 'user_id'
# We will include timestamp, modality, and fraud-related columns in the sequence to capture patterns
merged_df['user_sequence'] = merged_df[event_columns].apply(
    lambda row: ' '.join(row.astype(str)), axis=1)

# Step 4: Group by 'user_id' and combine the sequences into one per user
user_sequences = merged_df.groupby('user_id')['user_sequence'].apply(lambda x: ' '.join(x)).reset_index()

# Step 5: Display the final user_sequences dataframe (which has the sequences for each user)
print(user_sequences.head())


# In[11]:


# Assume df is your final processed or merged dataframe
merged_df.to_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/processed/merged_data.csv", index=False)


# CONVERT TO .py file
# 

# In[12]:


get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/fraud_detection_pipeline/src/merge.ipynb"')


# In[ ]:





# In[13]:


import shutil
shutil.copy("/content/drive/MyDrive/Colab Notebooks/merge.ipynb", "/content/drive/MyDrive/fraud_detection_pipeline/src/merge.ipynb")
# !cp /content/drive/MyDrive/Colab Notebooks/ingest.ipynb /content/drive/MyDrive/fraud_detection_pipeline/src/ingest.ipynb

