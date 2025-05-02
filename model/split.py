#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sklearn.model_selection import train_test_split

def split_dataset(df, label_col='is_fraud', test_size=0.2, random_state=42):
    """
    Splits dataset into train and test sets with stratification on fraud label.
    Returns: X_train, X_test, y_train, y_test
    """
    # Separate features and label
    X = df.drop(columns=[label_col, 'user_id'])  # Drop user_id too for model
    y = df[label_col]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# In[22]:


import pandas as pd
# cdr_processed = pd.read_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/processed/cdr_processed.csv")

merged_df=pd.read_csv('/content/drive/MyDrive/fraud_detection_pipeline/data/processed/merged_data.csv')
# print(merged_df.head())
def consolidate_is_fraud_labels(df):
    fraud_cols = [col for col in df.columns if 'is_fraud' in col]
    print("Found fraud-related columns:", fraud_cols)
    df['is_fraud'] = df[fraud_cols].max(axis=1)  # or use mean(axis=1).round()
    # df.drop(columns=fraud_cols, inplace=True)
    return df

merged_df = consolidate_is_fraud_labels(merged_df)

merged_df.head()


# In[23]:


X_train, X_test, y_train, y_test = split_dataset(merged_df)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Fraud % in Train: {y_train.mean()*100:.2f}%")
print(f"Fraud % in Test: {y_test.mean()*100:.2f}%")


# export split dataset

# In[24]:


X_train.to_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/training_&_testing/X_train.csv", index=False)
X_test.to_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/training_&_testing/X_test.csv", index=False)
y_train.to_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/training_&_testing/y_train.csv", index=False)
y_test.to_csv("/content/drive/MyDrive/fraud_detection_pipeline/data/training_&_testing/y_test.csv", index=False)


# In[25]:


get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/fraud_detection_pipeline/model/split.ipynb"')


# In[26]:


import shutil
shutil.copy("/content/drive/MyDrive/Colab Notebooks/split.ipynb", "/content/drive/MyDrive/fraud_detection_pipeline/model/split.ipynb")
# !cp /content/drive/MyDrive/Colab Notebooks/ingest.ipynb /content/drive/MyDrive/fraud_detection_pipeline/src/ingest.ipynb

