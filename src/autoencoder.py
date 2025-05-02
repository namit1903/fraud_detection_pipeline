#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np

graph_embeddings = np.load('/content/drive/MyDrive/fraud_detection_pipeline/data/embeddings/graphsage_user_embeddings.npy')


# prepare labels

# In[6]:


import pandas as pd

# Load the labels (if needed for evaluation)
y_train = pd.read_csv('/content/drive/MyDrive/fraud_detection_pipeline/data/training_&_testing/y_train.csv')
y_test = pd.read_csv('/content/drive/MyDrive/fraud_detection_pipeline/data/training_&_testing/y_test.csv')

# Combine them back to match embedding order
y_all = np.concatenate([y_train.values, y_test.values])


# autoencoders for anomaly detection

# In[7]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Prepare dataset
X = torch.tensor(graph_embeddings, dtype=torch.float32)
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize and train
model = Autoencoder(input_dim=X.shape[1]).to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train()
for epoch in range(10):
    epoch_loss = 0
    for batch in loader:
        x_batch = batch[0].to(device)
        optimizer.zero_grad()
        reconstructed = model(x_batch)
        loss = loss_fn(reconstructed, x_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")


# detect anomaly from reconstruction error

# In[8]:


model.eval()
with torch.no_grad():
    reconstructed = model(X.to(device)).cpu()
    mse = ((X - reconstructed) ** 2).mean(dim=1).numpy()

# Choose threshold (e.g., 95th percentile)
threshold = np.percentile(mse, 95)

# Predict: 1 = anomaly (fraud), 0 = normal
y_pred_autoencoder = (mse > threshold).astype(int)

# Evaluate (if labels are known)
print("Autoencoder Classification Report:")
print(classification_report(y_all, y_pred_autoencoder))


#  Apply Isolation Forest (as backup method)

# In[ ]:


from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
y_pred_iforest = clf.fit_predict(graph_embeddings)

# Map: -1 (anomaly) → 1 (fraud), 1 (normal) → 0
y_pred_iforest = (y_pred_iforest == -1).astype(int)

# Evaluate
print("Isolation Forest Classification Report:")
print(classification_report(y_all, y_pred_iforest))


# save notebook

# In[3]:


import shutil
shutil.copy("/content/drive/MyDrive/Colab Notebooks/autoencoder.ipynb", "/content/drive/MyDrive/fraud_detection_pipeline/src/autoencoder.ipynb")
# !cp /content/drive/MyDrive/Colab Notebooks/ingest.ipynb /content/drive/MyDrive/fraud_detection_pipeline/src/ingest.ipynb

