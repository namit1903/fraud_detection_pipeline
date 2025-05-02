#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


pip install fastapi uvicorn


# In[ ]:


from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = FastAPI()

# Load pre-trained model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("/content/drive/MyDrive/fraud_detection_pipeline/model")

# Fraud detection function
def predict_fraud(user_sequence: str) -> str:
    inputs = tokenizer(user_sequence, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "fraud" if prediction == 1 else "benign"

class UserSequence(BaseModel):
    user_sequence: str

@app.post("/predict/")
def predict(user: UserSequence):
    result = predict_fraud(user.user_sequence)
    return {"result": result}


# In[ ]:


uvicorn fraud_detection_engine:app --reload


# In[ ]:


import requests

url = "http://127.0.0.1:8000/predict/"
data = {"user_sequence": "user_id123,1617025380,call,0,caller1,callee1,1617025300,120,cell_1,location_1,imei123,source_modality_call,16,5,..."}  # Example user sequence

response = requests.post(url, json=data)
print(response.json())  # {'result': 'fraud'}


# In[ ]:


get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/fraud_detection_pipeline/src/api.ipynb"')


# In[ ]:


import shutil
shutil.copy("/content/drive/MyDrive/Colab Notebooks/api.ipynb", "/content/drive/MyDrive/fraud_detection_pipeline/src/fastapi.ipynb")
# !cp /content/drive/MyDrive/Colab Notebooks/ingest.ipynb /content/drive/MyDrive/fraud_detection_pipeline/src/ingest.ipynb

