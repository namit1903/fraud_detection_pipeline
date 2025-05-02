#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install transformers datasets sentencepiece -q')


# In[9]:


import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset


# In[10]:


# y_train = pd.read_csv('/content/drive/MyDrive/fraud_detection_pipeline/data/training_&_testing/y_train.csv').squeeze().astype(int)
X_train = pd.read_csv('/content/drive/MyDrive/fraud_detection_pipeline/data/training_&_testing/X_train.csv')
y_train = pd.read_csv('/content/drive/MyDrive/fraud_detection_pipeline/data/training_&_testing/y_train.csv')
X_test = pd.read_csv('/content/drive/MyDrive/fraud_detection_pipeline/data/training_&_testing/X_test.csv')
y_test = pd.read_csv('/content/drive/MyDrive/fraud_detection_pipeline/data/training_&_testing/y_test.csv')
X_train['user_sequence'] = X_train['user_sequence'].fillna('')
X_test['user_sequence'] = X_test['user_sequence'].fillna('')

X_train=X_train['user_sequence'] ;
X_test=X_test['user_sequence'] ;
print (X_train,y_train)
#


# In[5]:


pip install -U transformers


# In[14]:


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# In[15]:


from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Step 1: Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Step 2: Prepare Datasets
train_dataset = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.squeeze().tolist()})
test_dataset = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.squeeze().tolist()})

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Step 3: Load Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 4: TrainingArguments
from transformers import TrainingArguments
#just for fater training ,
train_dataset = train_dataset.select(range(100))# delete when actually need to train
test_dataset = test_dataset.select(range(100))# delete when actually need to train

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10
)


# Step 5: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
   compute_metrics=compute_metrics
)

# Step 6: Train
trainer.train()


metrics = trainer.evaluate()
print(metrics)
# 688b074d285c8217402167e0d970e4dc5d95c21a


# In[16]:


for metric, value in metrics.items():
    print(f'{metric}: {value}')


# In[17]:


model.save_pretrained("/content/drive/MyDrive/fraud_detection_pipeline/model")
tokenizer.save_pretrained("/content/drive/MyDrive/fraud_detection_pipeline/model")


# load model that is already trained

# In[18]:


model = DistilBertForSequenceClassification.from_pretrained("/content/drive/MyDrive/fraud_detection_pipeline/model")
tokenizer = DistilBertTokenizerFast.from_pretrained("/content/drive/MyDrive/fraud_detection_pipeline/model")



# In[ ]:


from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
from tqdm import tqdm

# Load tokenizer and your fine-tuned model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("/content/drive/MyDrive/fraud_detection_pipeline/model")  # adjust path
model.eval()  # set model to evaluation mode

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to get embeddings
def get_cls_embeddings(text_list):
    embeddings = []
    with torch.no_grad():
        for text in tqdm(text_list):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # [CLS] token
            embeddings.append(cls_embedding)
    return embeddings
# Use This to Get Train/Test Embeddings:
# train_embeddings = get_cls_embeddings(X_train.tolist())
test_embeddings = get_cls_embeddings(X_test.tolist())


# In[ ]:


import numpy as np
import os

# Create a directory for embeddings if not exist
# os.makedirs("embeddings", exist_ok=True)
# /content/drive/MyDrive/fraud_detection_pipeline/data/embeddings
# Save embeddings
np.save("/content/drive/MyDrive/fraud_detection_pipeline/data/embeddings/train_cls_embeddings.npy", np.array(train_embeddings))
np.save("/content/drive/MyDrive/fraud_detection_pipeline/data/embeddings/test_cls_embeddings.npy", np.array(test_embeddings))


#  How to Load the Model and Tokenizer Later (No Retraining Needed)

# In[ ]:


# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# # (Next time, directly load without retraining)
# tokenizer = DistilBertTokenizer.from_pretrained('/content/drive/MyDrive/fraud_detection_pipeline/model/transformer_model')
# model = DistilBertForSequenceClassification.from_pretrained('/content/drive/MyDrive/fraud_detection_pipeline/model/transformer_model')

# # Evaluate


# In[ ]:


get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/fraud_detection_pipeline/src/tranformer.ipynb"')


# In[ ]:


import shutil
shutil.copy("/content/drive/MyDrive/Colab Notebooks/tranformer.ipynb", "/content/drive/MyDrive/fraud_detection_pipeline/src/transformer.ipynb")
# !cp /content/drive/MyDrive/Colab Notebooks/ingest.ipynb /content/drive/MyDrive/fraud_detection_pipeline/src/ingest.ipynb

