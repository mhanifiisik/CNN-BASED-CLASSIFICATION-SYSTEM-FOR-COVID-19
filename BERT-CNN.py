#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import re
import string
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud


# In[38]:


df=pd.read_excel("dataset3.xlsx")  
df.head(20)


# In[39]:


df.shape


# In[40]:


from sklearn.utils import shuffle
df = shuffle(df)
df = df.reset_index(drop=True)

df.head(20)


# In[41]:


needed_columns=["text","Class"]
df=df[needed_columns]


# In[42]:


#Lower Case
df["text"] = df["text"].str.lower()
df.head()


# In[43]:


#!pip install pyspellchecker

#from spellchecker import SpellChecker

#spell = SpellChecker()

#correction=lambda x: ''.join(spell.correction(x))
#df['text']=df["text"].apply(correction)


# In[44]:


"""
!pip install contractions
import contractions

def contraction(text):
  return " ".join([contractions.fix(word) for word in text.split()])

df["text"] = df["text"].apply(lambda text: contraction(text))
"""


# In[45]:


#Tokenize
from nltk.tokenize import TweetTokenizer  
tk = TweetTokenizer()
df['text'] = df.text.apply(lambda row: tk.tokenize(row))


# In[46]:


# Removing url
texts=df.text
remove_url=lambda x:re.sub(r"https\S+","",str(x))
texts_lr=texts.apply(remove_url)


# In[47]:


# Removing multiple spaces
remove_space=lambda x:re.sub(r'\s+', ' ', str(x))
texts_lr=texts_lr.apply(remove_space)


# In[48]:


# Removing single words
remove_sword=lambda x:re.sub(r"\s+[a-zA-Z]\s+", ' ', str(x))
texts_lr=texts_lr.apply(remove_sword)


# In[49]:


remove_username=lambda x:re.sub('@[^\s]+','',str(x))
texts_lr=texts_lr.apply(remove_username)


# In[50]:


# To lowercase
lower=lambda x :x.lower()
lowercase_texts=texts_lr.apply(lower)


# In[51]:


# Removing asciiand special charachters
remove_ascii=lambda x:re.sub(r'[^\x00-\x7f]',r'',str(x))
lowercase_texts=lowercase_texts.apply(remove_ascii)
remove_b=lambda x:x.lstrip('b')
lowercase_texts=lowercase_texts.apply(remove_b)


# In[52]:


# Removing punctions
remove_puncs=lambda x:x.translate(str.maketrans("","",string.punctuation))
lowercase_texts=lowercase_texts.apply(remove_puncs)


# In[53]:


# Removing stop words
stop_words=set(stopwords.words("english"))
remove_words=lambda x: " ".join([word for word in x.split() if word not in stop_words])
text=lowercase_texts.apply(remove_words)


# In[54]:

'''
#Steeming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem_words(text):
  return " ".join([stemmer.stem(word) for word in text.split()])

df["text"] = df["text"].apply(lambda text: stem_words(text))


# In[ ]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df["text"] = df["text"].apply(lambda text: lemmatize_words(text))
'''

# In[81]:


df.text= text
df.head()
df.drop_duplicates(subset ="text",keep = False, inplace = True)


# In[82]:


data=df
data.head()


# In[83]:


data['Class'].value_counts()


# In[84]:


labelize=lambda x: 0 if x=="not relevant" else "1"
data["Class"]=data.Class.apply(labelize)
data.head()


# In[85]:


from transformers import logging as hf_logging
import transformers
model_class = transformers.BertModel
tokenizer_class = transformers.BertTokenizer
pretrained_weights='bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
bert_model = model_class.from_pretrained(pretrained_weights)


# In[86]:


import logging
import time
from platform import python_version

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable


# In[87]:


max_seq=75
target_columns='Class'
from sklearn.utils import shuffle
data = shuffle(data)


data_train = data.sample(frac=0.75,random_state=200)
data_train.reset_index()


data_val = data.sample(frac=0.1,random_state=200)
data_val.reset_index()


data_test = data.sample(frac=0.2,random_state=200)
data_test.reset_index()

#data_val = data[10001:12500].reset_index(drop=True)
#data_test = data[12501:].reset_index(drop=True)
#data_train = data[:100].reset_index(drop=True)
#data_val = data[101:200].reset_index(drop=True)
#data_test = data[201:300].reset_index(drop=True)


# In[88]:


data_train.shape


# In[89]:


data_test.shape


# In[90]:


def tokenize_text(data, max_seq):
    return [
        tokenizer.encode(text, add_special_tokens=True)[:max_seq] for text in data.text.values
    ]


def pad_text(tokenized_text, max_seq):
    return np.array([el + [0] * (max_seq - len(el)) for el in tokenized_text])


def tokenize_and_pad_text(data, max_seq):
    tokenized_text = tokenize_text(data, max_seq)
    padded_text = pad_text(tokenized_text, max_seq)
    return torch.tensor(padded_text)


def targets_to_tensor(data,target_columns):
    
    
    data[target_columns]= data[target_columns].astype(float)
    return torch.tensor(data[target_columns].values, dtype=torch.float32)


# In[91]:


train_indices = tokenize_and_pad_text(data_train, max_seq)
train_indices = Variable(train_indices, requires_grad=False).long()



test_indices = tokenize_and_pad_text(data_test, max_seq)
test_indices = Variable(test_indices, requires_grad=False).long()


val_indices = tokenize_and_pad_text(data_val, max_seq)
val_indices = Variable(val_indices, requires_grad=False).long()


# In[92]:


with torch.no_grad():
    x_train = bert_model(train_indices)[0]  # Models outputs are tuples
    x_test = bert_model(test_indices)[0]
    x_val = bert_model(val_indices)[0]


# In[93]:


y_train = targets_to_tensor(data_train, target_columns)
y_test = targets_to_tensor(data_test, target_columns)
y_val = targets_to_tensor(data_val, target_columns)


# In[94]:


x_train[0]


# In[95]:


class KimCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static):
        super(KimCNN, self).__init__()

        V = embed_num
        D = embed_dim
        C = class_num
        Co = kernel_num
        Ks = kernel_sizes
        
        self.static = static
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        output = self.sigmoid(logit)
        return output

embed_num = x_train.shape[1]
embed_dim = x_train.shape[2]
class_num = 1#y_train.shape[1]
kernel_num = 3
kernel_sizes = [2, 3, 4]
dropout = 0.5
static = True

model = KimCNN(
    embed_num=embed_num,
    embed_dim=embed_dim,
    class_num=class_num,
    kernel_num=kernel_num,
    kernel_sizes=kernel_sizes,
    dropout=dropout,
    static=static,
)
n_epochs = 10
batch_size = 10
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCELoss()

def generate_batch_data(x, y, batch_size):
    i, batch = 0, 0
    for batch, i in enumerate(range(0, len(x) - batch_size, batch_size), 1):
        x_batch = x[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        yield x_batch, y_batch, batch
    if i + batch_size < len(x):
        yield x[i + batch_size :], y[i + batch_size :], batch + 1
    if batch == 0:
        yield x, y, 1

# Commented out IPython magic to ensure Python compatibility.
train_losses, val_losses = [], []

for epoch in range(n_epochs):
    start_time = time.time()
    train_loss = 0

    model.train(True)
    

    model.eval() # disable dropout for deterministic output
    

model.eval() # disable dropout for deterministic output
with torch.no_grad(): # deactivate autograd engine to reduce memory usage and speed up computations
    y_preds = []
    batch = 0
    for x_batch, y_batch, batch in generate_batch_data(x_test, y_test, batch_size):
        y_pred = model(x_batch)
        y_preds.extend(y_pred.cpu().numpy().tolist())
    y_preds_np = np.array(y_preds)

y_test_np = data_test[target_columns].values
auc_scores = roc_auc_score(y_test_np, y_preds_np, average='micro')
print(auc_scores)



#df_accuracy = pd.DataFrame({"label": target_columns, "auc": auc_scores})
from sklearn.metrics import classification_report
print(classification_report(y_test_np, y_preds_np.round()))
from sklearn.metrics import accuracy_score
print("accuracy:",accuracy_score(y_test_np,y_preds_np.round()))
######
from sklearn.metrics import f1_score,precision_score,recall_score

print(f1_score(y_test_np, y_preds_np.round(), average='micro'))
print(precision_score(y_test_np, y_preds_np.round(), average='micro'))
print(recall_score(y_test_np, y_preds_np.round(), average='micro'))



