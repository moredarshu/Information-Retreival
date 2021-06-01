#!/usr/bin/env python
# coding: utf-8

# In[123]:


# Importing Libraries
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.neighbors import KDTree
import joblib


# In[85]:


# Reading Dataset
data_info = pd.read_csv(r'C:\Users\Darshana\Desktop\DSC_WKND20092020\NLP\infromation retrival\famous_people.csv')


# In[86]:


# Reading top 5 data
data_info.head()


# In[87]:


# Shape of the dataframe
data_info.shape


# ## Preprocessing
# #### 1. Lower
# ####  2.Remove stop words
# #### 3.Remove Punctuation
# #### 4.Lemmatization

# In[88]:


info_text = data_info.Text
info_text


# In[89]:


# Converting the text into Lower case
info_lower = []
for i in info_text:
    info_lower.append(i.lower())

print(info_lower[0])


# In[90]:


## Remove the stop words

sw = stopwords.words('english')
info_rmvsw_in = []
info_rmvsw_out = []
for i in range(len(info_lower)):
#     print(f'This is {i} {info_lower[i]}\n')
    for j in nltk.word_tokenize(info_lower[i]):
        if j not in sw:
             info_rmvsw_in.append(j)
#     print(info_rmvsw_in) 
    info_rmvsw_out.append(' '.join(info_rmvsw_in)) 
    info_rmvsw_in = []           
    
 


# In[91]:


info_rmvsw_out[5]


# In[92]:


info_rmv_punct = string.punctuation
info_rmv_punct


# In[93]:


## Remove Punctuation 
info_rmv_punct_in = []
info_rmv_punct_out = []
for i in range(len(info_rmvsw_out)):
#      print(f'This is {i} {info_rmvsw_out[i]}\n')
    for j in nltk.word_tokenize(info_rmvsw_out[i]):
        if j not in info_rmv_punct:
            info_rmv_punct_in.append(j)
#         print(info_rmv_punct_in) 
    info_rmv_punct_out.append(' '.join(info_rmv_punct_in)) 
    info_rmv_punct_in = []  
print(info_rmv_punct_out)    


# In[94]:


info_rmv_punct_out[5]


# In[95]:


## Lemmatization
lemma = WordNetLemmatizer()
info_lemma_in = []
info_lemma_out = []
for i in range(len(info_rmv_punct_out)):
#      print(f'This is {i} {info_rmvsw_out[i]}\n')
    for j in nltk.word_tokenize(info_rmv_punct_out[i]):
            info_lemma_in.append(lemma.lemmatize(j,pos='v'))
#         print(info_rmv_punct_in) 
    info_lemma_out.append(' '.join(info_lemma_in)) 
    info_lemma_in = []  
print(info_lemma_out[5])   


# In[96]:


info_lemma_out


# In[97]:


data_info['TextNew'] = info_lemma_out


# In[98]:


data_info.head()


# In[99]:


data_info.shape


# In[100]:


tfidf = TfidfVectorizer()
train_tfidf=tfidf.fit_transform(data_info.TextNew).toarray()


# In[101]:


tfidf.get_feature_names()


# In[102]:


data_info['TFID_VEC'] = list(train_tfidf)


# In[103]:


data_info.head()


# In[104]:


kdtree = KDTree(train_tfidf)


# In[105]:


# data_info.TFID_VEC[2].shape
# data_info.TFID_VEC[2].reshape(1,-1).shape
data_info.TFID_VEC[2].reshape(-1,1).shape


# In[119]:


distance,idx=kdtree.query(data_info.TFID_VEC[0].reshape(1,-1),k=5)


# In[120]:


list(enumerate(idx[0]))


# In[121]:


distance.shape


# In[122]:


for i,val in list(enumerate(idx[0])):
    print(f'Name:{data_info["Name"][val]}')
    print(f'Distance : {distance[0][i]}')
    print(f'URI : {data_info["URI"][val]}')


# In[124]:


## Saving the vectorizer
joblib.dump(tfidf,'tfidf_vec.pkl')
joblib.dump(kdtree,'kdtree_model.pkl')

