
# coding: utf-8

# # Prepare data (just for materialHardship)

# In[2]:

import numpy as np
import pandas as pd
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:

np.random.seed(1234)
print('reading in csv files')
# File read in
# See documentation for more descriptions
# http://www.fragilefamilieschallenge.org/apply/
background = pd.read_csv('background.csv', low_memory=False)
print('read background.csv')
train = pd.read_csv('train.csv', low_memory=False)
print('read train.csv')
prediction = pd.read_csv('prediction.csv', low_memory=False)
print('read prediction.csv')
print("Files read")


# Background is super WIDE. We have only 4242 IDs, but almost 13k features!

# In[4]:

background.shape


# OK...so they want predictions for...everything. So we are going to make a **training** set using the rows of background that have non-NA values for materialHardship.
# 
# Then we need to make predictions for every challengeID that is a) not train or b) is in train by has NA for materialHardship 

# In[5]:

all_y_train_materialHardship = train[['challengeID', 'materialHardship']]
# non NA y_train ds and data
non_na_y_train_materialHardship = all_y_train_materialHardship.dropna()
non_na_y_train_materialHardship.head()


# In[6]:

# y_train_ids that are na
na_y_train_materialHardship = all_y_train_materialHardship.loc[~all_y_train_materialHardship['challengeID'].isin(non_na_y_train_materialHardship['challengeID'])]
na_y_train_materialHardship.head()


# Our training set will have all ids that we have non-NA material hardship data for. 

# In[7]:

df_train = background.loc[background['challengeID'].isin(non_na_y_train_materialHardship['challengeID'])]
df_train.head()


# The prediction set contains all rows in 'background.csv' that are not in df_train.

# In[9]:

df_prediction = background.loc[~background['challengeID'].isin(df_train['challengeID'])]
df_prediction.sort_values(by='challengeID').head(10)


# SO, we have the data that we will train on:
#     
#     - Training features: 'df_train' 
#     - Training labels: 'non_na_y_train_materialHardship'
# 
# We'll use this model to predict output (materialHardship) values for:
# 
#     - Prediction features: 'df_prediction'

# ### df_train

# In[11]:

df_train.head()


# ### non_na_y_train_materialHardship

# In[12]:

non_na_y_train_materialHardship.head()


# ### df_prediction

# In[13]:

df_prediction.head()


# We should have that the sum of rows in df_train and df_prediction sum to all of the rows in background.csv. Let's ensure that.

# In[15]:

np.sum(df_train.shape[0]+df_prediction.shape[0])==background.shape[0]


# OK. We are off to the races and can reference these pandas dataframes in future data preparation and training notebooks!

# In[ ]:



