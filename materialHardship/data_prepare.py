
# coding: utf-8

# # Prepare data (just for materialHardship)

# In[1]:

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[4]:

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

# In[5]:

background.shape


# OK...so they want predictions for...everything. So we are going to make a **training** set using the rows of background that have non-NA values for materialHardship.
# 
# Then we need to make predictions for every challengeID that is a) not train or b) is in train by has NA for materialHardship 

# In[35]:

all_y_train_materialHardship = train[['challengeID', 'materialHardship']]
# non NA y_train ds and data
non_na_y_train_materialHardship = all_y_train_materialHardship.dropna()
non_na_y_train_materialHardship.head()


# In[34]:

# y_train_ids that are na
na_y_train_materialHardship = all_y_train_materialHardship.loc[~all_y_train_materialHardship['challengeID'].isin(non_na_y_train_materialHardship['challengeID'])]
na_y_train_materialHardship.head()


# Our training set will have all ids that we have non-NA material hardship data for. 

# In[19]:

df_train = background.loc[background['challengeID'].isin(non_na_y_train_materialHardship['challengeID'])]
df_train.head()


# In[32]:

# prediction set
# all ids in background, excluding train_ids
background_non_train_ids = background.loc[~background['challengeID'].isin(train['challengeID'])]

# add back in ids that had NA in material hardship train set
background_mh_nas = background.loc[background['challengeID'].isin(na_y_train_materialHardship['challengeID'])]
# background_mh_nas.head()

# combine 
df_prediction = background_non_train_ids.append(background_mh_nas)
df_prediction.sort_values(by='challengeID').head(10)


# SO, we have the data that we will train on:
#     
#     - 'df_train' with labels 'non_na_y_train_materialHardship'
#     - We'll use this model to predict output (materialHardship) values for 'df_prediction'

# In[1]:

df_train.head()
non_na_y_train_materialHardship.head()
df_prediction.head()


# In[ ]:



