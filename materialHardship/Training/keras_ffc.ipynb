
# coding: utf-8

# # Prepare data (just for materialHardship)

# In[1]:

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sys
import math
import keras
from sklearn.grid_search import GridSearchCV


# In[2]:

# mean_impute_df_train, median_impute_df_train  = pd.read_pickle('mean_impute_df_train'), pd.read_pickle('median_impute_df_train')


# In[3]:

# mean_impute_df_train_with_labels = pd.read_pickle('mean_impute_df_train_with_labels')
# median_impute_df_train_with_labels  = pd.read_pickle('median_impute_df_train_with_labels')


# In[4]:

final_over_balanced =  pd.read_pickle('final_over_balanced_decimals')


# In[5]:

# non_na_y_train_materialHardship =  pd.read_pickle('non_na_y_train_materialHardship')


# # In[6]:

df_prediction =  pd.read_pickle('df_prediction')


# Read files from pickles...we are reading full dataframes, with columns of features attached to label columns. Function below is used to split features from class labels and make models and predictions.

# In[7]:

# split data into features and labels
def split_data(df, label='materialHardship'):
    copy = df.copy() # copy df so i dont alter original df by popping
    mh = copy.pop(label) # pop label
    x_train_bal, x_valid_bal, y_train_bal, y_valid_bal = train_test_split(copy, mh, test_size=0.2) #train test split
    return x_train_bal, x_valid_bal, y_train_bal, y_valid_bal # return


# In[8]:

x_tr, x_val, y_tr, y_val = split_data(final_over_balanced)


# In[10]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

X = x_tr.values
Y = y_tr.values

# # define base model
# def baseline_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(40, input_dim=x_tr.values.shape[1], kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     # Compile model
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model


# # In[ ]:

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=1)

# seed = 1234
# kfold = KFold(n_splits=2, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/


# w/o sklearn wrap

from keras.models import Sequential

model = Sequential()
model.add(Dense(40, input_dim=x_tr.values.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit( X, Y, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=(x_val.values,y_val.values), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

df_prediction_good_cols = df_prediction[x_tr.columns.values] # keep the columns that are in training data...

preds  = model.predict(df_prediction_good_cols.values,verbose=1)

# # Create Submission file

# In[129]:

# # Need to add train labels to submission too, I think
# print('create submission')
# preds = pd.DataFrame()
# preds['challengeID'] = df_prediction['challengeID']
# preds['materialHardship'] = p_test
# preds.sort_values(by='challengeID').head(10)


# # In[131]:

# # tack on given train values
# sub = preds.append(non_na_y_train_materialHardship)
# sub.sort_values(by='challengeID').head(15)

# sub.to_csv('simple_xgb.csv', index=False)

