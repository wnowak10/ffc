import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

#Inline edit!
np.random.seed(12345) #12345 is a better seed.
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

# Select each column individually from train df
y_train_grit = train[['challengeID', 'grit']]
y_train_gpa = train[['challengeID', 'gpa']]
y_train_materialHardship = train[['challengeID', 'materialHardship']]
y_train_eviction = train[['challengeID', 'eviction']]
y_train_layoff = train[['challengeID', 'layoff']]
y_train_jobTraining = train[['challengeID', 'jobTraining']]



# Let's try to just predict materialHardship first
#########################################
########### materialHardship #############
#########################################

# Get rid of NAs in material hardship.
# There are individuals in the training set
# (which contains 6 total vars) that have NA
# for certain of the variables...so let's eliminate.
y_train_materialHardship = y_train_materialHardship.dropna()


# Create train df with rows in y_train
# Subset background df to do so.
df_train_mh = background.loc[background['challengeID'].isin(y_train_materialHardship['challengeID'])]

# Create a test dataframe too. This df contains
# individuals who are NOT in the training set.
# We will need to make predictions on these.
df_test_mh = background.loc[~background['challengeID'].isin(y_train_materialHardship['challengeID'])]



###############
# XGBoost train
###############

# Only include columns with DataFrame.dtypes
# that are int, float or bool.
print('drop non xgb data types')
df_train_mh_good_dtypes = df_train_mh.select_dtypes(include=(int, float, bool))

df_test_mh_good_dtypes = df_test_mh.select_dtypes(include=(int, float, bool))

# Get rid of challengeID column for training
y_train_mh = y_train_materialHardship['materialHardship']

# Create validation set
x_train, x_valid, y_train, y_valid = train_test_split(df_train_mh_good_dtypes, y_train_mh, test_size=0.2)
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
print('validation and training data ready')
# Set our parameters for xgboost
params = {}
params['objective'] = 'reg:logistic'
params['eval_metric'] = 'rmse'
params['eta'] = 0.02
params['max_depth'] = 20

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

# XGB train
# model = xgb.XGBClassifier()
print('train xgb')
tqdm.pandas()
# bst = model.fit(df_test_mh, y_train_mh)
bst = xgb.train(params, d_train, 200, watchlist, early_stopping_rounds=50, verbose_eval=10)

# Feature importances df
features = pd.DataFrame(bst.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)

# Make predictions
print('predict with xgb!!!')
tqdm.pandas()
d_test = xgb.DMatrix(df_test_mh_good_dtypes)
tqdm.pandas()
p_test = bst.predict(d_test)

# feature importance plot
features = pd.DataFrame(bst.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
# plot (awkwardly large plot, as is)
# plt.rcParams['figure.figsize'] = (12.0, 30.0)
# xgb.plot_importance(bst)
# plt.show()



# Need to add train labels to submission too, I think
print('create submission')
preds = pd.DataFrame()
preds['challengeID'] = df_test_mh['challengeID']
preds['materialHardship'] = p_test
# tack on given train values
orig = pd.DataFrame()
orig['challengeID'] = y_train_materialHardship['challengeID']
orig['materialHardship'] = y_train_materialHardship['materialHardship']
sub = preds.append(orig)
sub.sort_values(by='challengeID')

sub.to_csv('simple_xgb.csv', index=False)
