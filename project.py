# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 01:50:10 2021

@author: yichao 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# read and rename the dataset
df1 = pd.read_csv('C:/Users/27753/Desktop/MSA course/data mining/DM_project/bank_data_analysis-master/bank-full.csv',sep=";")
df1 = df1.rename(columns = {'y':'deposit'})

# change the value that pdays is -1 to 999
df1['pdays'] = df1['pdays'].replace({-1:999})
#transform string into numeric
plt.figure(figsize=[12,14])
features=["marital", "education", "contact", "default", "housing", "loan", "poutcome", "month"]
n=1
for f in features:
    plt.subplot(4,2,n)
    sns.countplot(x=f, hue='deposit', edgecolor="black", alpha=0.7, data=df1)
    sns.despine()
    plt.title("Countplot of {}  by deposit".format(f))
    n=n+1
plt.tight_layout()
plt.show()

for f in features:
    sns.countplot(x=f, hue='deposit', edgecolor="black", alpha=0.7, data=df1)
    plt.title("Countplot of {}  by deposit".format(f))
    plt.show()
    
# binary encoding
binary_cat = ['deposit', 'default', 'housing', 'loan']
df1[binary_cat] = df1[binary_cat].replace({'yes':1,'no':0})
# multi-label encoding
cat_features = ["marital", "job" ,"education", "contact", "poutcome", "month"]
df1 =pd.get_dummies(df1,columns=cat_features)

# numeric preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
scaler = StandardScaler()
num_feature = ['balance','age', 'previous', 'pdays','campaign','day','duration']
df1[num_feature] = scaler.fit_transform(df1[num_feature].to_numpy())


# Set train_set and label_set
X = df1.drop(columns = 'deposit')
y = df1.deposit

# split training set and test set by stratified split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 42, stratify = y)

#SMOTE remove outliers
from imblearn.combine import SMOTEENN
from collections import Counter
X_resampled_ENN, y_resampled_ENN = SMOTEENN(n_jobs=-1).fit_resample(X_train, y_train)
print(sorted(Counter(y_resampled_ENN).items()))
print(sorted(Counter(X_resampled_ENN).items()))


############
# modeling #
############
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import  make_scorer
from sklearn.metrics import  roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix
method = make_scorer(matthews_corrcoef)

# LogisticRegression
log_reg = LogisticRegression(penalty='l2',
                             dual=False,
                             tol=0.0001,
                             C=1.0,
                             fit_intercept=True,
                             intercept_scaling=1,
                             class_weight=None,
                             random_state=None,
                             solver='liblinear',
                             max_iter=150,
                             multi_class='ovr',
                             verbose=0,
                             warm_start=False,
                             n_jobs=1

)
#tune parameter
params = [{'penalty': ['l1', 'l2'],'max_iter':[125,150,175]}]
logitRegression = GridSearchCV(log_reg, params, cv=5).fit(X_resampled_ENN, y_resampled_ENN)
logitRegression.best_score_
logitRegression.best_params_

log_scores = cross_val_score(log_reg, X_resampled_ENN, y_resampled_ENN, scoring = method, cv=5)
log_reg_mean = log_scores.mean()
log_reg_mean
log_reg.fit(X_resampled_ENN, y_resampled_ENN)
y_pred_reg = log_reg.predict(X_test)
#
print('accuracy = ', accuracy_score(y_resampled_ENN, y_pred_reg))
print('confusion_matrix =\n ', confusion_matrix(y_test, y_pred_reg))
print('MCC = ', matthews_corrcoef(y_test, y_pred_reg))

# catboosting

cat_model = CatBoostClassifier(depth= 10,learning_rate =0.05,l2_leaf_reg=1, iterations=1000)
cat_tuned_parameter = [{'depth': [4, 6, 10],
          'learning_rate' : [0.03,0.05, 0.1],
          'l2_leaf_reg': [1,4,9]
                                 }]

grid_search = GridSearchCV(cat_model, param_grid=cat_tuned_parameter, cv=5).fit(X_resampled_ENN, y_resampled_ENN)
grid_search.best_score_
grid_search.best_params_
cat_scores = cross_val_score(cat_model, X_resampled_ENN, y_resampled_ENN,scoring = method, cv=5)
cat_mean = cat_scores.mean()
cat_mean
cat_model.fit(X_resampled_ENN, y_resampled_ENN)
y_pred_cat_1 = cat_model.predict(X_test)
matthews_corrcoef(y_test,y_pred_cat_1)
confusion_matrix(y_test,y_pred_cat_1)

# randomforest
rand_clf = RandomForestClassifier(
        n_estimators=50,
        criterion='gini',
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=0,
        verbose=0,
        warm_start=False,
        class_weight='balanced'
        )
#tune parameter
tuned_parameter = [{'max_depth':range(3,7), 'n_estimators':[20,50,100,200,300]}]
grid_search = GridSearchCV(rand_clf, param_grid=tuned_parameter, cv=5).fit(X_resampled_ENN, y_resampled_ENN)
grid_search.best_score_
grid_search.best_params_

rand_scores = cross_val_score(rand_clf, X_resampled_ENN, y_resampled_ENN,scoring = method, cv=5)
rand_mean = rand_scores.mean()
rand_mean
rand_clf.fit(X_resampled_ENN, y_resampled_ENN)
y_pred_ran = rand_clf.predict(X_test)
matthews_corrcoef(y_test,y_pred_ran)
confusion_matrix(y_test,y_pred_ran)

#NN
from sklearn.neural_network import MLPClassifier





#lightgbm
import lightgbm as lgb
lgb_clf = lgb.LGBMClassifier(objective='binary',
                        learning_rate=0.07,
                        n_estimators=125,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        num_leaves=90,
                        max_depth= 10
                        )
lgb_scores = cross_val_score(lgb_clf, X_resampled_ENN, y_resampled_ENN,scoring = method, cv=5)
#tune parameter
tuned_parameter_lgb = [{'max_depth':[7,8,10,12], 'num_leaves': [50,60,80,90],
                        'learning_rate':[0.05,0.04,0.06,0.07],'n_estimators':[50,75,100,125]}]
grid_search = GridSearchCV(lgb_clf, param_grid=tuned_parameter_lgb, cv=5).fit(X_resampled_ENN, y_resampled_ENN)
grid_search.best_score_
grid_search.best_params_
#results
lgb_mean = lgb_scores.mean()
lgb_mean
lgb_clf.fit(X_resampled_ENN, y_resampled_ENN)
y_pred_lgb = lgb_clf.predict(X_test)
matthews_corrcoef(y_test,y_pred_lgb)
confusion_matrix(y_test,y_pred_lgb)


# neural network
from sklearn.neural_network import MLPClassifier

MLPC = MLPClassifier()
MLPC.fit(X_resampled_ENN, y_resampled_ENN)
params = {'hidden_layer_sizes': [5, 10],
          'max_iter': [200, 600],
          'learning_rate': ['constant', 'adaptive'],
          'learning_rate_init': [0.1, 0.001],

}
nn_clf = GridSearchCV(MLPC, params, cv=5, return_train_score=True).fit(X_resampled_ENN, y_resampled_ENN)
print(nn_clf.best_params_)
print(nn_clf.best_score_)
print(nn_clf.cv_results_)
MLPC_score = cross_val_score(nn_clf.best_estimator_, X_resampled_ENN, y_resampled_ENN, cv=5, scoring=method)
#df = pd.DataFrame.from_dict(nn_clf.cv_results_)
#df.to_csv("/Users/shanhe/Downloads/nn.csv", sep=";")


print(MLPC_score)
pred_mlpc = nn_clf.predict(X_test)
print('accuracy = ', accuracy_score(y_test, pred_mlpc))
print('confusion_matrix =\n ', confusion_matrix(y_test, pred_mlpc))
print('MCC = ', matthews_corrcoef(y_test, pred_mlpc))





