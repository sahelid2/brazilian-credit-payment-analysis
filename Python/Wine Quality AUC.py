#!/usr/bin/env python
# coding: utf-8

# In[40]:


## Read the data
import matplotlib
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dateutil.relativedelta import relativedelta
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
data=r"/Volumes/ExtData/Documents/VCU/2nd Sem/Python/data/winequality.csv"
df = pd.read_csv(data, encoding='ISO-8859-1')
df = df.drop(df.columns[0], axis=1)


# In[41]:


df.loc[df['wine type'] == 'White', 'wine_type'] = 1
df.loc[df['wine type'] == 'Red', 'wine_type'] = 0
df.loc[df['quality'] >= 8, 'high_quality'] = 1
df.loc[df['quality'] < 8, 'high_quality'] = 0


# In[43]:


preds = df.columns[1:-1]
X = df[preds].drop('wine type', axis=1)
y=df['high_quality']


# In[54]:


msk = np.random.rand(len(df)) <= 0.5
X_train = X[msk]
X_test = X[~msk]
y_train = y[msk]
y_test = y[~msk]
# In[47]:
# target distribution in the trainig sample

y_train.value_counts()
# In[48]:
# target distribution in the test sample
y_test.value_counts()
# In[49]:


# In[52]:


from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(random_state=314)

param_grid = {'n_estimators': [100, 300, 500], 
              'learning_rate': [0.05, 0.1, 0.5],
              'max_depth': [3, 6],
              'min_samples_leaf': [0.01, 0.02, 0.05]}

gbm_rs = RandomizedSearchCV(gbm, param_grid, cv=3, n_iter=10, n_jobs=-1, random_state=314)

gbm_rs.fit(X_train, y_train)

print ('Best GBM Parameters:', gbm_rs.best_params_)

gbm_scores_train = gbm_rs.predict_proba(X_train)[:, 1]
gbm_scores_test = gbm_rs.predict_proba(X_test)[:, 1]

gbm_fpr_train, gbm_tpr_train, _ = roc_curve(y_train, gbm_scores_train)
gbm_fpr_test, gbm_tpr_test, _ = roc_curve(y_test, gbm_scores_test)


# In[53]:


# ROC Curve
plt.figure().set_size_inches(12, 9)

plt.plot(gbm_fpr_train, gbm_tpr_train, color='purple', lw=2, alpha = 0.2, linestyle = '-',
         label=f'GBM Train (AUC = {roc_auc_score(y_train, gbm_scores_train):0.3f})')

plt.plot(gbm_fpr_test, gbm_tpr_test, color='purple', lw=2, linestyle = '-',
         label=f'GBM Test (AUC = {roc_auc_score(y_test, gbm_scores_test):0.3f})')


plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 14)
plt.ylabel('True Positive Rate', fontsize = 14)
plt.title('Default Risk Model: Gradient Boosting', fontsize = 16)
plt.legend(loc='lower right', fontsize = 14)
plt.show();


# In[47]:


X_train


# In[48]:


gbm_gs = GridSearchCV(gbm, param_grid, cv=3, n_jobs=-1)

gbm_gs.fit(X_train, y_train)


# In[ ]:




