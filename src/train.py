#!/usr/bin/env python
# coding: utf-8

# # **Task: Copy last sessionâ€™s Machine Learning notebook**

# # **Task: Simplify your notebook to only keep the train-test split, the auto-sklearn model , evaluation and model explanability.**
# 

# In[ ]:


# get_ipython().system(u'pip install shap')


# In[ ]:


# get_ipython().system(u'pip3 install -U scikit-learn')


# In[ ]:


# get_ipython().system(u'pip install --upgrade plotly')


# In[ ]:


# get_ipython().system(u'sudo apt-get install build-essential swig')
# get_ipython().system(u'curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
# get_ipython().system(u'pip install auto-sklearn')


# In[ ]:


# get_ipython().system(u'pip install pipelineprofiler')


# In[ ]:


import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import set_config

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures



import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn
import matplotlib.pyplot as plt


# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[ ]:


data_path = "/content/drive/MyDrive/Data Science/Introduction2DataScience/tutorials/"


# In[ ]:


pd.set_option('display.max_rows', 20)


# In[ ]:


set_config(display='diagram')


# ### Train-Test Split

# In[ ]:


# load into csv file as a dataframe
wine = pd.read_csv(f'{data_path}winequality-red.csv', sep=';')


# In[ ]:


# columns and rows
wine.shape


# In[ ]:


wine.dtypes


# In[ ]:


# statistical details of the data
wine.describe()


# In[ ]:


# train_test split
train, test = train_test_split(wine, test_size=0.2, random_state=0)
train_features = train.drop(['quality'],axis = 1)


# In[ ]:


# getting rid of outliers
train_no_out = train[ (train['fixed acidity'] <= 13.4) ]
train_no_out = train_no_out [(train_no_out['volatile acidity'] <= 1.13) ]
train_no_out = train_no_out [(train_no_out['citric acid'] <= 0.78) ]
train_no_out = train_no_out [(train_no_out['chlorides'] <= 0.27) | (train_no_out['chlorides'] >= 0.038)]
train_no_out = train_no_out [(train_no_out['free sulfur dioxide'] <= 57.0) ]
train_no_out = train_no_out [(train_no_out['total sulfur dioxide'] <= 160.0)]
train_no_out = train_no_out [(train_no_out['density'] >= 0.991) | (train_no_out['density'] <= 1.0015)]
train_no_out = train_no_out [(train_no_out['pH'] >= 2.86) | (train_no_out['pH'] <= 3.72)]
train_no_out = train_no_out [(train_no_out['sulphates'] <= 1.36)]
train_no_out = train_no_out [(train_no_out['alcohol'] <= 13.4)]
train_no_out = train_no_out [(train_no_out['residual sugar'] <= 8.9)]
train_no_out


# ### Modelling

# In[ ]:


#definition of a pipline
scaler = StandardScaler()
regr = LinearRegression()
transf = PolynomialFeatures(degree = 1)
model = Pipeline(steps=[ 
                        ('scaler', scaler),
                        ('tranform', transf),
                        ('regressor', regr)])


# In[ ]:


#lets divide train set for samples and labeles
X_train, y_train = train.iloc[:,:-1], train.iloc[:,-1] 


# In[ ]:


#cross validation
cross_val_score(model, X_train, y_train)


# ### AutoML

# In[ ]:


total_time = 600
per_run_time_limit = 30


# In[ ]:


import autosklearn.regression
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=total_time,
    per_run_time_limit=per_run_time_limit,
)
automl.fit(X_train, y_train)


# In[ ]:


import PipelineProfiler

profiler_data= PipelineProfiler.import_autosklearn(automl)
PipelineProfiler.plot_pipeline_matrix(profiler_data)


# In[ ]:


#creation of teat samples set and test labeles set
X_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]


# In[ ]:


#predict the median house value from test set
y_pred = automl.predict(X_test)
y_pred


# ### Model Evaluation

# Lets use some metrics to evaluate model

# In[ ]:


#MSE score
mean_squared_error(y_test, y_pred)


# In[ ]:


#R2 score
automl.score(X_test, y_test)


# In[ ]:


plt.scatter(y=y_test,x=np.arange(len(y_test)))
plt.scatter(y=y_pred,x=np.arange(len(y_pred)), color='red')
plt.show()


# ### Model Explainability

# In[ ]:


import shap
explainer = shap.KernelExplainer(model = automl.predict, data = X_test.iloc[:50, :], link = "identity")


# In[ ]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X = X_test.iloc[X_idx:X_idx+1,:], nsamples = 100)
X_test.iloc[X_idx:X_idx+1,:]
# print the JS visualization code to the notebook
shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = X_test.iloc[X_idx:X_idx+1,:]
                )


# In[ ]:


shap_values = explainer.shap_values(X = X_test.iloc[0:50,:], nsamples = 100)


# In[ ]:


# print the JS visualization code to the notebook
shap.initjs()
shap.summary_plot(shap_values = shap_values,
                  features = X_test.iloc[0:50,:]
                  )

