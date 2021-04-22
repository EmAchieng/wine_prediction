#!/usr/bin/env python
# coding: utf-8

# Team members:
# Anastasiya Fokina
# Emily Achieng
# Mohammad Naim Dahee

# <a id='Q0'></a>
# <center><a target="_blank" href="http://www.propulsion.academy"><img src="https://drive.google.com/uc?id=1McNxpNrSwfqu1w-QtlOmPSmfULvkkMQV" width="200" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <center> <h4 style="color:#303030"> Python for Data Science, Homework, template: </h4> </center>
# <center> <h1 style="color:#303030">Predict the quality of white wine from its physico-chemical properties</h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center style="color:#303030"><h4>Propulsion Academy, 2021</h4></center>
# <p style="margin-bottom:1cm;"></p>
# 
# <div style="background:#EEEDF5;border-top:0.1cm solid #EF475B;border-bottom:0.1cm solid #EF475B;">
#     <div style="margin-left: 0.5cm;margin-top: 0.5cm;margin-bottom: 0.5cm">
#         <p><strong>Goal:</strong> Practice Linear Regression on wine data</p>
#         <strong> Sections:</strong>
#         <a id="P0" name="P0"></a>
#         <ol>
#             <li> <a style="color:#303030" href="#SU">Set Up </a> </li>
#             <li> <a style="color:#303030" href="#P1">Exploratory Data Analysis</a></li>
#             <li> <a style="color:#303030" href="#P2">Modeling</a></li>
#         </ol>
#         <strong>Topics Trained:</strong> Clustering.
#     </div>
# </div>
# 
# <nav style="text-align:right"><strong>
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/" title="momentum"> SIT Introduction to Data Science</a>|
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/weeks/week2/day1/index.html" title="momentum">Week 2 Day 1, Applied Machine Learning</a>|
#         <a style="color:#00BAE5" href="https://colab.research.google.com/drive/1DK68oHRR2-5IiZ2SG7OTS2cCFSe-RpeE?usp=sharing" title="momentum"> Assignment, Wine Quality Prediction</a>
# </strong></nav>

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# In[38]:


# get_ipython().system(u'pip install shap')
# get_ipython().system(u'pip3 install -U scikit-learn')
# get_ipython().system(u'pip install --upgrade plotly')
# get_ipython().system(u'sudo apt-get install build-essential swig')
# get_ipython().system(u'curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
# get_ipython().system(u'pip install auto-sklearn')
# get_ipython().system(u'pip install pipelineprofiler')


# In[1]:


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


import autosklearn.regression
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn
import matplotlib.pyplot as plt

import shap
import datetime

from joblib import dump

import logging


# In[2]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[3]:


data_path = "/content/drive/MyDrive/Data Science/Introduction2DataScience/tutorials/wine_prediction/data/raw/"


# In[4]:


model_path = "/content/drive/MyDrive/Data Science/Introduction2DataScience/tutorials/wine_prediction/models/"


# In[5]:


timesstr = str(datetime.datetime.now()).replace(' ', '_')


# In[6]:


logging.basicConfig(filename=f"{model_path}explog_{timesstr}.log", level=logging.INFO)


# <a id='P1' name="P1"></a>
# ## [Exploratory Data Analysis](#P0)
# 

# ### Understand the Context

# **What type of problem are we trying to solve?**
# 
# With this data set, we want to build a model that would predict the quality of a wine from its physico-chemical characteristics. 
# 
# **_This can be treated either as a classification or a regression problem_**

# **How was the data collected?/ Is there documentation on the Data?**
# 
# Red wine dataset description: 
# 
# **Context**:
# 
# The acidity, alcohol content, as well as other components contents have been measured on wine samples and are reported along with the quality of said wine. the question is: how well can we predict the quality of a wine from these measurements?
# 
# 
# **Content**:  
# 
# For more information, read [Cortez et al., 2009].
# Input variables (based on physicochemical tests):
# 
# 1. fixed acidity
# 2. volatile acidity
# 3. citric acid
# 4. residual sugar
# 5. chlorides
# 6. free sulfur dioxide
# 7. total sulfur dioxide
# 8. density
# 9. pH
# 10. sulphates
# 11. alcohol
# 
# Output variable (based on sensory data):
# 
# 12. quality (score between 0 and 10)

# **Can we foresee any challenge related to this data set?**

# I see no pitfalls

# ### Data Structure and types

# **Load the csv file as a DataFrame using Pandas**

# In[7]:


wine = pd.read_csv(f'{data_path}winequality-red.csv', sep=';')


# **How many columns and rows do we have?**

# **Perform test/train split here**
# 
# !!! Please think about it!!! How should the data be splitted?

# Dealing with outliers

# In[8]:


wine = wine[ (wine['fixed acidity'] <= 13.4) ]
wine = wine [(wine['volatile acidity'] <= 1.13) ]
wine = wine [(wine['citric acid'] <= 0.78) ]
wine = wine [(wine['chlorides'] <= 0.27) | (wine['chlorides'] >= 0.038)]
wine = wine [(wine['free sulfur dioxide'] <= 57.0) ]
wine = wine [(wine['total sulfur dioxide'] <= 160.0)]
wine = wine [(wine['density'] >= 0.991) | (wine['density'] <= 1.0015)]
wine = wine [(wine['pH'] >= 2.86) | (wine['pH'] <= 3.72)]
wine = wine [(wine['sulphates'] <= 1.36)]
wine = wine [(wine['alcohol'] <= 13.4)]
wine = wine [(wine['residual sugar'] <= 8.9)]


# In[9]:


test_size = 0.2
random_state = 50


# In[10]:


train, test = train_test_split(wine, test_size=test_size, random_state=random_state)


# In[11]:


logging.info(f'train test split with test_size={test_size} and random state={random_state}')


# In[12]:


train.to_csv(f'{data_path}Wine_Quality_Red_Train.csv',index=False)


# In[13]:


train = train.copy()


# In[14]:


test.to_csv(f'{data_path}Wine_Quality_Red_Test.csv',index=False)


# In[15]:


test = test.copy()


# <a id='P2' name="P2"></a>
# ## [Modelling](#P0)

# In[19]:


X_train, y_train = train.iloc[:,:-1], train['quality']


# In[22]:


time_left_for_this_task=600
per_run_time_limit=30


# In[24]:



automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=time_left_for_this_task,
    per_run_time_limit=per_run_time_limit,
)
automl.fit(X_train, y_train)


# In[26]:


logging.info(f'Ran autosklearn regressor for a total time of {time_left_for_this_task} seconds, with a maximum of {per_run_time_limit} seconds per model run')


# In[27]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[28]:


logging.info(f'Saved classification model at {model_path}model{timesstr}.pkl ')


# In[29]:


logging.info(f'autosklearn model statistics:')
logging.info(automl.sprint_statistics())


# ### Model Evaluation

# In[30]:


X_test, y_test = test.iloc[:,:-1], test['quality']


# In[31]:


y_pred = automl.predict(X_test)


# In[32]:


logging.info(f"Mean Squared Error is {mean_squared_error(y_test, y_pred)}, \n R2 score is {automl.score(X_test, y_test)}")


# In[33]:


df = pd.DataFrame(np.concatenate((X_test, y_test.to_numpy().reshape(-1,1), y_pred.reshape(-1,1)),  axis=1))


# In[35]:


df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol','Predicted Target','True Target']


# In[36]:


fig = px.scatter(df, x='Predicted Target', y='True Target')
fig.write_html(f"{model_path}residualfig_{timesstr}.html")


# In[37]:


logging.info(f"Figure of residuals saved as {model_path}residualfig_{timesstr}.html")


# In[38]:


explainer = shap.KernelExplainer(model = automl.predict, data = X_test.iloc[:50, :], link = "identity")


# In[39]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X = X_test.iloc[X_idx:X_idx+1,:], nsamples = 100)
X_test.iloc[X_idx:X_idx+1,:]
# print the JS visualization code to the notebook
# shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = X_test.iloc[X_idx:X_idx+1,:], 
                show=False,
                matplotlib=True
                )
plt.savefig(f"{model_path}shap_example_{timesstr}.png")
logging.info(f"Shapley example saved as {model_path}shap_example_{timesstr}.png")


# In[40]:


shap_values = explainer.shap_values(X = X_test.iloc[0:50,:], nsamples = 100)


# In[42]:


# print the JS visualization code to the notebook
# shap.initjs()
fig = shap.summary_plot(shap_values = shap_values,
                  features = X_test.iloc[0:50,:],
                  show=False)
plt.savefig(f"{model_path}shap_summary_{timesstr}.png")
logging.info(f"Shapley summary saved as {model_path}shap_summary_{timesstr}.png")

