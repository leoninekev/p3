#!/usr/bin/env python
# coding: utf-8

# In[1]:


import statsmodels.api as sm
import pandas as pd
from chapter01 import base

# loading the training dataset
# df = pd.read_csv('logit_train1.csv', index_col = 0)
 
# # defining the dependent and independent variables
# Xtrain = df[['gmat', 'gpa', 'work_experience']]
# ytrain = df[['admitted']]
  
# # building the model and fitting the data
# log_reg = sm.Logit(ytrain, Xtrain).fit()


# In[3]:


dogs_data= base.load_data()
dogs_data


# In[114]:


import pandas as pd

df= pd.DataFrame({"avoided":np.sum(dogs_data["Y"], axis=1), "shocks":dogs_data["Ntrials"]- np.sum(dogs_data["Y"], axis=1)})
df["Pij"]= np.round(np.mean(dogs_data["Y"], axis=1))
# df["Pij"]=df["Pij"]>0.5
df


# In[115]:


Xtrain= df[["avoided", "shocks"]].values#sm.add_constant(df[["avoided", "shocks"]])#df[["avoided", "shocks"]]#
ytrain= df[["Pij"]].values


# In[125]:


# log_reg = sm.Logit(df[["Pij"]], df[["avoided", "shocks"]]).fit()

log_reg = sm.Logit(ytrain, Xtrain)
# .fit()
logit_res= log_reg.fit()


# In[124]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

data_df= pd.DataFrame({"avoided":np.sum(dogs_data["Y"], axis=1), "shocks":dogs_data["Ntrials"]- np.sum(dogs_data["Y"], axis=1)})
data_df["Pij"]= np.round(np.mean(dogs_data["Y"], axis=1))

Xtrain= df[["avoided", "shocks"]].values#sm.add_constant(df[["avoided", "shocks"]])#df[["avoided", "shocks"]]#
ytrain= df[["Pij"]].values

clf = LogisticRegression(random_state=0).fit(Xtrain, ytrain)
clf.predict_proba(Xtrain[:2, :])
clf.score(Xtrain, ytrain)

clf.coef_# output alpha, beta


# In[123]:





# In[5]:


x_avoidance, x_shocked, y = base.transform_data(**dogs_data)
print("x_avoidance: %s, x_shocked: %s, y: %s"%(x_avoidance.shape, x_shocked.shape, y.shape))
print("\nSample x_avoidance: %s \n\nSample x_shocked: %s"%(x_avoidance[1], x_shocked[1]))


# In[ ]:





# In[74]:


df


# In[46]:


spector_data.exog


# In[48]:


spector_data.data


# In[50]:


get_ipython().run_line_magic('pinfo2', 'sm.datasets.spector.load_pandas')


# In[44]:


import statsmodels.api as sm

spector_data = sm.datasets.spector.load_pandas()

spector_data.exog = sm.add_constant(spector_data.exog)

# Logit Model
logit_mod = sm.Logit(spector_data.endog, spector_data.exog)

# logit_res = logit_mod.fit()
# print(logit_res.summary())


# In[73]:


loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x:  round(float(x.rstrip('%')) / 100, 4))
loanlength = loansData['Loan.Length'].map(lambda x: x.strip('months'))
loansData['FICO.Range'] = loansData['FICO.Range'].map(lambda x: x.split('-'))
loansData['FICO.Range'] = loansData['FICO.Range'].map(lambda x: int(x[0]))
loansData['FICO.Score'] = loansData['FICO.Range']


# In[69]:


from scipy import stats
import numpy as np
import pandas as pd 
import collections
import matplotlib.pyplot as plt
import statsmodels.api as sm

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

loansData = loansData.to_csv('loansData_clean.csv', header=True, index=False)

## cleaning the file
# loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x:  round(float(x.rstrip('%')) / 100, 4))
# loanlength = loansData['Loan.Length'].map(lambda x: x.strip('months'))
# loansData['FICO.Range'] = loansData['FICO.Range'].map(lambda x: x.split('-'))
# loansData['FICO.Range'] = loansData['FICO.Range'].map(lambda x: int(x[0]))
# loansData['FICO.Score'] = loansData['FICO.Range']

# #add interest rate less than column and populate
# ## we only care about interest rates less than 12%
# loansData['IR_TF'] = pd.Series('', index=loansData.index)
# loansData['IR_TF'] = loansData['Interest.Rate'].map(lambda x: True if x < 12 else False)

# #create intercept column
# loansData['Intercept'] = pd.Series(1.0, index=loansData.index)

# # create list of ind var col names
# ind_vars = ['FICO.Score', 'Amount.Requested', 'Intercept'] 

# #define logistic regression
# logit = sm.Logit(loansData['IR_TF'], loansData[ind_vars])

# #fit the model
# result = logit.fit()


# In[29]:


import numpy as np
import matplotlib.pyplot as plt

li1 = np.abs(np.random.normal(0, 316, 1100))
li2= np.random.uniform(0,10, 5000)
li3= np.exp(-li2)
# plt.plot(np.abs(np.random.normal(0, 316, 1100)))
_= plt.hist(li2, bins=30)


# In[28]:


_= plt.hist(li3, bins=8)


# In[16]:





# In[17]:


# print("For model '%s' Prior alpha Q(0.5) :%s | Prior beta Q(0.5) :%s"%(model_name, np.quantile(prior_samples["alpha"], 0.5), np.quantile(prior_samples["beta"], 0.5)))
fig = ff.create_distplot(li.tolist(), ["abc"])
fig.update_layout(title="Prior distribution of parameters", xaxis_title="parameter values", yaxis_title="density", legend_title="parameters")
fig.show()


# In[10]:


import torch
import pyro
import pandas as pd
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import pyro.distributions as dist
import seaborn as sns
import plotly
import plotly.express as px
import plotly.figure_factory as ff

import numpy as np

pyro.set_rng_seed(1)

plt.style.use('default')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')


# In[ ]:




