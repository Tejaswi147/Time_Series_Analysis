#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv('BAJFINANCE.csv')
df.head()


# In[4]:


df.set_index('Date',inplace=True)


# #### Plotting the target variable VWAP over time

# In[5]:


df['VWAP'].plot()


# ### Data Pre-Processing

# In[6]:


df.shape


# In[7]:


df.isna().sum()


# In[8]:


df.dropna(inplace=True)


# In[9]:


df.isna().sum()


# In[10]:


df.shape


# In[11]:


data=df.copy()


# In[12]:


data.dtypes


# In[13]:


data.columns


# In[14]:


lag_features=['High','Low','Volume','Turnover','Trades']
window1=3
window2=7


# In[15]:


for feature in lag_features:
    data[feature+'rolling_mean_3']=data[feature].rolling(window=window1).mean()
    data[feature+'rolling_mean_7']=data[feature].rolling(window=window2).mean()


# In[16]:


for feature in lag_features:
    data[feature+'rolling_std_3']=data[feature].rolling(window=window1).std()
    data[feature+'rolling_std_7']=data[feature].rolling(window=window2).std()


# In[17]:


data.head()


# In[18]:


data.columns


# In[19]:


data.shape


# In[20]:


data.isna().sum()


# In[21]:


data.dropna(inplace=True)


# In[22]:


data.columns


# In[23]:


ind_features=['Highrolling_mean_3', 'Highrolling_mean_7',
       'Lowrolling_mean_3', 'Lowrolling_mean_7', 'Volumerolling_mean_3',
       'Volumerolling_mean_7', 'Turnoverrolling_mean_3',
       'Turnoverrolling_mean_7', 'Tradesrolling_mean_3',
       'Tradesrolling_mean_7', 'Highrolling_std_3', 'Highrolling_std_7',
       'Lowrolling_std_3', 'Lowrolling_std_7', 'Volumerolling_std_3',
       'Volumerolling_std_7', 'Turnoverrolling_std_3', 'Turnoverrolling_std_7',
       'Tradesrolling_std_3', 'Tradesrolling_std_7']


# In[24]:


training_data=data[0:1800]
test_data=data[1800:]


# In[25]:


training_data


# In[26]:


get_ipython().system('pip install pmdarima')


# In[27]:


from pmdarima import auto_arima


# In[28]:


import warnings
warnings.filterwarnings('ignore')


# In[29]:


model=auto_arima(y=training_data['VWAP'],exogenous=training_data[ind_features],trace=True)


# In[30]:


model.fit(training_data['VWAP'],training_data[ind_features])


# In[31]:


forecast=model.predict(n_periods=len(test_data), exogenous=test_data[ind_features])


# In[32]:


test_data['Forecast_ARIMA']=forecast


# In[33]:


test_data[['VWAP','Forecast_ARIMA']].plot(figsize=(14,7))


# In[34]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[35]:


np.sqrt(mean_squared_error(test_data['VWAP'],test_data['Forecast_ARIMA']))


# In[36]:


mean_absolute_error(test_data['VWAP'],test_data['Forecast_ARIMA'])

