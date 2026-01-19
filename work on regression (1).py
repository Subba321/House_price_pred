#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math


# In[2]:


df = pd.read_csv("C:/Users/subba/OneDrive/Desktop/movie/Linear Regression.csv  -  version 1.0. 26-04-2025 22.42.csv")
df


# In[3]:


# Analyse the data 
#x = df.area
#y = df.price
#plt.scatter(x,y,marker="+",color="Red")
#                or 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(df.area,df.price,marker="+",color="Red")


# In[4]:


# deployment of the model
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[5]:


reg.predict([[3500]])


# In[6]:


reg.coef_ #the value of "m" in formula y = mx+b


# In[7]:


reg.intercept_ #the value of "b" in formula y = mx+b


# In[8]:


135.78767123*3300+180616.43835616432
#  m*x+b


# In[9]:


plt.scatter(df.area,df.price,marker="+",color="Red")
plt.plot(df.area,reg.predict(df[["area"]]),color = 'black')


# In[ ]:





# In[10]:


#*************linear Regression********************


# In[11]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model


# In[12]:


df = pd.read_csv("C:/Users/subba/OneDrive/Desktop/movie/Polynomial_Regression.csv")
df


# In[13]:


reg = linear_model.LinearRegression()


# In[14]:


reg


# In[16]:


reg.fit(df[['area','bedrooms','age']],df.price)


# In[17]:


reg.coef_


# In[18]:


reg.intercept_


# In[19]:


reg.predict([[3000,3,40]])


# In[22]:


reg.score(df[['area','bedrooms','age']],df.price)

