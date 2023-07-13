#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd


# In[11]:


import numpy as np


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


df = pd.read_csv("Desktop/SAMPLE.csv")


# In[14]:


df


# In[15]:


df.head()


# In[16]:


df.tail(10)


# In[17]:


df.info()


# In[18]:


df.isnull()


# In[19]:


df.shape


# In[20]:


plt.bar(df['Year'],df['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)'])
plt.xlabel("Year")
plt.ylabel("Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)")
plt.title("Year & Prevalence - Schizophrenia")
plt.show()


# In[69]:


plt.boxplot(df['Bipolar_disorder'])
plt.title("Prevalence - Bipolar disorder")
plt.show()


# In[22]:


plt.scatter(df['Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)'],df['Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)'])
plt.xlabel("Prevalence - Eating disorders")
plt.ylabel("Prevalence - Anxiety disorders")
plt.title("Prevalence - Eating disorders & Prevalence - Anxiety disorders")
plt.show()


# In[23]:


plt.hist(df['Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)'])
plt.title("Prevalence - Drug use disorders")
plt.show()


# In[24]:


df1=df.set_axis(['Entity','Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol','mental_fitness'], axis='columns', inplace=True)


# In[25]:


import seaborn as sns
sns.pairplot(df,corner=True)
plt.show()


# In[41]:


Features = df.iloc[:,:-1]


# In[42]:


Features


# In[43]:


Labels = df.iloc[:,-1:]


# In[44]:


Labels


# In[45]:


import sklearn


# In[46]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(Features, Labels, test_size=0.2, random_state=2)


# In[47]:


from sklearn.ensemble import RandomForestRegressor


# In[48]:


regressor = RandomForestRegressor(n_estimators=100,random_state=0)


# In[50]:


regressor.fit(xtrain,ytrain)


# In[51]:


ytrain_pred = regressor.predict(xtrain)


# In[52]:


from sklearn.metrics import mean_squared_error , r2_score


# In[53]:


ytrain_pred = regressor.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)


# In[54]:


print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[55]:


ytest_pred = regressor.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)


# In[56]:


print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[70]:


x_new = np.arange(8).reshape((-1, 8))


# In[71]:


x_new


# In[72]:


y_new = regressor.predict(x_new)


# In[73]:


y_new

