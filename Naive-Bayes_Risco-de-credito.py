
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


base = pd.read_csv('risco_credito.csv')


# In[3]:


previsores = base.iloc[:, 0:4].values


# In[4]:


classe = base.iloc[:, 4].values


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


labelencoder = LabelEncoder()


# In[10]:


previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])


# In[5]:


from sklearn.naive_bayes import GaussianNB


# In[6]:


classificador = GaussianNB()


# In[11]:


classificador.fit(previsores, classe)


# In[14]:


resultados = classificador.predict([[0,0,1,2], [3,0,0,0]])


# In[15]:


resultados


# In[16]:


print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)

