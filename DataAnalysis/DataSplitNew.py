#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts')
from Libs import *
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df_final = pd.read_csv('C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts/DataMerge/df_final.csv', index_col=0)


# In[4]:


df_final.head()


# In[5]:


df_final['amount'] = df_final['amount'].fillna(0)
df_final['conflict-1-to-3'] = df_final['conflict-1-to-3'].fillna(False)
df_final['conflict-4-to-6'] = df_final['conflict-4-to-6'].fillna(False)
df_final['conflict-7-to-9'] = df_final['conflict-7-to-9'].fillna(False)
df_final['Prom Tools'] = df_final['Prom Tools'].fillna(0)
df_final['Prom Vehicles'] = df_final['Prom Vehicles'].fillna(0)
df_final['Prom Weapons'] = df_final['Prom Weapons'].fillna(0)
df_final['Weapons'] = df_final['Weapons'].fillna(0)
df_final['Vehicles'] = df_final['Vehicles'].fillna(0)
df_final['Tools'] = df_final['Tools'].fillna(0)
df_final['conflict'] = df_final['conflict'].fillna(False)


# In[6]:


df_final


# In[ ]:




