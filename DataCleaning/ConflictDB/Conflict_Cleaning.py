
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, 'C:/Users/Franco/AnacondaProjects/Proyecto-Investigacion')
from Libs import *


# In[2]:


# Reading 2 csv

conflict_csv = pd.read_csv('C:/Users/Franco/AnacondaProjects/Proyecto-Investigacion/Data Cleaning/Conflict DB/ucdp-prio-acd-172.csv', index_col='conflict_id')
conflict2_csv = pd.read_csv('C:/Users/Franco/AnacondaProjects/Proyecto-Investigacion/Data Cleaning/Conflict DB/ucdp-dyadic-172.csv', index_col='conflict_id')

# We'll use conflict2

# Cleaning location column

conflict2_csv['location'] = conflict2_csv['location'].str.strip()
conflict2_csv['location'] = conflict2_csv['location'].str.strip().str.replace('Myanmar \(Burma\)', 'Myanmar').str.replace('Cambodia \(Kampuchea\)','Cambodia').str.replace('Russia \(Soviet Union\)','Russia').str.replace('Serbia \(Yugoslavia\)','Serbia').str.replace('Madagascar (Malagasy)', 'Madagascar')


# In[3]:


# We're going to use only this two columns of this csv (location and year)

df_conflicto = conflict2_csv.loc[:, ['location', 'year']]


# In[4]:


# Splitting up the rows which has two or more countries names values and creating one individual row for country per year

df_conflicto_new = pd.DataFrame(df_conflicto['location'].str.split(',').tolist(), index=df_conflicto['year']).stack().reset_index(level=1, drop=True).reset_index(name='location')
df_conflicto_new['location'] = df_conflicto_new['location'].str.strip()


# In[9]:


# Converting the dataframe into CSV

df_conflicto_new.to_csv('df_conflicto.csv')

