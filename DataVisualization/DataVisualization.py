
# coding: utf-8

# In[9]:


import sys
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts')
from Libs import *

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')


# In[2]:


df = pd.read_csv('C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts/DataVisualization/df_final_ver_3.csv', index_col=0)


# In[3]:


df.head()


# In[4]:


df_argentina = df[df['country'] == 'Argentina'].sort_values('year')

df_argentina = df_argentina.iloc[:, 0:10]
df_argentina.head()


# In[28]:


fig = plt.figure()
ax1 = plt.subplot2grid((7,1), (0,0), rowspan=6, colspan=1)
ax1.grid(True)
plt.title('Argentina')
plt.ylabel('Cantidad de armas')
plt.xlabel('Año')




ax1.set_xticks(df_argentina.iloc[:, 0])
plt.xticks(rotation=50)

ax1.plot(df_argentina.iloc[:, 0], df_argentina.iloc[:,5])
#ax1.plot(df_argentina.iloc[:, 0], df_argentina.iloc[:,2])

ax1.fill_between(df_argentina.iloc[:, 0],0,  df_argentina.iloc[:, 5], where=(df_argentina['conflict'] == True), facecolor='r', alpha=0.5, interpolate=True)
ax1.fill_between(df_argentina.iloc[:, 0], df_argentina.iloc[:, 5], where=(df_argentina['conflict'] == False), facecolor='g', alpha=0.5, interpolate=True)

#ax1.fill_between(df_argentina.iloc[:, 0], df_argentina.iloc[:, 8], alpha=0.3)

#ax1.plot(df_argentina.iloc[:, 0], df_argentina.iloc[:,4])
#ax1.plot(df_argentina.iloc[:, 0], df_argentina.iloc[:,3])
#ax1.plot(df_argentina.iloc[:, 0], df_argentina.iloc[:,2])
plt.savefig('argentina.png')
plt.show()

