
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, 'C:/Users/Franco/AnacondaProjects/Proyecto-Investigacion')
from Libs import *


# In[11]:


trade_csv = pd.read_csv('C:/Users/Franco/AnacondaProjects/Proyecto-Investigacion/Trade Register/Trade-Register-Recipient-1950-2017.csv')
trade_with_types_csv = pd.read_csv('C:/Users/Franco/AnacondaProjects/Proyecto-Investigacion/Trade Register/TradeWithTypes.csv')
trade_with_types_csv_copy = pd.DataFrame(trade_with_types_csv)


trade_with_types_csv['Suppliers'] = trade_with_types_csv['Suppliers'].str.strip()
trade_with_types_csv['Designation'] = trade_with_types_csv['Designation'].str.strip()
trade_with_types_csv['Recipient'] = trade_with_types_csv['Recipient'].str.strip()
trade_with_types_csv['Comments'] = trade_with_types_csv['Comments'].str.strip().str.replace('(\n)', '')
trade_with_types_csv['Year Weapon of order'] = trade_with_types_csv['Year Weapon of order'].str.strip().str.replace('\(', '').str.replace('\)', '')
trade_with_types_csv['Ordered'] = trade_with_types_csv['Ordered'].str.strip().str.replace('\(', '').str.replace('\)', '')
trade_with_types_csv['Or delivered'] = trade_with_types_csv['Or delivered'].str.strip().str.replace('\(', '').str.replace('\)', '')

index = trade_with_types_csv.index[trade_with_types_csv['Recipient'].str.contains('\*')]
index = index.tolist()
trade_with_types_csv.drop(trade_with_types_csv.index[index], inplace=True)

# Deleting rows with 'Missing[]' values
trade_with_types_csv.drop(trade_with_types_csv[trade_with_types_csv['Ordered'] == 'Missing[]'].index, inplace=True)

# Transforming the 'Ordered' column to float
trade_with_types_csv.Ordered = pd.to_numeric(trade_with_types_csv['Ordered'], errors='coerce')


# In[14]:


# Grouping by 'Recipient', 'Year Weapon of order', 'Type' of weapon.
trade_with_types_csv = pd.DataFrame(trade_with_types_csv.groupby(['Recipient', 'Year Weapon of order', 'Type'])['Ordered'].sum().unstack())

# Resetting the index
trade_with_types_csv.reset_index(inplace=True)


# In[16]:


# Creating the average columns 

trade_with_types_csv['Prom Tools'] = np.zeros(len(trade_with_types_csv))
trade_with_types_csv['Prom Vehicles'] = np.zeros(len(trade_with_types_csv))
trade_with_types_csv['Prom Weapons'] = np.zeros(len(trade_with_types_csv))


# In[17]:


# Converting the dataframe into CSV

trade_with_types_csv.to_csv('TradeClean.csv')

