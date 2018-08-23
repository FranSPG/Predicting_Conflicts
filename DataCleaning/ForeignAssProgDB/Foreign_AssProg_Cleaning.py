
# coding: utf-8

# In[2]:


import sys
sys.path.insert(0, 'C:/Users/Franco/AnacondaProjects/Proyecto-Investigacion')
from Libs import *


# In[3]:


# Reading 2 csv

total_csv = pd.read_csv('C:/Users/Franco/AnacondaProjects/Proyecto-Investigacion/Data Cleaning/Foreign AssProg DB/Foreign Assistance program_Totals.csv', index_col='key')
detail_csv = pd.read_csv('C:/Users/Franco/AnacondaProjects/Proyecto-Investigacion/Data Cleaning/Foreign AssProg DB/Foreign Assistance program_details.csv', index_col='key')


# In[4]:


# Dropping unnecessary columns

drop_total = ['expenditures', 'estimate', 'footnote', 'footnote2']
drop_detail = ['trainees',
               'recipient_unit', 'training_institution', 
               'location', 'course_title', 'model', 
               'eda_implementing_agency', 'eda_delivered_quantity',
               'eda_delivered_date', 'eda_unit_current_value',
               'eda_offered_current_value', 'eda_delivered_acquisition_value',
               'footnote']

total_csv.drop(drop_total, axis=1, inplace=True)
detail_csv.drop(drop_detail, axis=1, inplace=True)
detail_csv.drop(index= [20920, 27182], axis=0, inplace=True)
detail_csv.dropna(inplace=True)
total_csv.dropna(inplace=True)

# Cleaning country columns

total_csv['program'] = total_csv['program'].str.strip()
detail_csv['program'] = detail_csv['program'].str.strip()
detail_csv['country'] = detail_csv['country'].str.strip().str.replace('(\n)', ' ').str.replace('Trinidad-Tobago', 'Trinidad and Tobago').str.replace('Western Hemishpere Regional', 'Western Hemisphere Regional').str.replace('Columbia', 'Colombia').str.replace('EI Salvador', 'El Salvador').str.replace('Ethionia', 'Ethiopia').str.replace('Hungray', 'Hungary').str.replace('Latin America Regional', 'Latin America Region').str.replace('Morocoo', 'Morocco').str.replace('Nicaruaga', 'Nicaragua')
total_csv['country'] = total_csv['country'].str.strip().str.replace('(\n)', ' ').str.replace('Trinidad-Tobago', 'Trinidad and Tobago').str.replace('Western Hemishpere Regional', 'Western Hemisphere Regional').str.replace('Columbia', 'Colombia').str.replace('EI Salvador', 'El Salvador').str.replace('Ethionia', 'Ethiopia').str.replace('Hungray', 'Hungary').str.replace('Latin America Regional', 'Latin America Region').str.replace('Morocoo', 'Morocco').str.replace('Nicaruaga', 'Nicaragua').str.replace('Republic of Congo \(Brazzaville\)','Congo').str.replace('Democratic Republic of Congo \(Kinshasa\)','Congo')


# In[5]:


# Grouping by country and year, the total sum of spent amount

df_total = pd.DataFrame(total_csv.groupby(['country', 'year'])['amount'].sum())
df_total.reset_index(inplace=True)


# In[6]:


# Converting the dataframe into CSV

df_total.to_csv('df_total.csv')

