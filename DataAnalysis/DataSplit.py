#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts')
from Libs import *


# In[2]:


df_final = pd.read_csv('C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts/DataMerge/df_final.csv', index_col=0)


# In[3]:


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


# In[5]:


# Mezclo el dataframe df_final y hago un slice con todos los valores que tengan el campo año < a 2017
# El dataset de conflictos tiene información hasta el año 2016
# El dataset de transferencias de armas tiene información hasta el año 2017
df_mixed = pd.DataFrame(df_final[df_final['year'] < 2018].sample(frac=1))

# Dataframe con el slice de los datos con la columna year > 2016
df_to_predict = pd.DataFrame(df_final[df_final['year'] > 2016])
df_to_predict['conflict'] = np.nan


# Distribuyendo equitativamente los valores con conflict true y false
# df_validation tiene un 10% del total para hacer una re validación

df_mixed_true = df_mixed[df_mixed['conflict'] == True]
df_mixed_false = df_mixed[df_mixed['conflict'] == False]

ten_percent_true = int(len(df_mixed_true)*0.1)
ten_percent_false = int(len(df_mixed_false)*0.1)

df_validation = pd.DataFrame(pd.concat([df_mixed_true[:ten_percent_true], (df_mixed_false[:ten_percent_false])]))                            
df_train_models = pd.DataFrame(pd.concat([df_mixed_true[ten_percent_true:], df_mixed_false[ten_percent_false:]]))

df_validation_data = pd.DataFrame(df_validation.loc[:, ['country encoded', 'year', 'amount', 'Tools', 'Vehicles', 'Weapons', 'Prom Tools', 'Prom Vehicles', 'Prom Weapons', 'conflict-1-to-3', 'conflict-4-to-6', 'conflict-7-to-9', 'Prom USA']])
df_validation_target = pd.DataFrame(df_validation.loc[:, ['conflict']])


# Validacion para cuando se le pasa 1 variables de conflict
df_data_c3_validation = pd.DataFrame(df_validation.loc[:, ['country encoded', 'year', 'amount', 'Tools', 'Vehicles', 'Weapons', 'Prom Tools', 'Prom Vehicles', 'Prom Weapons', 'conflict-1-to-3', 'Prom USA']])
df_target_c3_validation = pd.DataFrame(df_validation.loc[:, ['conflict']])

df_data_c5_validation = pd.DataFrame(df_validation.loc[:, ['country encoded', 'year', 'amount', 'Tools', 'Vehicles', 'Weapons', 'Prom Tools', 'Prom Vehicles', 'Prom Weapons', 'conflict-4-to-6', 'Prom USA']])
df_target_c5_validation = pd.DataFrame(df_validation.loc[:, ['conflict']])

df_data_c10_validation = pd.DataFrame(df_validation.loc[:, ['country encoded', 'year', 'amount', 'Tools', 'Vehicles', 'Weapons', 'Prom Tools', 'Prom Vehicles', 'Prom Weapons', 'conflict-7-to-9', 'Prom USA']])
df_target_c10_validation = pd.DataFrame(df_validation.loc[:, ['conflict']])

#df_grouped = pd.DataFrame(df_validation.loc[])


# In[23]:


df_grouped = pd.DataFrame(df_validation.loc[:, ['year', 'Tools', 'Vehicles', 'Weapons', 'amount', 'conflict', 'conflict-1-to-3', 'conflict-4-to-6', 'conflict-7-to-9', 'country encoded']])
df_grouped['Armament'] = df_grouped[['Tools', 'Vehicles', 'Weapons']].sum(1)

df_grouped_data = pd.DataFrame(df_grouped.loc[:, ['year','Armament', 'amount', 'conflict-1-to-3', 'conflict-4-to-6', 'conflict-7-to-9', 'country encoded']])
df_grouped_target = pd.DataFrame(df_grouped.loc[:, ['conflict']])


# In[8]:


# Verifricación que ninguna row se repite en los 2 datasets (df_validation, df_train_models)
df_validation.index.isin(df_train_models.index).any()


# In[21]:


# Splits del dataframe de entrenamiento

# Split con todas las variables
X_train, X_test, y_train, y_test = train_test_split(df_train_models.loc[:, ['country encoded', 'year', 'amount', 'Tools', 'Vehicles', 
                                                                             'Weapons', 'Prom Tools', 'Prom Vehicles', 'Prom Weapons', 'conflict-1-to-3', 
                                                                             'conflict-4-to-6', 'conflict-7-to-9', 'Prom USA']],
                                                            df_train_models.loc[:, ['conflict']],
                                                            test_size = 0.25, random_state=np.random.randint(100000))


# Split sin las variables conflict-3, conflict-5, conflict-10
X_train_sin_conflictos, X_test_sin_conflictos, y_train_sin_conflictos, y_test_sin_conflictos = train_test_split(df_train_models.loc[:, ['country encoded', 'year', 'amount', 'Tools', 'Vehicles', 
                                                                                                                                         'Weapons', 'Prom Tools', 'Prom Vehicles', 'Prom Weapons', 'Prom USA']],
                                                                                                                df_train_models.loc[:, ['conflict']],
                                                                                                                test_size = 0.25, random_state=np.random.randint(100000))

# Split unicamente con las variables conflict-3, conflict-5, conflict-10
X_train_solo_conflictos, X_test_solo_conflictos, y_train_solo_conflictos, y_test_solo_conflictos = train_test_split(df_train_models.loc[:, ['country encoded', 'year', 'conflict-1-to-3', 'conflict-4-to-6', 'conflict-7-to-9']],
                                                                                                                    df_train_models.loc[:, ['conflict']],
                                                                                                                    test_size = 0.25, random_state=np.random.randint(100000))

# Splits variando una sola variable de conflict-3, conflict-5, conflict-10
X_train_c3, X_test_c3, y_train_c3, y_test_c3 = train_test_split(df_train_models.loc[:, ['country encoded', 'year', 'amount', 'Tools', 'Vehicles', 
                                                                                         'Weapons', 'Prom Tools', 'Prom Vehicles', 'Prom Weapons', 'conflict-1-to-3', 
                                                                                         'Prom USA']],
                                                                df_train_models.loc[:, ['conflict']],
                                                                test_size = 0.25, random_state=np.random.randint(100000))

X_train_c5, X_test_c5, y_train_c5, y_test_c5 = train_test_split(df_train_models.loc[:, ['country encoded', 'year', 'amount', 'Tools', 'Vehicles', 
                                                                                         'Weapons', 'Prom Tools', 'Prom Vehicles', 'Prom Weapons', 'conflict-4-to-6', 
                                                                                         'Prom USA']],
                                                                df_train_models.loc[:, ['conflict']],
                                                                test_size = 0.25, random_state=np.random.randint(100000))

X_train_c10, X_test_c10, y_train_c10, y_test_c10 = train_test_split(df_train_models.loc[:, ['country encoded', 'year', 'amount', 'Tools', 'Vehicles', 
                                                                                         'Weapons', 'Prom Tools', 'Prom Vehicles', 'Prom Weapons', 'conflict-7-to-9', 
                                                                                         'Prom USA']],
                                                                df_train_models.loc[:, ['conflict']],
                                                                test_size = 0.25, random_state=np.random.randint(100000))

X_train_grouped, X_test_grouped, y_train_grouped, y_test_grouped = train_test_split(df_grouped_data,
                                                                                   df_grouped_target,
                                                                                   test_size = 0.25,
                                                                                   random_state=np.random.randint(100000)) 


# In[ ]:




