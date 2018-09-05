
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts')
from Libs import *
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts/DataAnalisys')
from DataSplit import *


# In[2]:


# Random Forest Classifier

# Modelo con todas las variables
rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=20)
rf.fit(X_train, y_train.values.ravel())

# Modelo sin las variables conflict
rf_sin_conflictos = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
rf_sin_conflictos.fit(X_train_sin_conflictos, y_train_sin_conflictos.values.ravel())

# Modelo únicamente con las variables conflict
rf_solo_conflictos = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
rf_solo_conflictos.fit(X_train_solo_conflictos, y_train_solo_conflictos.values.ravel())

# Modelos con 1 variable conflicto a la vez

rf_c3 = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
rf_c3.fit(X_train_c3, y_train_c3.values.ravel())

rf_c5 = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
rf_c5.fit(X_train_c5, y_train_c5.values.ravel())

rf_c10 = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
rf_c10.fit(X_train_c10, y_train_c10.values.ravel())


# In[3]:


print("\nAccuracy\n")
y_pred_randforest = rf.predict(X_test)
print(accuracy_score(y_test, y_pred_randforest))


y_pred_randforest_sin_conflictos = rf_sin_conflictos.predict(X_test_sin_conflictos)
y_pred_randforest_solo_conflictos = rf_solo_conflictos.predict(X_test_solo_conflictos)
print("Random Forest\n\nEntero: {}\nSin conflictos: {}\nSolo conflictos: {}\n".format(accuracy_score(y_test, y_pred_randforest), accuracy_score(y_test_sin_conflictos, y_pred_randforest_sin_conflictos), accuracy_score(y_test_solo_conflictos, y_pred_randforest_solo_conflictos)))

y_pred_rf_c3 = rf_c3.predict(X_test_c3)
y_pred_rf_c5 = rf_c5.predict(X_test_c5)
y_pred_rf_c10 = rf_c10.predict(X_test_c10)
print("Pasando de a 1 conflict a la vez:\n\n conflict-3: {}\n\n conflict-5: {}\n\n conflict-10: {}\n".format(accuracy_score(y_test_c3, y_pred_rf_c3), accuracy_score(y_test_c5, y_pred_rf_c5), accuracy_score(y_test_c10, y_pred_rf_c10)))

# Accuracy con el dataframe de validación (Todas las variables)
y_pred_randforest_validation = rf.predict(df_validation_data)
print("\nRandom Forest validation dataframe\n{}".format(accuracy_score(df_validation_target, y_pred_randforest_validation)))


# In[4]:


# Matriz de confusión
cm_rf = confusion_matrix(y_test, y_pred_randforest)
print("\nRandForest:\n {}".format(cm_rf))

## Matriz de confusión dataframe de validacion
cm_rf_validation = confusion_matrix(df_validation_target, y_pred_randforest_validation)
print("\nRandom Forest validation dataframe:\n{}".format(cm_rf_validation))

# Matriz de confusion variables conflict
cm_rf_c3 = confusion_matrix(y_test_c3, y_pred_rf_c3)
cm_rf_c5 = confusion_matrix(y_test_c5, y_pred_rf_c5)
cm_rf_c10 = confusion_matrix(y_test_c10, y_pred_rf_c10)
print("\nConflict-3 \n{}\nConflict-5 \n{}\nConflict-10 \n{}\n".format(cm_rf_c3, cm_rf_c5, cm_rf_c10))

