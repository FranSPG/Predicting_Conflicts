
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts')
from Libs import *
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts/DataAnalisys')
from DataSplit import *


# In[2]:


# Support Vector Classifier

# Modelo con todas las variables
svc = SVC()
svc.fit(X_train, y_train.values.ravel())

# Modelo sin las variables conflict
svc_sin_conflictos = SVC()
svc_sin_conflictos.fit(X_train_sin_conflictos, y_train_sin_conflictos.values.ravel())

# Modelo únicamente con las variables conflict
svc_solo_conflictos = SVC()
svc_solo_conflictos.fit(X_train_solo_conflictos, y_train_solo_conflictos.values.ravel())


# Modelos con 1 variable conflicto a la vez

svc_c3 = SVC()
svc_c3.fit(X_train_c3, y_train_c3.values.ravel())

svc_c5 = SVC()
svc_c5.fit(X_train_c5, y_train_c5.values.ravel())

svc_c10 = SVC()
svc_c10.fit(X_train_c10, y_train_c10.values.ravel())


# In[3]:


print("\nAccuracy\n")

y_pred_svc = svc.predict(X_test)
y_pred_svc_sin_conflictos = svc_sin_conflictos.predict(X_test_sin_conflictos)
y_pred_svc_solo_conflictos = svc_solo_conflictos.predict(X_test_solo_conflictos)
print("\nSupport Vector Classification\n\nEntero: {}\nSin conflictos: {}\nSolo conflictos: {}\n".format(accuracy_score(y_test, y_pred_svc), accuracy_score(y_test_sin_conflictos, y_pred_svc_sin_conflictos), accuracy_score(y_test_solo_conflictos, y_pred_svc_solo_conflictos)))

y_pred_svc_c3 = svc_c3.predict(X_test_c3)
y_pred_svc_c5 = svc_c5.predict(X_test_c5)
y_pred_svc_c10 = svc_c10.predict(X_test_c10)
print("Pasando de a 1 conflict a la vez:\n\n conflict-3: {}\n\n conflict-5: {}\n\n conflict-10: {}\n".format(accuracy_score(y_test_c3, y_pred_svc_c3), accuracy_score(y_test_c5, y_pred_svc_c5), accuracy_score(y_test_c10, y_pred_svc_c10)))

# Accuracy con el dataframe de validación
y_pred_svc_validation = svc.predict(df_validation_data)
print("\nSupport Vector Classification validation dataframe\n{}".format(accuracy_score(df_validation_target, y_pred_svc_validation)))


# In[4]:


# Matriz de confusión
cm_svc = confusion_matrix(y_test, y_pred_svc)
print("\nSVC:\n {}".format(cm_svc))

# Matriz de confusion dataframe de validacion
cm_svc_validation = confusion_matrix(df_validation_target, y_pred_svc_validation)
print("\nSVC validation dataframe:\n {}".format(cm_svc_validation))

# Matriz de confusion variables conflict
cm_svc_c3 = confusion_matrix(y_test_c3, y_pred_svc_c3)
cm_svc_c5 = confusion_matrix(y_test_c5, y_pred_svc_c5)
cm_svc_c10 = confusion_matrix(y_test_c10, y_pred_svc_c10)
print("\nConflict-3 \n{}\nConflict-5 \n{}\nConflict-10 \n{}\n".format(cm_svc_c3, cm_svc_c5, cm_svc_c10))

