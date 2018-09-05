
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts')
from Libs import *
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts/DataAnalisys')
from DataSplit import *


# In[2]:


# Gaussian Naive Bayes

# Modelo con todas las variables
gnb = GaussianNB()
gnb.fit(X_train, y_train.values.ravel())

# Modelo sin las variables conflict
gnb_sin_conflictos = GaussianNB()
gnb_sin_conflictos.fit(X_train_sin_conflictos, y_train_sin_conflictos.values.ravel())

# Modelo únicamente con las variables conflict
gnb_solo_conflictos = GaussianNB()
gnb_solo_conflictos.fit(X_train_solo_conflictos, y_train_solo_conflictos.values.ravel())

# Modelos con 1 variable conflicto a la vez

gnb_c3 = GaussianNB()
gnb_c3.fit(X_train_c3, y_train_c3.values.ravel())
gnb_c5 = GaussianNB()
gnb_c5.fit(X_train_c5, y_train_c5.values.ravel())
gnb_c10 = GaussianNB()
gnb_c10.fit(X_train_c10, y_train_c10.values.ravel())


# In[3]:


print("\nAccuracy\n")
y_pred_gnb = gnb.predict(X_test)
y_pred_gnb_sin_conflictos = gnb_sin_conflictos.predict(X_test_sin_conflictos)
y_pred_gnb_solo_conflictos = gnb_solo_conflictos.predict(X_test_solo_conflictos)
print("\nGaussian Naive Bayes\n\nEntero: {}\nSin conflictos: {}\nSolo conflictos: {}\n".format(accuracy_score(y_test, y_pred_gnb), accuracy_score(y_test_sin_conflictos, y_pred_gnb_sin_conflictos), accuracy_score(y_test_solo_conflictos, y_pred_gnb_solo_conflictos)))

y_pred_gnb_c3 = gnb_c3.predict(X_test_c3)
y_pred_gnb_c5 = gnb_c5.predict(X_test_c5)
y_pred_gnb_c10 = gnb_c10.predict(X_test_c10)
print("\nPasando de a 1 conflict a la vez:\n\n conflict-3: {}\n\n conflict-5: {}\n\n conflict-10: {}\n".format(accuracy_score(y_test_c3, y_pred_gnb_c3), accuracy_score(y_test_c5, y_pred_gnb_c5), accuracy_score(y_test_c10, y_pred_gnb_c10)))

# Accuracy con el dataframe de validación
y_pred_gnb_validation = gnb.predict(df_validation_data)
print("\nGNB validation dataframe\n{}".format(accuracy_score(df_validation_target, y_pred_gnb_validation)))


# In[4]:


# Matriz de confusión
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
print("\nGNB:\n {}".format(cm_gnb))

# Matriz de confusion dataframe de validacion
cm_gnb_validation = confusion_matrix(df_validation_target, y_pred_gnb_validation)
print("\nGNB validation dataframe:\n {}".format(cm_gnb_validation))

# Matriz de confusion variables conflict
cm_gnb_c3 = confusion_matrix(y_test_c3, y_pred_gnb_c3)
cm_gnb_c5 = confusion_matrix(y_test_c5, y_pred_gnb_c5)
cm_gnb_c10 = confusion_matrix(y_test_c10, y_pred_gnb_c10)
print("\nConflict-3 \n{}\nConflict-5 \n{}\nConflict-10 \n{}\n".format(cm_gnb_c3, cm_gnb_c5, cm_gnb_c10))

