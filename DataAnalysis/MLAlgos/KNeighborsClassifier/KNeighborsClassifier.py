
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts')
from Libs import *
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts/DataAnalisys')
from DataSplit import *


# In[2]:


# KNeighbors Classifier

# Modelo con todas las variables
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train.values.ravel())

# Modelo sin las variables conflict
knn_sin_conflictos = KNeighborsClassifier(n_neighbors=15)
knn_sin_conflictos.fit(X_train_sin_conflictos, y_train_sin_conflictos.values.ravel())

# Modelo únicamente con las variables conflict
knn_solo_conflictos = KNeighborsClassifier(n_neighbors=15)
knn_solo_conflictos.fit(X_train_solo_conflictos, y_train_solo_conflictos.values.ravel())

# Modelos con 1 variable conflicto a la vez

knn_c3 = KNeighborsClassifier(n_neighbors=15)
knn_c3.fit(X_train_c3, y_train_c3.values.ravel())
knn_c5 = KNeighborsClassifier(n_neighbors=15)
knn_c5.fit(X_train_c5, y_train_c5.values.ravel())
knn_c10 = KNeighborsClassifier(n_neighbors=15)
knn_c10.fit(X_train_c10, y_train_c10.values.ravel())


# In[3]:


print("\nAccuracy\n")
y_pred_knn = knn.predict(X_test)
y_pred_knn_sin_conflictos = knn_sin_conflictos.predict(X_test_sin_conflictos)
y_pred_knn_solo_conflictos = knn_solo_conflictos.predict(X_test_solo_conflictos)
print("\nKNeighborsClassifier\n\nEntero: {}\nSin conflictos: {}\nSolo conflictos: {}\n".format(accuracy_score(y_test, y_pred_knn), accuracy_score(y_test_sin_conflictos, y_pred_knn_sin_conflictos), accuracy_score(y_test_solo_conflictos, y_pred_knn_solo_conflictos)))

y_pred_knn_c3 = knn_c3.predict(X_test_c3)
y_pred_knn_c5 = knn_c5.predict(X_test_c5)
y_pred_knn_c10 = knn_c10.predict(X_test_c10)
print("\nPasando de a 1 conflict a la vez:\n\n conflict-3: {}\n\n conflict-5: {}\n\n conflict-10: {}\n".format(accuracy_score(y_test_c3, y_pred_knn_c3), accuracy_score(y_test_c5, y_pred_knn_c5), accuracy_score(y_test_c10, y_pred_knn_c10)))

# Accuracy con el dataframe de validación
y_pred_knn_validation = knn.predict(df_validation_data)
print("\nKNN validation dataset\n{}".format(accuracy_score(df_validation_target, y_pred_knn_validation)))


# In[4]:


# Matriz de confusión
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("\nKNN:\n {}".format(cm_knn))

# Matriz de confusion dataframe de validacion
cm_knn_validation = confusion_matrix(df_validation_target, y_pred_knn_validation)
print("\nKNN validation dataset:\n {}".format(cm_knn_validation))

cm_knn_c3 = confusion_matrix(y_test_c3, y_pred_knn_c3)
cm_knn_c5 = confusion_matrix(y_test_c5, y_pred_knn_c5)
cm_knn_c10 = confusion_matrix(y_test_c10, y_pred_knn_c10)

# Matriz de confusion variables conflict
print("\nConflict-3 \n{}\nConflict-5 \n{}\nConflict-10 \n{}\n".format(cm_knn_c3, cm_knn_c5, cm_knn_c10))

