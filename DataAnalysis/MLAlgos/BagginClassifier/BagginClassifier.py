
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts')
from Libs import *
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts/DataAnalisys')
from DataSplit import *


# In[2]:


# Baggin Classifier

# Modelo con todas las variables
bg = BaggingClassifier(base_estimator=None, n_estimators=1000, n_jobs=-1)
bg.fit(X_train, y_train.values.ravel())

# Modelo sin las variables conflict
bg_sin_conflictos = BaggingClassifier(base_estimator=None, n_estimators=1000, n_jobs=-1)
bg_sin_conflictos.fit(X_train_sin_conflictos, y_train_sin_conflictos.values.ravel())

# Modelo únicamente con las variables conflict
bg_solo_conflictos = BaggingClassifier(base_estimator=None, n_estimators=1000, n_jobs=-1)
bg_solo_conflictos.fit(X_train_solo_conflictos, y_train_solo_conflictos.values.ravel())

# Modelos con 1 variable conflicto a la vez

bg_c3 = BaggingClassifier(base_estimator=None, n_estimators=1000, n_jobs=-1)
bg_c3.fit(X_train_c3, y_train_c3.values.ravel())
bg_c5 = BaggingClassifier(base_estimator=None, n_estimators=1000, n_jobs=-1)
bg_c5.fit(X_train_c5, y_train_c5.values.ravel())
bg_c10 = BaggingClassifier(base_estimator=None, n_estimators=1000, n_jobs=-1)
bg_c10.fit(X_train_c10, y_train_c10.values.ravel())


# In[3]:


print("\nAccuracy\n")
y_pred_bg = bg.predict(X_test)
y_pred_bg_sin_conflictos = bg_sin_conflictos.predict(X_test_sin_conflictos)
y_pred_bg_solo_conflictos = bg_solo_conflictos.predict(X_test_solo_conflictos)
print("\nBagging Classifier\n\nEntero: {}\nSin conflictos: {}\nSolo conflictos: {}\n".format(accuracy_score(y_test, y_pred_bg), accuracy_score(y_test_sin_conflictos, y_pred_bg_sin_conflictos), accuracy_score(y_test_solo_conflictos, y_pred_bg_solo_conflictos)))

y_pred_bg_c3 = bg_c3.predict(X_test_c3)
y_pred_bg_c5 = bg_c5.predict(X_test_c5)
y_pred_bg_c10 = bg_c10.predict(X_test_c10)
print("\nPasando de a 1 conflict a la vez:\n\n conflict-3: {}\n\n conflict-5: {}\n\n conflict-10: {}\n".format(accuracy_score(y_test_c3, y_pred_bg_c3), accuracy_score(y_test_c5, y_pred_bg_c5), accuracy_score(y_test_c10, y_pred_bg_c10)))

# Accuracy con el dataframe de validación
y_pred_bg_validation = bg.predict(df_validation_data)
print("\nBG validation dataset\n{}".format(accuracy_score(df_validation_target, y_pred_bg_validation)))


# In[4]:


# Matriz de confusión
cm_bg = confusion_matrix(y_test, y_pred_bg)
print("\nBG:\n {}".format(cm_bg))

# Matriz de confusion dataframe de validacion
cm_bg_validation = confusion_matrix(df_validation_target, y_pred_bg_validation)
print("\nBG validation dataset:\n {}".format(cm_bg_validation))

cm_bg_c3 = confusion_matrix(y_test_c3, y_pred_bg_c3)
cm_bg_c5 = confusion_matrix(y_test_c5, y_pred_bg_c5)
cm_bg_c10 = confusion_matrix(y_test_c10, y_pred_bg_c10)

print("\nConflict-3 \n{}\nConflict-5 \n{}\nConflict-10 \n{}\n".format(cm_bg_c3, cm_bg_c5, cm_bg_c10))

