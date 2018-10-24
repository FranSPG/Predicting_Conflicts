
# coding: utf-8

# In[3]:


import sys
sys.path.insert(0, 'C:/Users/Franco/AnacondaProjects/Predicting_Conflicts')
from Libs import *
sys.path.insert(0, 'C:/Users/Franco/AnacondaProjects/Predicting_Conflicts/DataAnalisys')
from DataSplit import *


# In[5]:


# Logistic Regression

# Modelo con todas las variables
classifier = linear_model.LogisticRegression()
classifier.fit(X_train, y_train.values.ravel())

# Modelo sin las variables conflict
classifier_sin_conflictos = linear_model.LogisticRegression()
classifier_sin_conflictos.fit(X_train_sin_conflictos, y_train_sin_conflictos.values.ravel())

# Modelo únicamente con las variables conflict
classifier_solo_conflictos = linear_model.LogisticRegression()
classifier_solo_conflictos.fit(X_train_solo_conflictos, y_train_solo_conflictos.values.ravel())

# Modelos con 1 variable conflicto a la vez

classifier_c3 = linear_model.LogisticRegression()
classifier_c3.fit(X_train_c3, y_train_c3.values.ravel())

classifier_c5 = linear_model.LogisticRegression()
classifier_c5.fit(X_train_c5, y_train_c5.values.ravel())

classifier_c10 = linear_model.LogisticRegression()
classifier_c10.fit(X_train_c10, y_train_c10.values.ravel())


# In[6]:


print("\nAccuracy\n")
y_pred_logregr = classifier.predict(X_test)
y_pred_logregr_sin_conflictos = classifier_sin_conflictos.predict(X_test_sin_conflictos)
y_pred_logregr_solo_conflictos = classifier_solo_conflictos.predict(X_test_solo_conflictos)
print("Logistic Regression\n\nEntero: {}\nSin conflictos: {}\nSolo conflictos: {}\n".format(accuracy_score(y_test, y_pred_logregr), accuracy_score(y_test_sin_conflictos, y_pred_logregr_sin_conflictos), accuracy_score(y_test_solo_conflictos, y_pred_logregr_solo_conflictos)))

y_pred_lr_c3 = classifier_c3.predict(X_test_c3)
y_pred_lr_c5 = classifier_c5.predict(X_test_c5)
y_pred_lr_c10 = classifier_c10.predict(X_test_c10)

print("Pasando de a 1 conflict a la vez:\n\n conflict-3: {}\n\n conflict-5: {}\n\n conflict-10: {}\n".format(accuracy_score(y_test_c3, y_pred_lr_c3), accuracy_score(y_test_c5, y_pred_lr_c5), accuracy_score(y_test_c10, y_pred_lr_c10)))

# Accuracy con el dataframe de validación (Todas las variables)
y_pred_logregr_validation = classifier.predict(df_validation_data)
print("\nLogistic Regression validation dataframe\n{}".format(accuracy_score(df_validation_target, y_pred_logregr_validation)))


# In[7]:


# Matriz de confusión
cm_logreg = confusion_matrix(y_test, y_pred_logregr)
print("\nLogRegression dataset entero:\n {}".format(cm_logreg))

# Matriz de confusión dataframe de validacion
cm_logreg_validation = confusion_matrix(df_validation_target, y_pred_logregr_validation)
print("\nLogRegression validation dataframe:\n {}".format(cm_logreg_validation))

# Matriz de confusion variables conflict
cm_lr_c3 = confusion_matrix(y_test_c3, y_pred_lr_c3)
cm_lr_c5 = confusion_matrix(y_test_c5, y_pred_lr_c5)
cm_lr_c10 = confusion_matrix(y_test_c10, y_pred_lr_c10)
print("\nConflict-3 \n{}\nConflict-5 \n{}\nConflict-10 \n{}\n".format(cm_lr_c3, cm_lr_c5, cm_lr_c10))

