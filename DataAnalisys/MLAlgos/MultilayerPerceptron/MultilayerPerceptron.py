
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts')
from Libs import *
sys.path.insert(0, 'C:/Users/Franco/Documents/AnacondaProjects/Predicting_Conflicts/DataAnalisys')
from DataSplit import *


# In[2]:


# Multilayer Perceptron

# Modelo con todas las variables
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X_train, y_train.values.ravel())

# Modelo sin las variables conflict
mlp_sin_conflictos = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp_sin_conflictos.fit(X_train_sin_conflictos, y_train_sin_conflictos.values.ravel())

# Modelo únicamente con las variables conflict
mlp_solo_conflictos = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp_solo_conflictos.fit(X_train_solo_conflictos, y_train_solo_conflictos.values.ravel())

# Modelos con 1 variable conflicto a la vez

mlp_c3 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp_c3.fit(X_train_c3, y_train_c3.values.ravel())
mlp_c5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp_c5.fit(X_train_c5, y_train_c5.values.ravel())
mlp_c10 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp_c10.fit(X_train_c10, y_train_c10.values.ravel())


# In[3]:


print("\nAccuracy\n")
y_pred_mlp = mlp.predict(X_test)
y_pred_mlp_sin_conflictos = mlp_sin_conflictos.predict(X_test_sin_conflictos)
y_pred_mlp_solo_conflictos = mlp_solo_conflictos.predict(X_test_solo_conflictos)
print("\nMulti Layer Perceptron\n\nEntero: {}\nSin conflictos: {}\nSolo conflictos: {}\n".format(accuracy_score(y_test, y_pred_mlp), accuracy_score(y_test_sin_conflictos, y_pred_mlp_sin_conflictos), accuracy_score(y_test_solo_conflictos, y_pred_mlp_solo_conflictos)))

y_pred_mlp_c3 = mlp_c3.predict(X_test_c3)
y_pred_mlp_c5 = mlp_c5.predict(X_test_c5)
y_pred_mlp_c10 = mlp_c10.predict(X_test_c10)
print("\nPasando de a 1 conflict a la vez:\n\n conflict-3: {}\n\n conflict-5: {}\n\n conflict-10: {}\n".format(accuracy_score(y_test_c3, y_pred_mlp_c3), accuracy_score(y_test_c5, y_pred_mlp_c5), accuracy_score(y_test_c10, y_pred_mlp_c10)))

# Accuracy con el dataframe de validación
y_pred_mlp_validation = mlp.predict(df_validation_data)
print("\nMLP validation dataframe\n{}".format(accuracy_score(df_validation_target, y_pred_mlp_validation)))


# In[4]:


# Matriz de confusión
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
print("\nMLP:\n {}".format(cm_mlp))

# Matriz de confusion dataframe de validacion
cm_mlp_validation = confusion_matrix(df_validation_target, y_pred_mlp_validation)
print("\nMLP validation dataframe:\n {}".format(cm_mlp_validation))

cm_mlp_c3 = confusion_matrix(y_test_c3, y_pred_mlp_c3)
cm_mlp_c5 = confusion_matrix(y_test_c5, y_pred_mlp_c5)
cm_mlp_c10 = confusion_matrix(y_test_c10, y_pred_mlp_c10)

# Matriz de confusion variables conflict
print("\nConflict-3 \n{}\nConflict-5 \n{}\nConflict-10 \n{}\n".format(cm_mlp_c3, cm_mlp_c5, cm_mlp_c10))

