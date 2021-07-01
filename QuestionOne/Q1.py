
print('Student Name: Fan Yang '
      'Student ID: 20104813')
#Question 1
#Importing Libraries
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
print('Question 1 answer')
#Loading Data
Path = "data.csv"
names = ['id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean','smoothness_mean', 'compactness_mean', 'concavity_mean',
          'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concavity points_ mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
          'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'area_worst', 'smoothness_worst',
          'campactness_worst','concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal dimension_worst','diagnosis',]
Rawdatas = pandas.read_csv(Path)
array = Rawdatas.values
nrow, ncol = Rawdatas.shape
# X, Y split .
predict = Rawdatas.iloc[:,:ncol-1]
object = Rawdatas.iloc[:,-1]
print('predict:',predict)
print('object:',object)
#random the iteration
Predict_train, Predict_test, object_train, object_test  = train_test_split(predict, object, test_size= 0.3)

for loo in [50, 100, 150, 200, 300, 400, 500, ]:
    clfr = MLPClassifier( hidden_layer_sizes=(25), max_iter=loo)
    clfr.fit(Predict_train,np.ravel(object_train,order='C'))
    predict = clfr.predict(Predict_test)
    print("Iteration Values:", loo, " and Accuracy Values :", accuracy_score(object_test, predict))

