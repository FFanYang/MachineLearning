print('Student Name: Fan Yang '
      'Student ID: 20104813')
#Question 3
#Importing Libraries
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

print('question 3 answers')
#A generic function to display training loss and testing accuracy of MLPClassifier

def make_plots_all(MLPC, Predictors_Training,  Object_training, predictors_testing,  Object_testing, ):
    Max_iteration = 100
    Error = []
    Loss = []
    for loop in range(Max_iteration):
        MLPC.fit(Predictors_Training, Object_training,)
        iter_error = (1 - MLPC.score(predictors_testing, Object_testing))* 25 #Accuracy deviation value
        Error.append(iter_error)
        Loss.append(MLPC.loss_)
    plt.plot(Error, label='Error')
    plt.plot(Loss, label='Loss')
    plt.title("Error and Loss over iterations", fontsize=16)
    plt.xlabel('Iterations')
    plt.legend(loc='upper right')
    plt.show()
#Load Data
Path = "data.csv"
names = ['id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean','smoothness_mean', 'compactness_mean', 'concavity_mean',
          'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concavity points_ mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
          'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'area_worst', 'smoothness_worst',
          'campactness_worst','concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal dimension_worst','diagnosis',]
Rawdatas = pandas.read_csv(Path)
Arrays = Rawdatas.values
RowN,ColN = Rawdatas.shape
# To do the X & Y split.
Predict = Rawdatas.iloc[:,:ColN-1]
Object = Rawdatas.iloc[:,-1]
print('predict:',Predict)
print('object:',Object)

# Iteration Values
Predict_training, Predict_testing, Object_training, Object_testing  = train_test_split(Predict, Object, test_size= 0.3)
for loop1 in range(25):
    clfr = MLPClassifier( hidden_layer_sizes=(loop1+1, 25-loop1), max_iter=400)
    clfr.fit(Predict_training,np.ravel(Object_training,order='C'))
    prediction = clfr.predict(Predict_testing)
    print( loop1+1,",", 25-loop1,",",accuracy_score(Object_testing, prediction))
