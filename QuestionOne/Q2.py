
print('Student Name: Fan Yang '
      'Student ID: 20104813')
#Question 2
#Importing useful Libraries
import pandas
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
print('question 2 answers')
#A generic function to display training loss and testing accuracy of MLPClassifier
def make_plots_all(MLPC, predictors_Training, Object_training, predictors_testing, Object_testing, ):
    Max_iters = 100
    error = []
    loss = []
    for lo in range(Max_iters):
        MLPC.fit(predictors_Training, Object_training)
        iterations_error = (1 - MLPC.score(predictors_testing, Object_testing))* 25 #Precision deviation values
        error.append(iterations_error)
        loss.append(MLPC.loss_)
    plt.plot(error, label='Error')
    plt.plot(loss, label='Loss')
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
array = Rawdatas.values
RowN,ColN = Rawdatas.shape

# To do the X & Y split.
Predict = Rawdatas.iloc[:,:ColN-1]
Object = Rawdatas.iloc[:,-1]
print('predict:',Predict)
print('object:',Object)

# Iteration Values set 400
Predict_training, Predict_testing, Object_training, Object_testing  = train_test_split(Predict, Object, test_size= 0.3)
for loo in [400 ]:
    clfr = MLPClassifier( hidden_layer_sizes=(25), max_iter=loo)
    clfr.fit(Predict_training,np.ravel(Object_training,order='C'))
    predict = clfr.predict(Predict_testing)
    print("Iteration Values:", loo, " and Accuracy Values :", accuracy_score(Object_testing, predict))
    make_plots_all(clfr, Predict_training, Object_training, Predict_testing, Object_testing)
