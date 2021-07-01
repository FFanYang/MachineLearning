# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Student Name: Fan Yang')

# See PyCharm help at https://www.jetbrains.com/help/pycharm
print('Question 1 answer')

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Student Name: Fan Yang')

# See PyCharm help at https://www.jetbrains.com/help/pycharm
print('Question 1 answer')


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Student Name: Fan Yang')

# See PyCharm help at https://www.jetbrains.com/help/pycharm


#Question 1
# Use the sklearn.MLPClassifier with default values for parameters and a single hiddenlayer
# with k= 25 neurons. Use default values for all parameters other than the number of iterations.
# Determine the best number for iteration that gives the highest  accuracy.
# Use  this  classification  accuracy  as  a  baseline  for  comparison in later parts of this question

#Importing Libraries
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

print('question 2 answers')

#A generic function to display training loss and testing accuracy of MLPClassifier
def make_plots_all(mlp, predictors_train, target_train, predictors_test, target_test, ):
    max_iter = 100
    error = []
    losses = []
    for i in range(max_iter):
        mlp.fit(predictors_train, target_train)
        iter_error = (1 - mlp.score(predictors_test, target_test))* 25 #精度偏差值
        error.append(iter_error)
        losses.append(mlp.loss_)
    plt.plot(error, label='Error')
    plt.plot(losses, label='Loss')
    plt.title("Error and Loss over iterations", fontsize=16)
    plt.xlabel('Iterations')
    plt.legend(loc='upper right')
    plt.show()
#Load Data
Path = "Breast Cancer Coimbra Data set R2.csv"
names = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA',
         'Leptin', 'Adiponectin', 'Resistin', 'MCP.1', 'Classification']
rawdata = pandas.read_csv(Path)
array = rawdata.values
nrow, ncol = rawdata.shape
#split X and y.
predictors = rawdata.iloc[:,:ncol-1]
print('pred:',predictors)
target = rawdata.iloc[:,-1]
print('tar:',target)

#random the iteration
pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target, test_size= 0.3)
for i in range(25):
    clf = MLPClassifier( hidden_layer_sizes=(i+1, 25-i), max_iter=400)
    clf.fit(pred_train,np.ravel(tar_train, order='C'))
    predictions = clf.predict(pred_test)
    print( i+1,",", 25-i,",",accuracy_score(tar_test, predictions))


