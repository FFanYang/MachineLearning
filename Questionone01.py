# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Student Name: Fan Yang')

# See PyCharm help at https://www.jetbrains.com/help/pycharm
print('Question 1 answer')

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
print('Question 1 answer')

#Load Data
Path = "data.csv"
names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
         'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concavity_points']


#!!!!!names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']


rawdata = pandas.read_csv(Path, names=names)
array = rawdata.values
nrow, ncol = rawdata.shape
predictors = array[:,0:8]
target = array[:, 8]

print(rawdata) #test to print rawdata

# The funcation to see some of the attributes of Neural Network
def NN_properties(model):
    loss_values = model.loss_
    print("Loss", loss_values)
    iterations = model.n_iter_
    print("iterations", iterations)
    classes_assigned = model.classes_
    print("Assigned classes", classes_assigned)

#Using loss_curve method with the MLP default solver “adam”
def make_plots_default(model):
    plt.plot(model.loss_curve_)
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

###A generic function to display training loss and testing accuracy of MLPClassifier
def make_plots_all(mlp, target_train, target_test,
                   predictors_test,predictors_train):
    max_iter = 100
    accuracy = []
    losses = []
    for i in range(max_iter):
        mlp.fit(predictors_train, target_train)
        iter_acc = mlp.score(predictors_test, target_test)
        accuracy.append(iter_acc)
        losses.append(mlp.loss_)
        plt.plot(accuracy, label='Test accuracy')

        plt.plot(losses, label='Loss')
        plt.title("Accuracy and Loss over Interations", fontsize=14)
        plt.xlabel('Iterations')
        plt.legend(loc='upper right')
        plt.show()

####A function for model building and calculating accuracy
def get_accuracy(target_train, target_test,
                 predictors_test,predictors_train):

    # a single hidden layers with k=25 neurons  -NN
    clf = MLPClassifier(solver='adam', learning_rate_init=0.01, hidden_layer_sizes=(25),
                        random_state=1, max_iter=200, warm_start=True)

    #Calling the make_plots_allfunction with unfitted model
    make_plots_all(clf, target_train, target_test, predictors_test, predictors_train)
    clf.fit(predictors_train, np.ravel(target_train, order='C'))
    predictions = clf.predict(predictors_test)
    NN_properties(clf) ####Calling NN_properties to see the model attributes

    make_plots_default(clf)  ##Calling make_plots function to see the error plots

    return accuracy_score(target_test, predictions)

    # # train-test split
    # pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=.3,
    #                      random_state=4)

    # Calling get_accuracy function which also invoke other functions NN_properties, make_plots, make_plots_all
    print("Accuracy score: %.2f" % get_accuracy(tar_train,tar_test, pred_test, pred_train))