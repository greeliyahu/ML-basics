import os                                                                                                               # import modules
import random
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

def sigmoid(z):                                                                                                         # computes sigmoid function
    return 1.0 / (1.0 + np.exp(-z))

def predict(Theta1, Theta2, X, Y):
    m = X.shape[0]                                                                                                      # number of examples
    # forward propagation
    Layer_1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))                                   # values of hidden layer
    Layer_2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), Layer_1], axis=1), Theta2.T))                             # values of output layer
    prediction = np.argmax(Layer_2, axis=1)                                                                             # prediction values
    counter = 0                                                                                                         # correct answer counter
    for i in np.ndindex(prediction.shape[0]):                                                                           # iterate over predictions
        if prediction[i] == Y[i]-1:                                                                                     # if the prediction is correct
            counter += 1                                                                                                # add 1 to the correct answer counter
    accuracy = counter/prediction.shape[0]                                                                              # calculate the percentage of correct answers
    return prediction, accuracy

def displayData(array, img_h, img_w):
    fig, axs = plt.subplots(10,10)                                                                                      # create 10x10 image
    for i in range(10):                                                                                                 # iterate over the cells of the large image
        for j in range(10):
            rand_num = random.randint(0, data['X'].shape[0]-1)                                                          # generate a random number between 0 and the number of rows to display a random image
            matrix = array[rand_num].reshape(img_h, img_w)                                                              # reshape a random line from X to 20x20 matrix
            axs[i,j].imshow(matrix.T, cmap = 'gray')                                                                    # show numeral in grayscale
            axs[i,j].get_xaxis().set_visible(False)                                                                     # turn off axes
            axs[i,j].get_yaxis().set_visible(False)
    plt.show(fig)

# main code
folder = os.path.dirname(__file__)                                                                                      # determine folder path of script file
path_data = folder + '/Coursera/Data/ex4data1.mat'                                                                      # create path variable for data
path_theta = folder + '/Coursera/Data/ex4weights.mat'                                                                   # create path variable for weights
data = loadmat(path_data)                                                                                               # load data from .mat file
displayData(data['X'], 20, 20)                                                                                          # display image
thetas = loadmat(path_theta)                                                                                            # load thetas from .mat file
Theta1 = thetas['Theta1']
Theta2 = thetas['Theta2']
pre, acc = predict(Theta1, Theta2, data['X'], data['y'])                                                                # use X and theta values to predict y and check the accuracy of prediction
print('Prediction accuracy is', '%5.2f' % (acc * 100),'%.')