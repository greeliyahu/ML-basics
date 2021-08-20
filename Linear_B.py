import os                                                                                                               # import modules
import numpy as np
from matplotlib import pyplot as plt

def readTxt(path):
    with open(path, 'r') as f:
        content = f.readlines()                                                                                         # open txt file as read-only
    x1_data = []                                                                                                        # create lists for X and Y values
    x2_data = []
    y_data = []
    # read every line and append it to data list after cleaning
    for line in content:
        vals = line.split(',')
        vals[1] = vals[1].split('\n')[0]
        x1_data.append(float(vals[0]))
        x2_data.append(float(vals[1]))
        y_data.append(float(vals[2]))
    x_vals = np.array([x1_data, x2_data])
    y_vals = np.array(y_data)
    return x_vals, y_vals

def featureNormalize(values):
    S_1 = (np.amax(values[0])) - (np.amin(values[0]))                                                                   # calculate range of vector
    mean_1 = np.mean(values[0])                                                                                         # calculate mean of vector
    S_2 = (np.amax(values[1])) - (np.amin(values[1]))                                                                   # calculate range of vector
    mean_2 = np.mean(values[1])                                                                                         # calculate mean of vector
    normal_1 = (values[0] - mean_1) / S_1                                                                               # normalize values in vectors
    normal_2 = (values[1] - mean_2) / S_2
    normal_values = np.array([normal_1, normal_2])                                                                      # array of normalized values
    normal_params = [S_1, mean_1, S_2, mean_2]                                                                          # list of normalization parameters
    return normal_values, normal_params

def computeCostMulti(x_vals, y_vals, theta_vals):                                                                       # cost function for linear regression
    ones = np.ones_like(y_vals)                                                                                         # create a vector of 1s to add to the X vectors
    x_vals = x_vals.T                                                                                                   # transpose X values
    x_vals = np.concatenate(([ones], [x_vals[:, 0]], [x_vals[:, 1]]), axis = 0)                                         # # add the 1s vector to the other X values
    hypothesis = np.dot(theta_vals, x_vals)                                                                             # calculate hypothesis
    dif = np.subtract(hypothesis, y_vals)                                                                               # calculate difference between hypothesis and Y values
    square = np.square(dif)                                                                                             # square the difference
    sum = np.sum(square)                                                                                                # calculate the sum of squares
    J = sum / (2*(y_vals.shape[0]))                                                                                     # calculate half the average of the sum
    return J, dif

def gradientDescentMulti(x_vals, y_vals, theta_vals, alpha, iter):                                                      # gradient descent for linear regression
    reg = alpha / y_vals.shape[0]                                                                                       # expression to multiply the difference by
    J_vals = []                                                                                                         # initialize list for cost values
    for i in range(iter):                                                                                               # repeat until convergence
        j_0, dif_0 = computeCostMulti(x_vals, y_vals, theta_vals)                                                       # calculate cost with current thetas
        multi1 = np.multiply(dif_0, x_vals[0])                                                                          # multiply errors by X value
        multi2 = np.multiply(dif_0, x_vals[1])
        thetas0_temp = theta_vals[0] - (reg * np.sum(dif_0))                                                            # correct thetas as temporary variables
        thetas1_temp = theta_vals[1] - (reg * np.sum(multi1))
        thetas2_temp = theta_vals[2] - (reg * np.sum(multi2))
        thetas_temp = np.array([thetas0_temp, thetas1_temp, thetas2_temp])                                              # make array of temporary theta values
        j_1 = computeCostMulti(x_vals, y_vals, thetas_temp)[0]                                                          # calculate cost with temporary thetas
        if j_1 >= j_0:                                                                                                  # if cost does not decrease - break
            i = i - 1                                                                                                   # count the previous iteration as the last
            J_vals.append(j_0)                                                                                          # add j_0 to cost list
            break
        else:                                                                                                           # if cost decreases
            theta_vals[0] = thetas0_temp                                                                                # update theta values
            theta_vals[1] = thetas1_temp
            theta_vals[2] = thetas2_temp
            J_vals.append(j_0)                                                                                          # add j_0 to cost list
    return theta_vals, i, J_vals

def predictValues(input, theta_vals):
    prediction = theta_vals[0] + (input[0] * theta_vals[1]) + (input[1] * theta_vals[2])                                # calculate prediction values
    return prediction

def plotData(x_vals, y_vals, label_x, label_y, bounds = [0, 10000, 0, 10000], type = 'b-'):
    plt.plot(x_vals, y_vals, type)                                                                                      # data to plot
    plt.xlabel(label_x)                                                                                                 # set labels for axes
    plt.ylabel(label_y)
    plt.axis(bounds)                                                                                                    # set bounds for axes
    plt.show()

# main code
folder = os.path.dirname(__file__)                                                                                      # determine folder path of script file
path = folder + '/data/PartA_B/ex1data2.txt'                                                                            # create path variable for data
x_vals, y_vals = readTxt(path)                                                                                          # read file

x_vals, n_params = featureNormalize(x_vals)                                                                             # normalize features
theta_vals = np.zeros((x_vals.shape[0]+1), dtype = 'f8')                                                                # set initial theta values

alpha = 0.1                                                                                                             # set learning rate
iter = 400                                                                                                              # set maximum number of iterations
theta_values, iteration, J_vals = gradientDescentMulti(x_vals, y_vals, theta_vals, alpha, iter)                          # run gradient descent
# print results
print('Theta0 is', theta_values[0], ', Theta1 is', theta_values[1], 'and Theta2 is', theta_values[2])
print('Gradient descent converged at iteration', iteration + 1)
print('Minimal cost J is', J_vals[-1])

input = [1650, 3]                                                                                                       # test example
input[0] = (input[0] - n_params[1]) / n_params[0]                                                                       # normalize test example
input[1] = (input[1] - n_params[3]) / n_params[2]
predict = predictValues(input, theta_values)                                                                            # run prediction
print('The predicted price is', predict)

label_i = 'No. of iterations'                                                                                           # set labels for axes
label_j = 'Cost function J'
I = np.arange(iteration + 1)                                                                                            # X values for graph
bounds_2 = [0, np.amax(I)+1, 0, np.amax(J_vals) + 1]                                                                    # set bounds for axes
plotData(I, J_vals, label_i, label_j, bounds_2)                                                                         # plot the data