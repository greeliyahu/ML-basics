import os                                                                                                               # import modules
import numpy as np
from matplotlib import pyplot as plt

def readTxt(path):
    with open(path, 'r') as f:
        content = f.readlines()                                                                                         # open txt file as read-only
    x_data = []                                                                                                         # create lists for X and Y values
    y_data = []
    # read every line and append it to data list after cleaning
    for line in content:
        vals = line.split(',')
        vals[1] = vals[1].split('\n')[0]
        x_data.append(float(vals[0]))
        y_data.append(float(vals[1]))
    x_vals = np.array(x_data)
    y_vals = np.array(y_data)
    return x_vals, y_vals

def plotData(x_vals, y_vals, label_x, label_y, bounds = [0, 10000, 0, 10000], type = 'b-'):
    plt.plot(x_vals, y_vals, type)                                                                                      # data to plot
    plt.xlabel(label_x)                                                                                                 # set labels for axes
    plt.ylabel(label_y)
    plt.axis(bounds)                                                                                                    # set bounds for axes
    plt.show()

def computeCost(x_vals, y_vals, theta_vals):                                                                            # cost function for linear regression
    ones = np.ones_like(x_vals)                                                                                         # create a vector of 1s to add to the X values
    x_vals = np.array([ones, x_vals])                                                                                   # add the 1s vector to the other X values
    x_vals = x_vals.T                                                                                                   # transpose X values
    hypothesis = np.dot(x_vals, theta_vals)                                                                             # calculate hypothesis
    dif = np.subtract(hypothesis, y_vals)                                                                               # calculate difference between hypothesis and Y values
    square = np.square(dif)                                                                                             # square the difference
    sum = np.sum(square)                                                                                                # calculate the sum of squares
    J = sum / (2*(y_vals.shape[0]))                                                                                     # calculate half the average of the sum
    return J, dif

def gradientDescent(x_vals, y_vals, theta_vals, alpha, iter):                                                           # gradient descent for linear regression
    reg = alpha / y_vals.shape[0]                                                                                       # expression to multiply the difference by
    J_vals = []                                                                                                         # initialize list for costs
    for i in range(iter):                                                                                               # repeat until convergence
        j_0, dif_0 = computeCost(x_vals, y_vals, theta_vals)                                                            # calculate cost with current thetas
        multi = np.multiply(dif_0, x_vals)                                                                              # multiply error by X value
        thetas0_temp = theta_vals[0] - (reg * np.sum(dif_0))                                                            # correct thetas as temporary variables
        thetas1_temp = theta_vals[1] - (reg * np.sum(multi))
        thetas_temp = np.array([thetas0_temp, thetas1_temp])                                                            # make array of temporary theta values
        j_1 = computeCost(x_vals, y_vals, thetas_temp)[0]                                                               # calculate cost with temporary thetas
        if j_1 >= j_0:                                                                                                  # if cost does not decrease - break
            i = i - 1                                                                                                   # count the previous iteration as the last
            J_vals.append(j_0)                                                                                          # add cost to costs list
            break
        else:                                                                                                           # if cost decreases
            theta_vals[0] = thetas0_temp                                                                                # update theta values
            theta_vals[1] = thetas1_temp
            J_vals.append(j_0)                                                                                          # add cost to costs list
    return theta_vals, i, J_vals

def predictValues(input, theta_vals):
    prediction = theta_vals[0] + (input * theta_vals[1])                                                                # calculate prediction value
    return prediction

# main code
folder = os.path.dirname(__file__)                                                                                      # determine folder path of script file
path = folder + '/data/PartA_B/ex1data1.txt'                                                                            # create path variable for data
x_vals, y_vals = readTxt(path)                                                                                          # read file

label_x = 'Population of City in 10,000s'                                                                               # set labels for axes
label_y = 'Profit in 10,000$'
type = 'rx'                                                                                                             # set graphic for points
bounds_1 = [4, 24, -5, 25]                                                                                              # set bounds for graph
plotData(x_vals, y_vals, label_x, label_y, bounds_1, type)                                                              # plot the data

theta_vals = np.array([0, 0], dtype = 'f8')                                                                             # initial theta values
J, dif = computeCost(x_vals, y_vals, theta_vals)                                                                        # compute cost function
print('The cost for Theta values of 0 is', J)                                                                           # print J value for thetas of 0

alpha = 0.01                                                                                                            # set learning rate
iter = 1500                                                                                                             # set maximum number of iterations
theta_values, iteration, J_vals = gradientDescent(x_vals, y_vals, theta_vals, alpha, iter)                              # run gradient descent
# print results
print('Theta0 is', theta_values[0], 'and Theta1 is', theta_values[1])
print('Gradient descent converged at iteration', iteration + 1)
print('Minimal cost J is', J_vals[-1])

label_i = 'No. of iterations'                                                                                           # set labels for axes
label_j = 'Cost function J'
I = np.arange(iteration+1)                                                                                              # X values for graph
bounds_2 = [0, 10, 0, np.amax(J_vals)+1]                                                                                # set bounds for graph
plotData(I, J_vals, label_i, label_j, bounds_2)                                                                         # plot the data

predict_1 = predictValues(35000, theta_values)                                                                          # predict case 1
predict_2 = predictValues(70000, theta_values)                                                                          # predict case 2
print('The expected profit for a city with 35K inhabitants is', predict_1)
print('The expected profit for a city with 70K inhabitants is', predict_2)