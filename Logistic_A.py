import os                                                                                                               # import modules
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

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
    return x_vals, y_vals                                                                                               # return arrays of X and Y values

def sigmoidFunc(x_vals, theta_vals):                                                                                    # sigmoid function for logistic regression
    z = np.dot(theta_vals, x_vals)                                                                                      # multiply X matrix by theta vector
    H = []                                                                                                              # initialize a list for hypothesis values
    for x in np.nditer(z):                                                                                              # iterate over z values
        Hx = 1 / (1 + (np.exp(x * -1)))                                                                                 # calculate hypothesis
        H.append(Hx)                                                                                                    # add hypothesis to list
    H = np.array(H)                                                                                                     # turn H into an array
    return H

def costFunction(x_vals, y_vals, theta_vals, alpha = 1):
    # cost fuction
    hypothesis = sigmoidFunc(x_vals, theta_vals)                                                                        # calculate logistic hypothesis
    sum = (y_vals * np.log(hypothesis)) + (1 - y_vals) * np.log(1-hypothesis)
    sum = np.sum(sum)
    J = -1 * sum / y_vals.shape[0]                                                                                      # calculate cost function
    # gradient
    reg = alpha / y_vals.shape[0]
    dif = hypothesis - y_vals
    multi_1 = np.multiply(dif, x_vals[1])
    multi_2 = np.multiply(dif, x_vals[2])
    thetas0 = theta_vals[0] - (reg * np.sum(dif))                                                                       # update theta values
    thetas1 = theta_vals[1] - (reg * np.sum(multi_1))
    thetas2 = theta_vals[2] - (reg * np.sum(multi_2))
    G = np.array([thetas0, thetas1, thetas2])                                                                           # array of new theta values
    return J, G

def optimize(x_vals, y_vals, theta_values, iter):                                                                       # optimization for logistic regression
    x_vals = x_vals / 50                                                                                                # normalization
    theta_values_0 = theta_values                                                                                       # initial theta values
    Thetas = theta_values                                                                                               # array for final theta values
    for i in range(iter):                                                                                               # repeat until convergence
        j_1, theta_values_1 = costFunction(x_vals, y_vals, theta_values_0)                                              # compare the cost with current theta values
        j_2, theta_values_2 = costFunction(x_vals, y_vals, theta_values_1)                                              # to the cost with updated theta values
        if j_2 >= j_1:                                                                                                  # if cost does not decrease - break
            i = i - 1                                                                                                   # count the previous iteration as the last one
            J_min = j_1                                                                                                 # update minimal cost
            Thetas = theta_values_1                                                                                     # update final theta values
            break
        else:                                                                                                           # if cost decreases
            J_min = j_2                                                                                                 # update minimal cost
            theta_values_0 = theta_values_2                                                                             # update initial theta values
            Thetas = theta_values_2                                                                                     # update final theta values
    return J_min, Thetas, i

def predict(input, theta_vals, cutoff):                                                                                 # prediction function for logistic regression
    hypothesis = np.array([sigmoidFunc(input, theta_vals)])                                                             # the hypothesis is the sigmoid function
    predictions = []                                                                                                    # initialize list for prediction values
    for h in np.nditer(hypothesis):                                                                                     # iterate over hypothesis values
        if h >= cutoff:                                                                                                 # if the value is larger or equal to the cutoff value
            pred =  1                                                                                                   # prediction is 0
            predictions.append(pred)                                                                                    # add value to predictions list
        else:
            pred =  0                                                                                                   # prediction is 0
            predictions.append(pred)                                                                                    # add value to predictions list
    return predictions, hypothesis

def plotData(x_vals, y_vals, label_x, label_y, bounds = [25, 110, 25, 110]):
    mask_1 = np.ma.masked_equal(y_vals, 0)                                                                              # create mask for rejected candidates
    mask_2 = np.logical_not(mask_1)                                                                                     # create mask for accepted candidates
    base_array = np.full(y_vals.shape, 50)                                                                              # create arbitrary array the size of Y
    masked_array_1 = np.ma.masked_array(base_array, mask_1)                                                             # create masked array for rejected candidates
    masked_array_2 = np.ma.masked_array(base_array, mask_2)                                                             # create masked array for accepted candidates
    rejected = plt.scatter(x_vals[1], x_vals[2], s = masked_array_1, c='red', marker='+')                               # create scatter plot for rejected candidates
    admitted = plt.scatter(x_vals[1], x_vals[2], s = masked_array_2, c='blue', marker='o')                              # create scatter plot for accepted candidates
    plt.xlabel(label_x)                                                                                                 # determine labels for axes
    plt.ylabel(label_y)
    plt.axis(bounds)                                                                                                    # determine bounds for axes
    plt.legend([admitted, rejected], ['Admitted', 'Rejected'])                                                          # labels for legend
    plt.show()

# main code
folder = os.path.dirname(__file__)                                                                                      # determine folder path of script file
path = folder + '/data/PartC/Part_C_ex2data1.txt'                                                                       # create path variable for data
x_vals, y_vals = readTxt(path)                                                                                          # read file
ones = np.ones_like(y_vals)                                                                                             # create a vector of 1s
x_vals = np.concatenate(([ones], [x_vals[0]], [x_vals[1]]), axis=0)                                                     # add the vector to the X values

label_x = 'Exam 1 score'                                                                                                # set labels for axes
label_y = 'Exam 2 score'
plotData(x_vals, y_vals, label_x, label_y)                                                                              # plot the data

initial_theta_vals = np.full((x_vals.shape[0]), 0, dtype='f8')                                                          # set initial theta values
ex_x = np.array([1, 78, 55])                                                                                            # example values
H = sigmoidFunc(ex_x, initial_theta_vals)                                                                               # calculate sigmoid function value
print('Sigmoid function with input 0 equals', H)

J, G = costFunction(x_vals, y_vals, initial_theta_vals)                                                                 # calculate cost function
print('Cost function with input 0 is', J)
print('Gradient with input 0 is', G)

"""
# THIS CODE DOES NOT WORK #
opt_x = np.transpose(x_vals)
opt_y = np.transpose(y_vals)
res = minimize(costFunction,
                initial_theta_vals,
                (opt_x, opt_y),
                jac = True,
                method = 'TNC')
cost = res.fun
theta = res.x
print('cost is', cost)
print('theta is', theta)
"""

iter = 5000                                                                                                             # set maximum number of iterations for optimization
J_min, Opt_theta, iterations = optimize(x_vals, y_vals, initial_theta_vals, iter)                                       # run gradient descent
print('Minimum cost is', J_min)
print('Theta values are', Opt_theta)
print('Gradient descent converged after', iterations + 1, 'iterations.')

test_data = np.array([1, 45, 85])                                                                                       # test data array
test_theta = np.array([-25.101, 0.206, 0.201])                                                                          # correct theta array
cutoff = 0.5                                                                                                            # the value above which the prediction output is 1
prediction, certainty = predict(test_data, test_theta, cutoff)                                                          # predict example
if prediction[0] == 1:                                                                                                  # print answer
    print('The candidate will be accepted with', '%7.3f' % (certainty[0] * 100), '% certainty.')
else:
    print('The candidate will be rejected  with', '%7.3f' % ((1 - certainty[0])* 100), '% certainty.')

pred, cert = predict(x_vals, test_theta, cutoff)                                                                        # calculate predictions and certainty
correct = 0                                                                                                             # initialize correct answers count
for i in range(y_vals.shape[0]):                                                                                        # iterate over the Y values
    if pred[i] == y_vals[i]:                                                                                            # if the answer is correct
        correct += 1                                                                                                    # increase correct answer counter
accuracy = correct / y_vals.shape[0]                                                                                    # calculate algorithm accuracy
print('Algorithm prediction accuracy is', '%7.3f' % (accuracy * 100),'%')