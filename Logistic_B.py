import os                                                                                                               # import necessary modules
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from shapely.geometry import Point
import contextily as ctx
from matplotlib import pyplot as plt

def featureNormalize(values):                                                                                           # feature normalization function
    normal_values = pd.DataFrame([])                                                                                    # initialize dataframe for normalized values
    normal_params = []                                                                                                  # initialize list for normalization parameters
    for i in range(values.shape[1]):                                                                                    # iterate over vectors of values to normalize
        c_range = values.iloc[:, i].max() - values.iloc[:, i].min()                                                     # difference between the smallest and largest values in vector
        c_mean = values.iloc[:, i].mean()                                                                               # mean value in vector
        c_normal = (values.iloc[:, i] - c_mean) / c_range                                                               # normalize the values in vector
        normal_params.append([c_range, c_mean])                                                                         # append the vector's normalization parameters to their list
        normal_values = normal_values.join(c_normal, how = 'right')                                                     # add the normalized vector to the normalized values dataframe
    return normal_values, normal_params

def sigmoidFunc(x_vals, theta_vals):                                                                                    # sigmoid function for logistic regression
    z = np.dot(x_vals, theta_vals)                                                                                      # multiply X matrix by theta vector
    H = []                                                                                                              # initialize a list for hypothesis values
    for x in np.nditer(z):                                                                                              # iterate over z values
        Hx = 1 / (1 + (np.exp(x * -1)))                                                                                 # calculate hypothesis
        H.append(Hx)                                                                                                    # add hypothesis to list
    H = np.array(H)                                                                                                     # turn H into an array
    return H

def computeCostLinear(x_vals, y_vals, theta_vals):                                                                      # cost function for linear regression
    hypothesis = np.dot(x_vals, theta_vals)                                                                             # calculate hypothesis
    dif = np.subtract(hypothesis, y_vals)                                                                               # calculate difference between hypothesis and Y values
    square = np.square(dif)                                                                                             # square the difference
    sum = np.sum(square)                                                                                                # calculate the sum of squares
    J = sum / (2*(y_vals.shape[0]))                                                                                     # calculate half the average of the sum
    return J, dif

def computeCostLogistic(x_vals, y_vals, theta_vals, alpha = 1):
    # cost fuction
    hypothesis = sigmoidFunc(x_vals, theta_vals)                                                                        # calculate logistic hypothesis
    sum = (y_vals * np.log(hypothesis)) + (1 - y_vals) * np.log(1-hypothesis)
    sum = np.sum(sum)
    J = -1 * sum / y_vals.shape[0]                                                                                      # calculate cost function
    # gradient
    reg = alpha / y_vals.shape[0]
    dif = hypothesis - y_vals
    multi = x_vals.mul(dif, axis = 0)
    thetas0 = theta_vals[0] - (reg * np.sum(dif))                                                                       # update theta values
    thetas1 = theta_vals[1] - (reg * np.sum(multi.iloc[1, :]))
    thetas2 = theta_vals[2] - (reg * np.sum(multi.iloc[2, :]))
    G = np.array([thetas0, thetas1, thetas2])                                                                           # array of new theta values
    return J, G

def optimizeLinear(x_vals, y_vals, theta_vals, alpha, iter):                                                            # gradient descent for linear regression
    reg = alpha / y_vals.shape[0]                                                                                       # expression to multiply the difference by
    for i in range(iter):                                                                                               # repeat until convergence
        j_0, dif_0 = computeCostLinear(x_vals, y_vals, theta_vals)                                                      # calculate cost with current thetas
        multi = x_vals.mul(dif_0, axis = 0)                                                                             # multiply errors by X value
        thetas0_temp = theta_vals[0] - (reg * np.sum(dif_0))                                                            # correct thetas as temporary variables
        thetas1_temp = theta_vals[1] - (reg * np.sum(multi.iloc[1, :]))
        thetas2_temp = theta_vals[2] - (reg * np.sum(multi.iloc[2, :]))
        thetas3_temp = theta_vals[3] - (reg * np.sum(multi.iloc[3, :]))
        thetas_temp = np.array([thetas0_temp, thetas1_temp, thetas2_temp, thetas3_temp])                                # make array of temporary theta values
        j_1 = computeCostLinear(x_vals, y_vals, thetas_temp)[0]                                                         # calculate cost with temporary thetas
        if j_1 >= j_0:                                                                                                  # if cost does not decrease - break
            i = i - 1                                                                                                   # count the previous iteration as the last one
            break
        else:                                                                                                           # if cost decreases
            theta_vals[0] = thetas0_temp                                                                                # update theta values
            theta_vals[1] = thetas1_temp
            theta_vals[2] = thetas2_temp
            theta_vals[3] = thetas3_temp
    return theta_vals, i, j_0

def optimizeLogistic(x_vals, y_vals, theta_values, iter):                                                               # optimization for logistic regression
    x_vals = x_vals / 10                                                                                                # normalization
    theta_values_0 = theta_values                                                                                       # initial theta values
    Thetas = theta_values                                                                                               # array for final theta values
    for i in range(iter):                                                                                               # repeat until convergence
        j_1, theta_values_1 = computeCostLogistic(x_vals, y_vals, theta_values_0)                                       # compare the cost with current theta values
        j_2, theta_values_2 = computeCostLogistic(x_vals, y_vals, theta_values_1)                                       # to the cost with updated theta values
        if j_2 >= j_1:                                                                                                  # if cost does not decrease - break
            i = i - 1                                                                                                   # count the previous iteration as the last one
            Thetas = theta_values_1                                                                                     # update final theta values
            J_min = j_1                                                                                                 # update minimal cost
            break
        else:                                                                                                           # if cost decreases
            theta_values_0 = theta_values_2                                                                             # update initial theta values
            Thetas = theta_values_2                                                                                     # update final theta values
            J_min = j_2                                                                                                 # update minimal cost
    return Thetas, i, J_min

def predictLinear(input, theta_vals):                                                                                   # prediction function for linear regression
    predictions = []                                                                                                    # initialize for prediction values
    for i in range(input.shape[0]):                                                                                     # iterate over the input X value vectors
        pred = theta_vals[0]                                                                                            # theta0 is the base
        for j in range(input.shape[1]):                                                                                 # iterate over the X values in the vector
            pred = pred + (input.iloc[i, j] * theta_vals[(j+1)])                                                        # calculate prediction
        predictions.append(pred)                                                                                        # add prediction to predictions list
    return predictions

def predictLogistic(input, theta_vals, cutoff):                                                                         # prediction function for logistic regression
    hypothesis = sigmoidFunc(input, theta_vals)                                                                         # the hypothesis is the sigmoid function
    predictions = []                                                                                                    # initialize list for prediction values
    for h in np.nditer(hypothesis):                                                                                     # iterate over hypothesis values
        if h > cutoff:                                                                                                  # if the value is larger than the cutoff value
            pred =  1                                                                                                   # prediction is 1
            predictions.append(pred)                                                                                    # add value to predictions list
        else:
            pred =  0                                                                                                   # prediction is 0
            predictions.append(pred)                                                                                    # add value to predictions list
    return predictions, hypothesis

def plotExpChp(input):                                                                                                  # map with price categories
    # create the map
    ax = input.to_crs(epsg=3857).plot(figsize=(10, 10),
                                        column ='price',
                                        cmap='seismic',
                                        legend=True,
                                        legend_kwds={'title': 'Apartment price', 'loc': 3},
                                        markersize=2)
    ctx.add_basemap(ax, zoom=12, url=ctx.providers.Stamen.TonerLite)                                                    # add base map of Amsterdam
    plt.show()                                                                                                          # show the map

def plotBubbles(input):                                                                                                 # bubble chart
    # create the map
    ax = input.to_crs(epsg=3857).plot(figsize=(10, 10),
                                      column='price/night',
                                      cmap = 'Greens',
                                      scheme = 'Percentiles',
                                      legend=True,
                                      legend_kwds={'title': 'Apartment price', 'loc': 3},
                                      markersize = input.iloc[:,-1])
    ctx.add_basemap(ax, zoom=12, url=ctx.providers.Stamen.TonerLite)                                                    # add base map of Amsterdam
    plt.show()                                                                                                          # show the map

# main code
folder = os.path.dirname(__file__)                                                                                      # determine folder path of script file
path = folder + '/data/PartD/BigML_Dataset_5dd97775fb7bdd184c00024f.csv'                                                # create path variable for data
# read only the relevant data from file
data = pd.read_csv(path, usecols = ['accommodates',
                                    'cleaning_fee',
                                    'review_scores_rating',
                                    'latitude',
                                    'longitude',
                                    'room_type',
                                    'price'])

### LINEAR REGRESSION ###
y_data_lin = data['price']                                                                                              # determine the Y column for linear regression
x_data_lin = data.loc[:, lambda df: ['accommodates', 'cleaning_fee', 'review_scores_rating']]                           # choose three parameters to serve as X vectors
x_data_lin.fillna(0, inplace = True)                                                                                    # replace missing X values with 0
# split the data into training (80%) and testing (20%) sets
x_train_lin, x_test_lin, y_train_lin, y_test_lin = train_test_split(x_data_lin , y_data_lin, test_size = 0.2)
norm_values_lin, norm_params_lin = featureNormalize(x_train_lin)                                                        # normalize training X values and determine normalization parameters
ones = pd.DataFrame(np.ones_like(y_train_lin), columns = ['ones'])                                                      # create a vector of 1s to add to the X vectors
x_values_lin = ones.join(norm_values_lin, how = 'right')                                                                # add the 1s vector to the other X vectors
x_values_lin.fillna(1.0, inplace = True)                                                                                # fill the missing values in the 1s vector that were created because of index misalignment with 1s
initial_theta_lin = np.full((x_values_lin.shape[1]), 0, dtype='f8')                                                     # initialize theta values
alpha = 0.002                                                                                                           # set alpha value for gradient descent
iter = 7000                                                                                                             # set maximum number of iterations for gradient descent
theta_values_lin, iter_lin, J_lin = optimizeLinear(x_values_lin, y_train_lin, initial_theta_lin, alpha, iter)           # run gradient descent
# print theta values, number of iterations until convergence and minimal cost
print('Theta0 is', theta_values_lin[0], ', Theta1 is', theta_values_lin[1], ', Theta2 is', theta_values_lin[2], 'and Theta 3 is', theta_values_lin[3])
print('Gradient descent converged at iteration', iter_lin + 1)
print('Minimal cost J is', J_lin)

for i in range(x_test_lin.shape[0]):                                                                                    # iterate over the test X values
    x_test_lin.iloc[i, 0] = (x_test_lin.iloc[i, 0] - norm_params_lin[0][1]) / norm_params_lin[0][0]                     # normalize the test X values
    x_test_lin.iloc[i, 1] = (x_test_lin.iloc[i, 1] - norm_params_lin[1][1]) / norm_params_lin[1][0]
    x_test_lin.iloc[i, 2] = (x_test_lin.iloc[i, 2] - norm_params_lin[2][1]) / norm_params_lin[2][0]
predictions_lin = predictLinear(x_test_lin, theta_values_lin)                                                           # use the test set to predict the values of Y
errors = y_test_lin - predictions_lin                                                                                   # count the size of the errors against the Y test values
avg_error = errors.sum() / y_test_lin.shape[0]                                                                          # calculate average error
print('Average error is', avg_error)                                                                                    # print average error value

### LOGISTIC REGRESSION ###
loc_data = data.loc[:, lambda df: ['latitude', 'longitude']]                                                            # extract coordinates of bnbs
pts = []                                                                                                                # initialize a list of points for bnb locations
for row in loc_data.itertuples():                                                                                       # iterate over location data
    pt = Point(row.longitude, row.latitude)                                                                             # create point
    pts.append(pt)                                                                                                      # add the point to the points list
address_pts = gpd.GeoSeries(data = pts, crs = {'init': 'epsg:4326'})                                                    # turn point list into geodataframe and set crs to the netherlands
address_pts_fixed = address_pts.to_crs('+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +units=m +no_defs ')
center = gpd.tools.geocode('Amsterdam', timeout = None)                                                                 # geocode the center of amsterdam and set crs to the netherlands
center_fixed = center.to_crs('+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +units=m +no_defs ')
distances = []                                                                                                          # initialize a list of distances between the center and the bnbs
for i in range(address_pts_fixed.shape[0]):                                                                             # iterate over the bnb locations
    dist = address_pts_fixed.iloc[i].distance(center_fixed.iloc[0, 0])                                                  # measure distance to center
    distances.append(dist)                                                                                              # add the distance to the distance list
room_types = data['room_type']                                                                                          # extract room type column from data
room_types_int = room_types.replace(to_replace = {'Shared room' : 0, 'Private room' : 1, 'Entire home/apt': 2})         # encode room types as integers
# create X and Y dataframes
x_data_log = pd.DataFrame({'room_type': room_types_int, 'dist_to_downtown': distances})                                 # create dataframe for X values
y_data_log = (y_data_lin > 100).astype(int)                                                                             # sort the apartments into those above 100 euro/night (1) and below 100 euro/night (0)
# split the data into training (80%) and testing (20%) sets
x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(x_data_log, y_data_log, test_size = 0.2)
norm_values_log, norm_params_log = featureNormalize(x_train_log)                                                        # normalize training X values and determin
x_values_log = ones.join(norm_values_log, how = 'right')                                                                # add that vector to the X training values
x_values_log.fillna(1.0, inplace = True)                                                                                # fill the missing values in the 1s vector that were created because of index misalignment with 1s
initial_theta_log = np.full((x_values_log.shape[1]), 0, dtype='f8')                                                     # initialize theta values
iterations = 10000                                                                                                       # set number of iterations for optimization
theta_values_log, iter_log, J_log = optimizeLogistic(x_values_log, y_train_log, initial_theta_log, iterations)          # run optimization
# print minimal cost, theta values and number of iterations until convergence
print('Minimum cost is', J_log)
print('Theta values are', theta_values_log)
print('Gradient descent converged after', iter_log + 1, 'iterations.')

test_log_1 = np.array([1, 0, 3000])                                                                                     # test case 1
test_log_2 = np.array([1, 2, 3000])                                                                                     # test case 2
test_log_1[1] = (test_log_1[1] - norm_params_log[0][1]) / norm_params_log[0][0]                                         # normalize the X values
test_log_1[2] = (test_log_1[2] - norm_params_log[1][1]) / norm_params_log[1][0]
test_log_2[1] = (test_log_2[1] - norm_params_log[0][1]) / norm_params_log[0][0]
test_log_2[2] = (test_log_2[2] - norm_params_log[1][1]) / norm_params_log[1][0]
cutoff = 0.9                                                                                                            # the value above which the prediction output is 1
prediction_1, chance_1 = predictLogistic(test_log_1, theta_values_log, cutoff)                                          # predict the price category of a shared room at 3000 meters from the center
prediction_2, chance_2 = predictLogistic(test_log_2, theta_values_log, cutoff)                                          # predict the price category of an entire apartment at 3000 meters from the center
if prediction_1[0] == 0:                                                                                                # print answers
    print('The chance of getting a cheap room is', '%7.3f' % (chance_1[0] * 100), '%.')
else:
    print('The chance of getting a cheap room is', '%7.3f' % ((1 - chance_1[0])* 100), '%.')
if prediction_2[0] == 1:
    print('The chance of getting an expensive room is', '%7.3f' % (chance_2[0] * 100), '%.')
else:
    print('The chance of getting an expensive room is', '%7.3f' % ((1 - chance_2[0])* 100), '%.')

predictions_log_1, cert_1 = predictLogistic(x_values_log, theta_values_log, cutoff)                                     # calculate predictions
correct_1 = 0                                                                                                           # initialize correct answers count
for i in range(y_train_log.shape[0]):                                                                                   # iterate over the Y test values
    if predictions_log_1[i] == y_train_log.iloc[i]:                                                                     # if the answer is correct
        correct_1 += 1                                                                                                  # increase correct answer counter
accuracy_1 = correct_1 / y_train_log.shape[0]                                                                           # calculate algorithm accuracy
print('Algorithm prediction accuracy for training set is', '%7.3f' % (accuracy_1 * 100),'%')

for i in range(x_test_log.shape[0]):                                                                                    # iterate over the test X values
    x_test_log.iloc[i, 0] = (x_test_log.iloc[i, 0] - norm_params_log[0][1]) / norm_params_log[0][0]                     # normalize the test X values
    x_test_log.iloc[i, 1] = (x_test_log.iloc[i, 1] - norm_params_log[1][1]) / norm_params_log[1][0]
x_testing_log = ones.join(x_test_log, how = 'right')                                                                    # add that vector to the X testing values
x_testing_log.fillna(1.0, inplace = True)                                                                               # fill the missing values in the 1s vector that were created because of index misalignment with 1s
predictions_log_2, cert_2 = predictLogistic(x_testing_log, theta_values_log, cutoff)                                    # calculate predictions and certainty
correct_2 = 0                                                                                                           # initialize correct answers count
for i in range(y_test_log.shape[0]):                                                                                    # iterate over the Y test values
    if predictions_log_2[i] == y_test_log.iloc[i]:                                                                      # if the answer is correct
        correct_2 += 1                                                                                                  # increase correct answer counter
accuracy_2 = correct_2 / y_test_log.shape[0]                                                                            # calculate algorithm accuracy
print('Algorithm prediction accuracy for testing set is', '%7.3f' % (accuracy_2 * 100),'%')

apartments = gpd.GeoDataFrame(y_data_log, geometry = address_pts)                                                       # create a geodataframe for plots
apartments.replace(to_replace = {1: 'Expensive', 0: 'Cheap'}, inplace = True)                                           # change numbers to string labels
apartments['price/night'] = y_data_lin                                                                                  # add a column of actual prices
plotExpChp(apartments)                                                                                                  # plot price map
plotBubbles(apartments)                                                                                                 # plot bubble map
