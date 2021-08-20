# import all necessary modules
import time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
# read data
folder = os.path.dirname(__file__)                                                                                      # determine folder path of script file
path = folder + '/hw1_data.csv'                                                                                         # create path variable for data
data = pd.read_csv(path)                                                                                                # read data file and save it as a pandas data frame
# calculate mean values to replace missing values
wind_speed = data.iloc[:, 12]                                                                                           # isolate wind speed column
wind_dir = data.iloc[:, 13]                                                                                             # isolate wind direction column
ws_mean = wind_speed.mean(skipna = True)                                                                                # calculate mean value of wind speed
wd_mean = wind_dir.mean(skipna = True)                                                                                  # calculate mean value of wind direction
# remove station names and numbers as they are unnecessary for predictions
data = data.drop(['station_name', 'station_no'], axis = 1)
# split the data into features and labels
features = data.iloc[:, 0:8]
labels = data.iloc[:, 8:12]
# replace missing values with the mean of the respective column
labels.fillna(value={'wind_speed': ws_mean, 'wind_dir': wd_mean}, inplace = True)
# split the data into training (80%) and testing (20%) sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2)
# scale feature values
scaler = StandardScaler()                                                                                               # create a StandardScaler instance
scaler.fit(features_train)                                                                                              # fit the scaler to the training features
features_train = scaler.transform(features_train)                                                                       # scale features to improve learning rate
features_test = scaler.transform(features_test)                                                                         # scale features to improve learning rate
# define neural network instance
ann = MLPRegressor(hidden_layer_sizes = (100, 100),
                   activation = 'logistic',
                   solver = 'sgd',
                   alpha = 1,
                   batch_size = 5,
                   learning_rate = 'constant',
                   learning_rate_init = 0.0001,
                   max_iter=1000)
# neural network training
start = time.time()                                                                                                     # start counting run time
ann.fit(features_train, labels_train)                                                                                   # train the neural network
end = time.time()                                                                                                       # stop measuring run time
print ('Training time is ', '%7.3f' % (end - start), 'seconds')                                                         # format run time to display as 'xxx.xxx'
# neural network testing
start1 = time.time()                                                                                                    # start counting run time
print('Score 1 is', '%7.3f' % ann.score(features_test, labels_test))                                                    # print the score of the prediction for dataset 1
end1 = time.time()                                                                                                      # stop measuring run time
print('Runtime 1 is ', '%7.3f' % (end1 - start1), 'seconds')                                                            # format run time to display as 'xxx.xxx'
# neural network testing second data batch
# read data
path2 = folder + '/hw1_data_2.csv'                                                                                      # create path variable for second data batch
data2 = pd.read_csv(path2)                                                                                              # read data file and save it as a pandas data frame
data2 = data2.drop(['station_name', 'station_no'],axis = 1)                                                             # remove station names and numbers as they are unnecessary for predictions
# split the data into features amd labels
features2 = data2.iloc[:, 0:8]
labels2 = data2.iloc[:, 8:12]
features2 = scaler.transform(features2)                                                                                 # scale features to improve learning rate
# replace missing values with the mean of the respective column
labels2.fillna(value={'wind_speed': ws_mean, 'wind_dir': wd_mean}, inplace=True)
start2 = time.time()                                                                                                    # start counting run time
print('Score 2 is ', '%7.3f' % ann.score(features2, labels2))                                                           # print the score of the prediction for dataset 2
end2 = time.time()                                                                                                      # stop measuring run time
print('Runtime 2 is ', '%7.3f' % (end2 - start2), 'seconds')                                                            # format run time to display as 'xxx.xxx'
