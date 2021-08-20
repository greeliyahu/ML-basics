import pandas as pd                                                                                                     # import all necessary modules
import os
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

folder = os.path.dirname(__file__)					                                                                    # determine folder path of script file
path = folder + '/hw3_data_nums.csv'                                                                                    # create path variable for data
data = pd.read_csv(path)                                                                                                # read data file and save it as a pandas data frame

train_features = data.iloc[:257,:-1]                                                                                    # split the data into training (80%) and testing (20%) sets
test_features = data.iloc[257:,:-1]
train_classes = data.iloc[:257,-1]
test_classes = data.iloc[257:,-1]

d_tree = DecisionTreeClassifier(criterion = 'entropy')                                                                  # define the decision tree instance and classification criterion (entropy)
d_tree = d_tree.fit(train_features,train_classes)                                                                       # train decision tree with training data

features = ['crashday', 'ped_sex', 'weather', 'center_dist', 'road_type', 'no_of_poi']                                  # define lists of names for features and classification categories for the chart
classes = ['Unknown Injury', 'O: No Injury', 'C: Possible Injury', 'B: Evident Injury', 'A: Disabling Injury', 'K: Killed']
dot_data = tree.export_graphviz(d_tree, out_file = None, feature_names = features, class_names = classes,  filled = True)   # define chart parameters
graph = graphviz.Source(dot_data)                                                                                       # define chart instance
graph.render('accidents')                                                                                               # create pdf file of the chart named 'accidents' in the script folder

prediction = d_tree.predict(test_features)                                                                              # use the decision tree to classify the test data

print('The prediction accuracy is: ',d_tree.score(test_features,test_classes) * 100, "%")                               # calculate and print the prediction success rate