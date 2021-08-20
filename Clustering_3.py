from tkinter import *                                                                                                   # import all the required modules
from tkinter import messagebox
import os
import time
from pyclustering.utils import read_sample
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster import cluster_visualizer

def mainFunc():                                                                                                         # the main processs
    pathname = ent1.get()                                                                                               # read and save the path to input data
    if os.path.isfile(pathname):                                                                                        # check if the input file is present at the location
        # Read parameters and define clustering process
        if funcType.get():                                                                                              # check if clustering type was selected
            data = read_sample(pathname)                                                                                # read data from file and prepare it for clustering analysis
            error = 'At least one parameter is missing.'
            if funcType.get() == 1:                                                                                     # if kmeans was selected
                if ent2.get() and ent3.get() and ent4.get():                                                            # check if all parameters had been inserted
                    meanK = int(ent2.get())                                                                             # read user input for K as integer
                    delta = float(ent3.get())                                                                           # read user input for delta as float
                    iter = int(ent4.get())                                                                              # read user input for iterations as integer
                    centers = random_center_initializer(data, meanK).initialize()                                       # create k number of random initial centere
                    start = time.time()                                                                                 # start counting run time
                    process = kmeans(data, centers, delta, itermax = iter)                                              # define analysis process with user input parameters and initial centers
                else:
                    box3 = messagebox.showinfo('Error', error)                                                          # error message to show if launched with missing parameters
                    return
            elif funcType.get() == 2:                                                                                   # if kmedoids was selected
                if ent5.get() and ent6.get() and ent7.get():                                                            # check if all parameters had been inserted
                    medK = int(ent5.get())                                                                              # read user input for K as integer
                    delta = float(ent6.get())                                                                           # read user input for delta as float
                    iter = int(ent7.get())                                                                              # read user input for iterations as integer
                    centers = random_center_initializer(data, medK).initialize(return_index = True)                     # choose k number of random initial centers
                    start = time.time()                                                                                 # start counting run time
                    process = kmedoids(data, centers, delta, itermax = iter)                                            # define analysis process with user input parameters and initial centers
                else:
                    box3 = messagebox.showinfo('Error', error)                                                          # error message to show if launched with missing parameters
                    return
            elif funcType.get() == 3:                                                                                   # if dbscan was selected
                if ent8.get() and ent9.get():                                                                           # check if all parameters had been inserted
                    rad = float(ent8.get())                                                                             # read user input for search radius as float
                    minPts = int(ent9.get())                                                                            # read user input for minimum number of points per cluster as integer
                    start = time.time()                                                                                 # start counting run time
                    process = dbscan(data, rad, minPts)                                                                 # define analysis process with user input parameters
                else:
                    box3 = messagebox.showinfo('Error', error)                                                          # error message to show if launched with missing parameters
                    return
            # Run analysis
            process.process()                                                                                           # run clustering analysis
            clusters = process.get_clusters()                                                                           # save cluster classification
            # Display clusters and run time
            visualizer = cluster_visualizer()                                                                           # initiate visualizer
            visualizer.append_clusters(clusters, data)                                                                  # add the points and cluster classification to visualizer
            end = time.time()                                                                                           # stop measuring run time
            runtime1 = (end - start)                                                                                    # calculate the run time
            runtime2 = '%7.3f' % (runtime1)                                                                             # format run time to display as 'xxx.xxx'
            message = str('Runtime is' + runtime2 + ' seconds.\nPlease close box to view clusters.')                    # write message about run time to show in box
            box2 = messagebox.showinfo('Runtime', message)                                                              # display box with run time
            visualizer.show()                                                                                           # show clusters
        else:
            box1 = messagebox.showinfo('Error', 'Please choose clustering type.')                                       # error message to show if clustering type was not selected
    else:
        box = messagebox.showinfo('Error', 'Please set data source.')                                                   # error message in case data file is not fount at the specified location

# GUI
root = Tk()                                                                                                             # main window for GUI
root.title('Cluster Analyst HW2')                                                                                       # set title of window
funcType = IntVar()                                                                                                     # set variable type for radiobuttons that select the clustering algorithm
# Path to data
lab1 = Label(root, text = "Data source:")                                                                               # create discription label
lab1.pack(anchor = NW)                                                                                                  # place label
ent1 = Entry(root, bd = 2, exportselection = 0, width = 70)                                                             # create input entry for data
ent1.pack(anchor = NE)                                                                                                  # place entry
# K-means input
radB1 = Radiobutton(root, text = "K-Means", variable = funcType, value = 1)                                             # create kmeans radiobutton
radB1.pack(anchor = SW)                                                                                                 # place radiobutton
lab2 = Label(root, text = "K:")                                                                                         # create label for number of centers (K)
lab2.pack(anchor = CENTER)                                                                                              # place label
ent2 = Entry(root, bd = 2, exportselection = 0, width = 7)                                                              # create input entry for K
ent2.pack()                                                                                                             # place entry
lab3 = Label(root, text = "Delta:")                                                                                     # create label for minimal difference between iterations (Delta)
lab3.pack()                                                                                                             # place label
ent3 = Entry(root, bd = 2, exportselection = 0, width = 7)                                                              # create input entry for Delta
ent3.pack()                                                                                                             # place entry
lab4 = Label(root, text = "Iterations:")                                                                                # create label for maximum no. of iterations (Iterations)
lab4.pack()                                                                                                             # place label
ent4 = Entry(root, bd = 2, exportselection = 0, width = 7)                                                              # create input entry for Iterations
ent4.pack()                                                                                                             # place entry
# K-medoids input
radB2 = Radiobutton(root, text = "K-Medoids", variable = funcType, value = 2)                                           # create kmedoids radiobutton
radB2.pack(anchor = SW)                                                                                                 # place radiobutton
lab5 = Label(root, text = "K:")                                                                                         # create label for number of centers (K)
lab5.pack(anchor = CENTER)                                                                                              # place label
ent5 = Entry(root, bd = 2, exportselection = 0, width = 7)                                                              # create input entry for K
ent5.pack(anchor = CENTER)                                                                                              # place entry
lab6 = Label(root, text = 'Delta:')                                                                                     # create label for minimal difference between iterations (Delta)
lab6.pack()                                                                                                             # place label
ent6 = Entry(root, bd = 2, exportselection = 0, width = 7)                                                              # create input entry for Delta
ent6.pack()                                                                                                             # place entry
lab7 = Label(root, text = 'Iterations:')                                                                                # create label for maximum no. of iterations (Iterations)
lab7.pack()                                                                                                             # place label
ent7 = Entry(root, bd = 2, exportselection = 0, width = 7)                                                              # create input entry for Iterations
ent7.pack()                                                                                                             # place entry
# DBscan input
radB3 = Radiobutton(root, text = "DBSCAN", variable = funcType, value = 3)                                              # create dbscan radiobutton
radB3.pack(anchor = SW)                                                                                                 # place radiobutton
lab8 = Label(root, text = "Search radius:")                                                                             # create label for Search radius
lab8.pack(anchor = CENTER)                                                                                              # place label
ent8 = Entry(root, bd = 2, exportselection = 0, width = 5)                                                              # create input entry for search radius
ent8.pack(anchor = CENTER)                                                                                              # place entry
lab9 = Label(root, text = "Min # of points/cluster:")                                                                   # create label for minimum no. of points per cluster
lab9.pack(anchor = CENTER)                                                                                              # place label
ent9 = Entry(root, bd = 2, exportselection = 0, width = 5)                                                              # create input entry for minimum no. of points per cluster
ent9.pack(anchor = CENTER)                                                                                              # place entry
# Activation button
button = Button(root, text = "Launch!", command = mainFunc)                                                             # create launch button for the analysis
button.pack(anchor = S)                                                                                                 # place button in window

root.mainloop()                                                                                                         # end of GUI

