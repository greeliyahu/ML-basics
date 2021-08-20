import os                                                                                                               # import modules
import pandas as pd
import geopandas as gpd
from sklearn import cluster as skl
import contextily as ctx
from matplotlib import pyplot as plt
"""""
from shapely.geometry import point
from shapely.geometry import polygon
from scipy.spatial import ConvexHull
"""
"""
def clusterGrader(pois, clusters, cl_alg):    
    points = pois.tolist()
    
    counts = clusters[cl_alg].value_counts()
    c_sorted = clusters.sort_values(by = cl_alg)
    c_hulls = []
    for index, value in counts.items():
        start = c_sorted['kmeans'].ne(index).idxmax()
        ch_pts = []
        for j in range(start, (start+value)):
            pt = (c_sorted.iloc[j,0], c_sorted.iloc[j,1])
            ch_pts.append(pt)
        hull = ConvexHull(ch_pts, qhull_options='QJ')
        c_hulls.append(hull)
        
    polygons = []
    for h in c_hulls:
        perimiter_pts = []
        for p in range(0, (h.vertices.shape[0]-1),2):
            p_pt = (h.vertices[p], h.vertices[p+1])
            perimiter_pts.append(p_pt)
        poly = polygon.Polygon(perimiter_pts)
        polygons.append(poly)

    correct = 0
    for k in polygons:
        for l in points:
            if k.contains(l):
                correct +=1
                
    return correct/pois.shape[0] 
"""
def plotData(input_1, method, zoom, input_2 = False):                                                                         # map with clusters and possibly POIs
    # create the map
    ax = input_1.to_crs(epsg=3857).plot(figsize=(20, 20),
                                        column = method,
                                        markersize=2)
    ctx.add_basemap(ax, zoom=zoom, url = ctx.providers.Stamen.TonerLite)                                                # add base map of NY
    try:                                                                                                                # if input_2 exists
        # add the data of input_2 on top of input_1
        input_2.to_crs(epsg=3857).plot(ax=ax,
                                       marker='x',
                                       color='red',
                                       markersize=10)
        plt.show()                                                                                                      # show the map
    except:                                                                                                             # if input_2 is missing
        plt.show()                                                                                                      # show the map

def gradePopularity(clusters, method, places):                                                                          # rate popularity
    c_sorted = clusters.sort_values(by=method)                                                                          # sort the by labels of the clustering algorithm
    counts = clusters[method].value_counts()                                                                            # order labels by frequency
    p_range = []                                                                                                        # if the method is kmeans take the first 'places' answers
    if method == 'kmeans':
        for i in range(places):
            p_range.append(i)
    else:                                                                                                               # else, omit the first answer as it is unclassified points
        for i in range(places):
            p_range.append(i+1)
    first = counts.take(p_range)                                                                                        # take the number of 'places' clusters with most points
    indices = first.index.tolist()                                                                                      # make a list of their indices
    pop_pts = []                                                                                                        # initialize popular points list
    for j in range(places):                                                                                             # iterate over the labels
        temp_df = c_sorted[c_sorted[method] == indices[j]]                                                              # create with the labels that match the indices
        pop_pts.append(temp_df.iloc[:,-1])                                                                              # and take only the geometry to the popular points list
    popular = gpd.GeoDataFrame(pop_pts[0], crs = {'init': 'epsg:4326'})                                                 # make a dataframe of the first point group in the list
    popular['group'] = 0                                                                                                # add a column to serve as a  marker
    for k in range(1, places):                                                                                          # add the rest of the points to the dataframe
        pop_pts[k] = gpd.GeoDataFrame(pop_pts[k])                                                                       # and mark each in turn
        pop_pts[k]['group'] = k
        popular = popular.append(pop_pts[k])
    return popular                                                                                                      # return dataframe of popular points with markers

### main code ###

# determine folder path of script file
folder = os.path.dirname(__file__)
# create path variables for data
path_flickr = folder + '/Data/flickr_output.txt'
path_park = folder + '/Data/NYC_POIs/central_park.shp'
path_poi = folder + '/Data/NYC_POIs/NYC_POIs.shp'
# columns to use for picture dataframe
cols = ['lon', 'lat', 'url', 'day', 'month', 'year', 'date', 'user', 'likes', 'acc_level', 'FID']
# read pictures from file into geodataframe
flickr = pd.read_csv(path_flickr,
                     sep = ',',
                     header=0,
                     names = cols)
pics = gpd.GeoDataFrame(flickr,
                        crs = {'init': 'epsg:4326'},
                        geometry = gpd.points_from_xy(flickr.iloc[:,0], flickr.iloc[:,1]))
# read park geometry from file
park = gpd.read_file(path_park)
# filter pictures outside park
filterred = gpd.sjoin(pics, park)
# filter irrelevant fields
filterred.drop(columns=['acc_level','index_right','osm_id','code','fclass','name'], inplace = True)
# filter future dates and dates before flickr was founded
filterred = filterred[filterred.year > 2003]
filterred = filterred[filterred.year < 2020]
# filter multiple pictures from the same user in the same place
filt_1 = filterred.drop_duplicates(subset='geometry', keep='first')
filt_2 = filterred.drop_duplicates(subset='user', keep='first')
filt_join = pd.merge(filt_1,filt_2, how='outer', sort=False)
# points for clustering algorithms
clusters = filt_join.iloc[:, 0:2]
# K-MEANS clustering
kmeans = skl.KMeans(n_clusters=169, max_iter=1000, copy_x=True).fit(clusters)
clusters['kmeans'] = kmeans.labels_
# DBSCAN clustering
dbscan = skl.DBSCAN(eps=0.00008, min_samples=5, algorithm='ball_tree').fit(clusters)
clusters['dbscan'] = dbscan.labels_
# OPTICS clustering
optics = skl.OPTICS(min_samples = 5, max_eps=0.0001, metric='euclidean', p=2, algorithm='ball_tree').fit(clusters)
clusters['optics'] = optics.labels_
# add all clustering results to a geodataframe
clusters = gpd.GeoDataFrame(clusters,
                            crs = {'init': 'epsg:4326'},
                            geometry = gpd.points_from_xy(clusters.iloc[:,0], clusters.iloc[:,1]))
#print maps of clusters
plotData(clusters,'kmeans', 15)
plotData(clusters,'dbscan', 15)
plotData(clusters,'optics', 15)
# load POIs from file
pois = gpd.read_file(path_poi)
# filter only the points in the park
pois_og = gpd.sjoin(pois, park)
# extract geometry only for display on map
pois_og = gpd.GeoDataFrame(pois_og.iloc[:,10],
                           crs = {'init': 'epsg:4326'})
# print maps of clusters with POIs
plotData(clusters,'kmeans', 15, pois_og)
plotData(clusters,'dbscan', 15, pois_og)
plotData(clusters,'optics', 15, pois_og)

"""
### This was an attempt at an accurate measurement of clustering ###
kmeans_accuracy = clusterGrader(pois_og, clusters, 'kmeans')
print(kmeans_accuracy)
dbscan_accuracy = clusterGrader(pois_og, clusters, 'dbscan')
print(dbscan_accuracy)
optics_accuracy = clusterGrader(pois_og, clusters, 'optics')
print(optics_accuracy)
"""

# find the 5 most popular clusters
popular_pois = gradePopularity(clusters, 'optics', 5)
# print them on a map
plotData(popular_pois,'group', 20)


