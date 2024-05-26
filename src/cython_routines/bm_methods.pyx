import numpy
import time
import operator
cimport numpy
from libc.math cimport exp, M_PI

##### Continuous PDF
def continuous_pdf(double x, numpy.ndarray[numpy.float64_t, ndim= 1] mu, double s):
    cdef double mult, total
    cdef int arr_shape = mu.shape[0]
    cdef int i
    mult = (1 / (((2 * M_PI)**0.5) * arr_shape * s))
    total = 0.
    for i in range(arr_shape):
        total += exp(-((x - mu[i])**2.) / (2. * s ** 2.))
    return total*mult

#####First Derivative of PDF
def first_deriv_pdf(double x, numpy.ndarray[numpy.float64_t, ndim= 1] mu, double s):
    """
    evaluate the first derivative of a Gaussian KDE at x
    mu should be the array of ensemble members and
    s the bandwidth
    """
    cdef double mult, total
    cdef int arr_shape = mu.shape[0]
    cdef int i
    mult = -(1 / (((2 * M_PI)**0.5) * arr_shape * s**3.))
    total = 0
    for i in range(arr_shape):
        total += exp(-((x - mu[i]) ** 2.) / (2 * s ** 2.)) * (x - mu[i])
    return total*mult

#####Second Derivative of PDF
def second_deriv_pdf(double x, numpy.ndarray[numpy.float64_t, ndim= 1] mu, double s):
    cdef double mult
    mult = -(1. / (((2 * numpy.pi)**0.5) * len(mu) * s**3.))
    cdef double total
    total = 0
    cdef int arr_shape = mu.shape[0]
    cdef int i
    for i in range(arr_shape):
        total += exp(-((x - mu[i]) ** 2.) / (2 * s ** 2.))*(1-((x-mu[i])**2)/s)
    return total*mult

def is_root(double fa, double fb):
    """
    simle root determinate helper
    check if f(a), f(b) are opposite signs
    """
    return fa*fb<0

def recursive_rootfinding(a, b, tol, fargs, f= first_deriv_pdf):
    """
    f is function to be evaluated, (a, b) are bounds, tol is search interval
    returns roots of function
    """
    roots = [] #save roots
    mp=(b+a)/2. #midpoint
        
    if (b-a>tol):# not within tol, refine search
        roots+=recursive_rootfinding(a, mp,tol, fargs) #check left side
        roots+=recursive_rootfinding(mp, b,tol, fargs) #check right side 
        
    else: #if within tolerance determine if root or not
        if is_root( f(a,*fargs) , f(b,*fargs) ): #use mp as root
            return [mp]
        else: #not a root
            return []
        
    return roots


def root_finding(ensemble,bw, tol=0.1):
    roots = recursive_rootfinding(numpy.min(ensemble), numpy.max(ensemble), tol=tol, fargs=(ensemble,bw))
    return roots
######

#####DISTANCE ALGORITHM FOR CLUSTERING
def binary_dis_C(numpy.ndarray[numpy.int_t, ndim= 1] p1, numpy.ndarray[numpy.int_t, ndim= 1] p2):
    cdef int arr_shape = p1.shape[0]
    cdef int i
    cdef float dis = 0
    for i in range(arr_shape):
        dis += abs(p1[i] - p2[i])
    return (dis) ** 0.5

def C_pairwise(numpy.ndarray[numpy.int_t, ndim = 2] matrix):
    cdef int arr_shape = matrix.shape[0]
    cdef int i, j
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] dist_matrix = numpy.empty((arr_shape,arr_shape))
    for i in range(arr_shape):
        for j in range(arr_shape):
            dist_matrix[i,j] = binary_dis_C(matrix[i], matrix[j])
    return dist_matrix

#####
#####Begin by fitting a kernel density to our forecasts
# def clustering_alg(data):
# 	cdef int l, y, x
# 	cdef numpy.ndarray[numpy.float64_t, ndim= 1] ensemble
# 	cdef float bw
# 	mode_matrix = []
# 	cdef int arr_shape_l = data.shape[0]
# 	cdef int arr_shape_y = data.shape[2]
# 	cdef int arr_shape_x = data.shape[3]
# 	min_location = numpy.zeros((arr_shape_l, arr_shape_y, arr_shape_x), dtype=float)#//for every (l,y,x) we will have a value
# 	cluster_idx = numpy.zeros((arr_shape_l, arr_shape_y, arr_shape_x), dtype=float)
# 	for l in range(arr_shape_l):
# 		start_time = time.time()
# 		print (l)
# 		for y in range(arr_shape_y):
# 			for x in range(arr_shape_x):
# 				ensemble = data[l, :, y, x]
# 				bw = 0.45730505192732634 * numpy.std(ensemble)
# 				means,mins = root_finding(ensemble, bw)
# 				if len(means) == 2 and (4 < len(numpy.where(ensemble < mins[0])[0]) < 46) \
# 						and ((continuous_pdf(mins[0], ensemble, bw)/continuous_pdf(means[0], ensemble, bw)) < 0.85) \
# 							and ((continuous_pdf(mins[0], ensemble, bw)/continuous_pdf(means[1], ensemble, bw)) < 0.85):
# 								min_location[l, y, x] = mins[0]
# 								binary_ensemble = [0] * 50
# 								binary_ensemble = numpy.array(binary_ensemble, dtype=int)
# 								binary_ensemble[numpy.where(ensemble > mins[0])[0]] = 1
# 								mode_matrix.append(binary_ensemble)
# 		#print (time.time() - start_time)
# 	#####Now find clusters based on ensemble groupings
# 	start_time= time.time()
# 	clusters_1 = fclusterdata(mode_matrix, t=3 ** 0.5, criterion='distance', metric=binary_dis)
# 	print (time.time() - start_time, "clustering_1")
# 	#start_time= time.time()
# 	#dist_mode2 = C_pairwise(numpy.array(mode_matrix))
# 	#clusters_2 = AgglomerativeClustering(n_clusters = None, linkage = "single", affinity = "precomputed", compute_full_tree = True, distance_threshold = 3**0.5).fit_predict(dist_mode2)
# 	#print (time.time() - start_time, "clustering_2")
# 	cluster_idx[:] = numpy.nan
# 	i = 0
# 	for l in range(arr_shape_l):
# 		for y in range(arr_shape_y):
# 			for x in range(arr_shape_x):
# 				if min_location[l, y, x] != 0:
# 					cluster_idx[l, y, x] = clusters_1[i]
# 					i += 1
# 	common_clusters = list(
# 		dict(Counter(clusters_1).most_common(30)))  # //THESE ARE OUR MOST COMMON CLUSTERS looking at top 10
# 	cluster_medoids = []
# 	mode_matrix = numpy.array(mode_matrix)
# 	for c in common_clusters:
# 		clus_loc = numpy.where(clusters_1 == c)
# 		c_mode_matrix = mode_matrix[clus_loc[0]]
# 		#m_c1 = pairwise_distances(c_mode_matrix, metric=binary_dis)
# 		m_c1 = C_pairwise(c_mode_matrix)
# 		cluster_medoids.append(c_mode_matrix[numpy.argmin(m_c1.sum(axis=0))])  #// the minimum of each cluster is the medoid
# 	cluster_medoids = numpy.array(cluster_medoids)
# 	return cluster_medoids,cluster_idx, min_location


