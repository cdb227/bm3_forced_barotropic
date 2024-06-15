import numpy
import time
import operator
cimport numpy
from libc.math cimport exp, M_PI, sqrt
ctypedef numpy.float64_t DOUBLE_t

###
#Functions for KDE fitting
### Continuous PDF
def continuous_pdf(double x, double[:] mu, double s):
    cdef double mult, total
    cdef int arr_shape = mu.shape[0]
    cdef int i
    mult = (1 / (((2 * M_PI)**0.5) * arr_shape * s))
    total = 0.
    for i in range(arr_shape):
        total += exp(-((x - mu[i])**2.) / (2. * s ** 2.))
    return total*mult

#First Derivative of PDF
def first_deriv_pdf(double x, double[:] mu, double s):
    """
    evaluate the first derivative of a Gaussian KDE at x
    mu should be the array of ensemble members and
    s the bandwidth
    """
    cdef double mult, total, denom
    cdef int arr_shape = mu.shape[0]
    cdef int i
    mult = -(1 / ((sqrt(2 * M_PI)) * arr_shape * s**3.))
    denom = (2 * s ** 2.)
    total = 0
    for i in range(arr_shape):
        total += exp(-((x - mu[i]) ** 2.) / denom ) * (x - mu[i])
    return total*mult

#Second Derivative of PDF
def second_deriv_pdf(double x, double[:] mu, double s):
    cdef double mult
    mult = -(1. / (((2 * numpy.pi)**0.5) * len(mu) * s**3.))
    cdef double total
    total = 0
    cdef int arr_shape = mu.shape[0]
    cdef int i
    for i in range(arr_shape):
        total += exp(-((x - mu[i]) ** 2.) / (2 * s ** 2.))*(1-((x-mu[i])**2)/s)
    return total*mult

##
#helper functions
##
cdef bint is_root(double fa, double fb):
    return fa*fb<0 # check if f(a) , f(b) are opposite signs

cdef double calculate_std(double[:] arr): #calculate variance of arr
    cdef int n = arr.shape[0]
    cdef double mean = 0.0
    cdef double var = 0.0
    cdef double std_dev = 0.0
    cdef int i
    
    # Calculate mean
    for i in range(n):
        mean += arr[i]
    mean /= n
    
    # Calculate variance
    for i in range(n):
        var += (arr[i] - mean) * (arr[i] - mean)
    var /= n
    
    return sqrt(var)

def minmax(double[:] arr): # find min, max of arr
    cdef int i
    cdef int n = arr.shape[0]
    cdef DOUBLE_t min_val = arr[0]
    cdef DOUBLE_t max_val = arr[0]

    for i in range(1, n):
        if arr[i] < min_val:
            min_val = arr[i]
        elif arr[i] > max_val:
            max_val = arr[i]

    return min_val, max_val
   
##
#root finding functions + bm fitting
##
cpdef list recursive_rootfinding(double a, double b, double itol, double mtol, double[:] ens, double bw ):
    cdef double mp = (b + a) / 2.0 #mid point
    cdef list roots = [] #list for roots
    
    if b - a > itol:
        roots.extend(recursive_rootfinding(a, mp, itol,mtol,ens,bw))  # Check left side
        roots.extend(recursive_rootfinding(mp, b, itol,mtol,ens,bw))  # Check right side
    else:
        if is_root(first_deriv_pdf(a, ens, bw), first_deriv_pdf(b, ens, bw)): #check if we switch signs
            roots.append(mp)
    itol = max(itol*0.9, mtol) #adaptive tolerance
    return roots

def find_bimodality(double[:] ensemble):
    #calculate bw
    cdef int ens_size = ensemble.shape[0]
    cdef double estd_dev = calculate_std(ensemble)
    cdef bw = (ens_size ** (-1.0 / 5.0)) * estd_dev
    
    #calculate stuff for rootfinding
    cdef double m, M
    cdef double mtol = bw/10 # minimum tolerance
    cdef double itol = bw/5  # initial tolerance
    
    m, M = minmax(ensemble)

    cdef list roots = recursive_rootfinding(m, M, itol, mtol, ensemble, bw)
    
    cdef list Ms = roots[::2]  # maxima
    cdef list ms = roots[1::2] # minima
    
    cdef int m1_mem = 0 # size of mode 1

    cdef int mem_req = ens_size//10 # mem req is 10% of ensemble
    
    cdef i

    if len(Ms)==2:
        for i in range(ens_size):
            if ensemble[i] < ms[0]:
                m1_mem+=1      
    cdef bint bimodal = (ens_size - mem_req >= m1_mem >= mem_req)
    
    return bimodal



def apply_find_bimodality(numpy.ndarray[numpy.float64_t, ndim=4] large_array):
    cdef int t = large_array.shape[1]
    cdef int y = large_array.shape[2]
    cdef int x = large_array.shape[3]
    cdef int i,j,k
    
    cdef numpy.ndarray[numpy.uint8_t, ndim=3] results = numpy.zeros((t, y, x), dtype=numpy.uint8)

    large_array = numpy.ascontiguousarray(large_array)
    
    for i in range(t):
        for j in range(y):
            for k in range(x):
                results[i, j, k] = find_bimodality(large_array[:, i, j, k])
    
    return results



