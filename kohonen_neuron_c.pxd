#!python
#cython: boundscheck=False
#cython: wraparound=False

cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef class Neurone:
    cdef public int weight_count, col, row, win_count
    cdef public np.ndarray weights
    cpdef double calc_error(self, np.ndarray[DTYPE_t, ndim=1] obs)
    cpdef double interneuron_dist(self, Neurone n2)
    cpdef double weights_dist(self, np.ndarray[DTYPE_t, ndim=1] w2)

cdef class Kohonen:
    cdef public int row, col, neighbor, min_win
    cdef public double alpha
    cdef public object network, good_neurons, groups, img_ext, show, save
    cpdef Neurone find_best_neuron(self, np.ndarray[DTYPE_t, ndim=1] obs)
    cpdef double find_mean_best_dist(self, np.ndarray[DTYPE_t, ndim=1] obs, int elements_range)


cdef class Group_neuron:
    cdef public int number
    cdef public np.ndarray template
    cdef public object plot_color, neurons, color, spikes
    cpdef double dist(self, np.ndarray[DTYPE_t, ndim=1] val)
    cpdef double min_dist(self, np.ndarray[DTYPE_t, ndim=1] val)
