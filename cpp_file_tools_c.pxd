#!python
#cython: boundscheck=False
#cython: wraparound=False

cimport numpy as np
cimport kohonen_neuron_c as kn
DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef class cpp_file_tools:
    cdef public int chan_count, group_chan
    cdef public object ext_img, save, show

    cdef public object stop, walk, stop_healthy, init_healthy, walk_healthy, stop_SCI, init_SCI, walk_SCI
    cdef public int stop_index, walk_index, first_chan, kc_col, kc_row, cue_col, result_col
    cdef public double kc_max_weight, kc_alpha, kc_neighbor, kc_min_win

    cpdef np.ndarray convert_brain_state(self, object obs)

    cpdef double success_rate(self, object l_res, object l_expected_res)
    cpdef double accuracy(self, l_res, l_expected_res)

