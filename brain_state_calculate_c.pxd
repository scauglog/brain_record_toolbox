#!python
#cython: boundscheck=False

cimport numpy as np
cimport kohonen_neuron_c as kn
DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef class brain_state_calculate:
    cdef public int chan_count, weight_count
    cdef public object ext_img, save, show, name

    cdef public object stop, default_res, test_all, koho, mod_chan, verbose, use_obs_quantile, HMM
    cdef public int combination_to_test, history_length, koho_row, koho_col, neighbor, min_win, dist_count
    cdef public double alpha, max_weight, change_alpha_factor, tsa_alpha_start
    cdef public np.ndarray A, prevP, history, qVec
    cdef public int change_alpha_iteration, was_bad, tsa_max_iteration, tsa_max_accuracy, raw_res, result

    cpdef public int test_one_obs(self, object obs, object on_modulate_chan=*)
    cpdef int train_nets(self,object l_obs,object l_res,object cft,object with_RL=*,int obs_to_add=*,object train_mod_chan=*)
    cpdef np.ndarray compute_network_accuracy(self, object best_ns, object dist_res, np.ndarray obs)

