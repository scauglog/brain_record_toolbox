#!python
#cython: boundscheck=False

import numpy as np
import copy
import random as rnd
import math
from itertools import combinations
import pickle
from scipy.stats.mstats import mquantiles
import Tkinter
import tkFileDialog
import settings
import time
from os.path import basename, splitext
import kohonen_neuron_c as kn

cdef class brain_state_calculate:
    def __init__(self, int weight_count, name='koho', ext_img='.png', save=False, show=False, settings_path="classifierSettings.yaml"):
        self.ext_img = ext_img
        self.save = save
        self.show = show
        self.chan_count = weight_count
        self.weight_count = weight_count
        self.name = name
        self.load_settings(settings_path)

    def load_settings(self, settings_path):
        cset = settings.Settings(settings_path).get()

        #rnd.seed(42)
        #result for the state
        self.stop = cset['stop']
        self.default_res = cset['default_res']

        #params for Test
        self.test_all = cset['test_all']
        self.combination_to_test = cset['combination_to_test']
        #self.A = np.array([[0.99, 0.01], [0.01, 0.99]])
        self.A = np.array(cset['A'])
        #history length should be a prime number
        self.history_length = cset['history_length']

        #koho parameters
        self.alpha = cset['alpha']
        self.koho_row = cset['koho_row']
        self.koho_col = cset['koho_col']
        self.koho = []
        #number of neighbor to update in the network
        self.neighbor = cset['neighbor']
        #min winning count to be consider as a good neuron
        self.min_win = cset['min_win']
        #number of best neurons to keep for calculate distance of obs to the network
        self.dist_count = cset['dist_count']
        self.max_weight = cset['max_weight']

        #simulated annealing parameters
        #change alpha each X iteration
        self.change_alpha_iteration = cset['change_alpha_iteration']
        #change alpha by a factor of
        #/!\ should be float
        self.change_alpha_factor = cset['change_alpha_factor']

        #other parameter
        #store consecutive not a successfull trial used for train_nets
        self.was_bad = 0
        #channel modulated they are all modulated at the beginning
        self.mod_chan = range(self.chan_count)
        self.verbose = cset['verbose']

        #quantile parameter
        #enable quantile
        self.use_quantile_shrink(cset['use_obs_quantile'], step=cset['quantile_step'])

        #train simulated annealing parameter
        self.tsa_alpha_start = cset['tsa_alpha_start']
        self.tsa_max_iteration = cset['tsa_max_iteration']
        self.tsa_max_accuracy = cset['tsa_max_accuracy']

    def use_quantile_shrink(self, use, step=0):
        self.use_obs_quantile = use
        self.qVec = np.arange(0.0, 1.0, step)
        if use:
            self.weight_count = self.qVec.shape[0]

    def build_networks(self):
        #build the network
        cdef kn.Kohonen koho_stop, koho_walk
        koho_stop = kn.Kohonen(self.koho_row, self.koho_col, self.weight_count, self.max_weight, self.alpha, self.neighbor, self.min_win, self.ext_img, self.save, self.show, rnd.random())
        koho_walk = kn.Kohonen(self.koho_row, self.koho_col, self.weight_count, self.max_weight, self.alpha, self.neighbor, self.min_win, self.ext_img, self.save, self.show, rnd.random())
        self.koho = [koho_stop, koho_walk]

    def load_networks(self, path):
        print(path)
        pkl_file = open(path, 'rb')
        dict = pickle.load(pkl_file)
        keys = dict.keys()
        self.koho = dict['networks']
        if 'mod_chan' in keys:
            self.mod_chan = dict['mod_chan']
        if 'A' in keys:
            self.A = dict['A']
        if 'history_length' in keys:
            self.history_length = dict['history_length']
        if 'use_obs_quantile' in keys:
            self.use_obs_quantile = dict['use_obs_quantile']
        if 'qVec' in keys:
            self.qVec = dict['qVec']
        if 'weight_count' in keys:
            self.weight_count = dict['weight_count']

        print(len(self.koho))

    def load_networks_file(self, initdir):
        root = Tkinter.Tk()
        root.withdraw()
        file_path = tkFileDialog.askopenfilename(initialdir=initdir, title="select classifier file",filetypes=[('all files', '.*'), ('classifier', '.pyObj')])
        if file_path == "":
            return -1
        self.load_networks(file_path)
        return 0

    def save_obj(self, filename):
        dict = {'networks': self.koho, 'mod_chan': self.mod_chan}
        dict['A'] = self.A
        dict['history_length'] = self.history_length
        dict['use_obs_quantile'] = self.use_obs_quantile
        if self.use_obs_quantile:
            dict['qVec'] = self.qVec

        dict['weight_count'] = self.weight_count

        if self.koho == [] or self.mod_chan == []:
            return -2

        with open(filename, 'wb') as my_file:
            my_pickler = pickle.Pickler(my_file)
            my_pickler.dump(dict)
        return 0

    def save_networks(self, dir_name, date):
        #save networks
        print('Saving network')
        return self.save_obj(dir_name + 'koho_networks_' + date)

    def save_networks_on_file(self, initdir, date):
        root = Tkinter.Tk()
        root.withdraw()
        file_path = tkFileDialog.asksaveasfilename(initialdir=initdir, title="save as", initialfile="koho_networks_"+date, defaultextension="pyObj")
        if file_path == "":
            return -1
        return self.save_obj(str(file_path))


    def init_networks(self, files, cft, train_mod_chan=False):
        l_res, l_obs = cft.read_cpp_files(files, use_classifier_result=False, cut_after_cue=False)
        l_obs_koho = cft.obs_classify_kohonen(l_obs)
        #train networks
        self.build_networks()
        cpt = 0
        while cpt < 700:
            cpt += len(l_res)
            self.simulated_annealing(l_obs, l_obs_koho, l_res, self.tsa_alpha_start, self.tsa_max_iteration, self.tsa_max_accuracy, over_train_walk=True)
        if train_mod_chan:
            self.mod_chan = cft.get_mod_chan(l_obs)
        else:
            self.mod_chan = range(self.chan_count)

    def init_networks_on_files(self, initdir, cft, train_mod_chan=False, autosave=False):
        root = Tkinter.Tk()
        root.withdraw()
        file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=initdir, title="select cpp file to initialize the classifier")
        if file_path == "":
            return -1
        files = root.tk.splitlist(file_path)
        try:
            self.init_networks(files, cft, train_mod_chan=train_mod_chan)
        except Exception:
            return -2
        if autosave:
            self.save_obj(files[-1]+str(time.time())+'.pyObj')
        return 0

    def swap(self):
        self.koho[0], self.koho[1] = self.koho[1], self.koho[0]

    def init_test(self, HMM = True):
        #initilise test for live processing
        self.history = np.array([self.stop])
        #matrix which contain the rank of the result
        self.prevP = np.array(self.stop)
        self.HMM = HMM
        self.raw_res = 0
        self.result = 0

    def set_HMM(self,HMM):
        self.A = np.array(HMM)

    def get_HMM(self):
        return self.A.tolist()

    cpdef int test_one_obs(self, object tmp_obs, object on_modulate_chan=True):
        cdef np.ndarray[DTYPE_t, ndim=1] dist_res, P, obs
        cdef int k, i, rank
        cdef kn.Neurone neur
        #we should tranform
        obs = <np.ndarray[DTYPE_t, ndim=1]> np.array(tmp_obs)
        #we transform the obs according to our need
        if on_modulate_chan and not self.use_obs_quantile:
            obs = self.get_only_mod_chan(obs)
        if self.use_obs_quantile:
            obs = self.obs_to_quantiles(obs, on_modulate_chan=on_modulate_chan)
        #test one obs
        dist_res = np.arange(0, <int>len(self.koho), 1.0)
        best_ns = []
        res = copy.deepcopy(self.default_res)

        #find the best distance of the obs to each network
        for k in range(<int>len(self.koho)):
            # best_ns.append()
            # best_ns[k]=self.koho[k].find_best_X_neurons(obs, self.dist_count)
            # dist_res[k] = self.koho[k].find_mean_best_dist(obs, self.dist_count)
            #we add extra neurons to best_ns in order to remove null probability
            tmp_ns = self.koho[k].find_best_X_neurons(obs, self.dist_count)
            best_ns += tmp_ns
            mean = 0
            for neur in tmp_ns:
                mean += neur.calc_error(obs)
            dist_res[k] = mean/float(self.dist_count)

        self.raw_res = <int> dist_res.argmin()
        if self.HMM:
            #flatten list
            #best_ns = [item for sublist in best_ns for item in sublist]

            #prob_res = self.compute_network_accuracy_p(best_ns,obs)
            prob_res = self.compute_network_accuracy(best_ns, dist_res, obs)

            #compute result with HMM
            P = np.arange(0, prob_res.shape[0], 1.0)
            for i in range(<int>prob_res.shape[0]):
                P[i]=<double>self.prevP.T.dot(self.A[:, i])*prob_res[i]
            #repair sum(P) == 1
            P = P.T/P.sum()

            #transform in readable result
            rank = <int>P.argmax()
            res[rank] = 1

            #save P.T
            self.prevP = copy.deepcopy(P.T)
        else:
            print "no HMM"
            #transform in readable result
            rank = <int>dist_res.argmin()
            res[rank] = 1

        #use history to smooth change
        if self.history_length > 1:
            self.history = np.vstack((self.history, res))
            if <int>self.history.shape[0] > self.history_length:
                self.history = self.history[1:, :]

            #transform in readable result
            rank = <int>self.history.mean(0).argmax()

        self.result=rank
        return rank

    def test(self, l_obs, l_res, on_modulate_chan=True):
        cdef int good, i
        good = 0

        #matrix which contain the rank of the result
        results_dict={'gnd_truth':[], self.name+'_raw':[], self.name:[]}
        self.init_test()

        for i in range(<int>len(l_obs)):
            self.test_one_obs(l_obs[i], on_modulate_chan)
            results_dict[self.name+'_raw'].append(self.raw_res)
            results_dict[self.name].append(self.result)
            results_dict['gnd_truth'].append(<int>np.array(l_res[i]).argmax())
            if self.result == <int>np.array(l_res[i]).argmax():
                good += 1

        if <int>len(l_obs) > 0:
            return <double>good/float(len(l_obs)), results_dict
        else:
            print('l_obs is empty')
            return <double>0.0, {}

    def get_only_mod_chan(self, obs):
        cdef int c
        obs_mod = copy.deepcopy(np.array(map(float, obs)))
        #keep only chan that where modulated
        for c in range(<int>obs_mod.shape[0]):
            if c not in self.mod_chan:
                obs_mod[c] = 0
        return obs_mod

    def obs_to_quantiles(self, obs, on_modulate_chan=False):
        if on_modulate_chan:
            return mquantiles(obs[self.mod_chan], self.qVec)
        else:
            return mquantiles(obs, self.qVec)

    cpdef np.ndarray compute_network_accuracy(self, object best_ns, object dist_res, np.ndarray obs):
        #we test combination of each best n
        #return an array of probability (prob of the obs to be state X)
        cdef np.ndarray prob_res, dist_comb
        cdef int k
        cdef double prob
        cdef kn.Neurone n
        l_dist_comb = []
        if self.test_all:
            #test all combinations
            all_comb = combinations(best_ns, self.dist_count)
            for c in all_comb:
                l_dist = []
                for n in c:
                    l_dist.append(<double>n.calc_error(obs))
                l_dist_comb.append(<double>np.array(l_dist).mean())
        else:
            #test some combinations
            for c in range(self.combination_to_test):
                l_dist = []
                for n in self.random_combination(best_ns, self.dist_count):
                    l_dist.append(<double>n.calc_error(obs))
                l_dist_comb.append(<double>np.array(l_dist).mean())

        prob_res = <np.ndarray[DTYPE_t, ndim=1]> np.arange(0, <int>len(self.koho), 1.0)
        #sort each dist for combination and find where the result of each network is in the sorted list
        #this give a percentage of accuracy for the network
        dist_comb = <np.ndarray[DTYPE_t, ndim=1]> np.array(sorted(l_dist_comb, reverse=True))
        for k in range(<int>len(self.koho)):
            prob = <double>abs(dist_comb-dist_res[k]).argmin()/float(len(dist_comb))
            if prob == 0.0:
                prob = 0.01
            prob_res[k] = prob

        return prob_res

    def compute_network_accuracy_p(self, best_ns, obs):
        #return an array of probability (prob of the obs to be state X)
        #prob is computed using the class of the nearest neighbor of the obs
        #store all distance
        dist_all = []
        #store all distance to a specific network
        dist_net = []
        #find the distance between the obs and each nearest neighbor
        for k in best_ns:
            tmp_l = []
            for n in k:
                dst = n.calc_error(obs)
                dist_all.append(dst)
                tmp_l.append(dst)

            dist_net.append(tmp_l)
        dist_all.sort()
        prob_res = []
        #for each state (network)
        for net in dist_net:
            count = 0
            for i in range(self.dist_count):
                if dist_all[i] in net:
                    count += 1
            prob_res.append(count/float(self.dist_count))
        return np.array(prob_res)

    @staticmethod
    def random_combination(iterable, r):
        cdef int i, n
        #"Random selection from itertools.combinations(iterable, r)"
        pool = tuple(iterable)
        n = <int>len(pool)
        indices = sorted(rnd.sample(xrange(n), r))
        return tuple(pool[i] for i in indices)

    def simulated_annealing(self, l_obs, l_obs_koho, l_res, double alpha_start, int max_iteration, double max_success, over_train_walk=False):
        #inspired from simulated annealing, to determine when we should stop learning
        #initialize
        cdef double success, success_cp, alpha
        cdef int n, i, cpt
        success, lor_trash = self.test(l_obs, l_res, on_modulate_chan=False)
        success -= 0.1
        alpha = alpha_start
        n = 0
        while success <= max_success and n < max_iteration:
            koho_cp = copy.deepcopy(self.koho)
            #train each kohonen network
            for i in range(<int>len(koho_cp)):
                #update learning coefficient
                koho_cp[i].alpha = alpha
                #no neighbor decrease for the first iteration
                if n == 0:
                    koho_cp[i].algo_kohonen(l_obs_koho[i], False)
                else:
                    koho_cp[i].algo_kohonen(l_obs_koho[i])
            #we train walk multiple time to have same training has rest
            if over_train_walk:
                cpt = 0
                if <int>len(l_obs_koho[1]) > 0:
                    while cpt < <int>len(l_obs_koho[0]):
                        koho_cp[1].algo_kohonen(l_obs_koho[1])
                        cpt += <int>len(l_obs_koho[1])
            #compute success of the networks
            success_cp, lor_trash = self.test(l_obs, l_res, on_modulate_chan=False)
            #if we keep the same network for too long we go there
            if <double>math.exp(-abs(success-success_cp)/(alpha*1.0)) in [0.0, 1.0]:
                print('break')
                break
            #simulated annealing criterion to keep or not the trained network
            if success < success_cp or <double>rnd.random() < <double>math.exp(-abs(success-success_cp)/(alpha*1.0)):
                success = copy.deepcopy(success_cp)
                self.koho = copy.deepcopy(koho_cp)

            #learning rate decrease over iteration
            #change learning rate
            if n % self.change_alpha_iteration == 0:
                alpha /= self.change_alpha_factor
            #alpha *= Lambda
            n += 1

    cpdef int train_nets(self, object l_obs, object l_res, object cft, object with_RL=True, int obs_to_add=0, object train_mod_chan=True):
        cdef double success
        #cdef np.ndarray[np.int_t, ndim=1] walk_get, walk_expected
        cdef int win1, win2
        #cdef int i, k, obs_ind
        if not train_mod_chan:
            self.mod_chan = range(self.chan_count)

        if <int>len(l_obs) <= 0:
            print "l_obs empty"
            return -1

        #we use l_obs_mod only to classify result
        if with_RL:
            save_koho = copy.deepcopy(self.koho)

        success, l_of_res = self.test(l_obs, l_res, on_modulate_chan=False)
        success, l_of_res_classify = self.test(l_obs, l_res, on_modulate_chan=True)
        l_obs_koho = cft.obs_classify_mixed_res(l_obs, l_res, l_of_res_classify[self.name], obs_to_add)
        self.simulated_annealing(l_obs, l_obs_koho, l_res, self.tsa_alpha_start, self.tsa_max_iteration, self.tsa_max_accuracy)

        # success, l_of_res = self.test(l_obs, l_res, test_mod=False)
        #we look the walk in the raw result
        walk_get = np.nonzero(l_of_res[self.name])[0]
        if <int>walk_get.shape[0] == 0:
            self.was_bad += 1
            if self.was_bad > 1:
                l_obs_koho = cft.obs_classify_kohonen(l_obs)
                self.simulated_annealing(l_obs, l_obs_koho, l_res, self.tsa_alpha_start, self.tsa_max_iteration, self.tsa_max_accuracy)
                success, l_of_res = self.test(l_obs, l_res)
        else:
            self.was_bad = 0

        if with_RL:
            success, l_of_res_new = self.test(l_obs, l_res, on_modulate_chan=False)
            win1, win2 = cft.compare_result(l_of_res[self.name], l_of_res_new[self.name], l_of_res['gnd_truth'], True)
            if win2 >= win1:
                #update l_of_res in case the for loop are not in the else
                l_of_res = copy.deepcopy(l_of_res_new)
                save_koho = copy.deepcopy(self.koho)
                print("better with training --------")
            else:
                self.koho = copy.deepcopy(save_koho)
                print("worst with training")

                self.koho[1].alpha = 0.1
                self.koho[0].alpha = 0.1
                walk_get = np.nonzero(l_of_res[self.name])[0]
                walk_expected = np.nonzero(l_of_res['gnd_truth'])[0]
                for i in range(14):
                    #when algo say walk we try to exclude the obs from walk network and include it in rest network
                    if <int>walk_get.shape[0] > 0:
                        for k in range(3):
                            obs_ind = walk_get[rnd.randrange(walk_get.shape[0])]
                            self.koho[0].update_closest_neurons(<np.ndarray[DTYPE_t, ndim=1]>l_obs[obs_ind])
                            self.koho[1].update_closest_neurons(<np.ndarray[DTYPE_t, ndim=1]>l_obs[obs_ind], push_away=True)

                    if <int>walk_expected.shape[0] > 0:
                        #when we want walk we try to include the obs in walk and exclude it from rest
                        for k in range(3):
                            obs_ind = walk_expected[rnd.randrange(walk_expected.shape[0])]
                            self.koho[0].update_closest_neurons(<np.ndarray[DTYPE_t, ndim=1]>l_obs[obs_ind], push_away=True)
                            self.koho[1].update_closest_neurons(<np.ndarray[DTYPE_t, ndim=1]>l_obs[obs_ind])
                    success, l_of_res_new = self.test(l_obs, l_res, on_modulate_chan=False)
                    win1, win2 = cft.compare_result(l_of_res[self.name], l_of_res_new[self.name], l_of_res['gnd_truth'], True)
                    #if result are better we keep the network
                    if win2 > win1:
                        l_of_res = copy.deepcopy(l_of_res_new)
                        save_koho = copy.deepcopy(self.koho)
                        print("better ---")
                    else:
                        self.koho = save_koho
                        print("worst")

        if train_mod_chan:
            self.mod_chan = cft.get_mod_chan(l_obs)
        return 0

    def train_on_files(self, initdir, cft, is_healthy=False, new_day=True, obs_to_add=0, with_RL=True, train_mod_chan=True, on_stim=False, autosave=False):
        root = Tkinter.Tk()
        root.withdraw()
        file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=initdir,  title="select cpp file to train the classifier", filetypes=[('all files', '.*'), ('text files', '.txt')])
        if file_path == "":
            return -1

        paths = root.tk.splitlist(file_path)

        all_res, all_obs = cft.read_cpp_files(paths, use_classifier_result=is_healthy, cut_after_cue=False,
                                              init_in_walk=True, on_stim=on_stim)

        if new_day:
            self.train_nets_new_day(all_obs, all_res, cft)

        return_value = self.train_nets(all_obs, all_res, cft, with_RL=with_RL, obs_to_add=obs_to_add, train_mod_chan=train_mod_chan)
        if autosave:
            self.save_obj(paths[-1]+str(time.time())+'.pyObj')
        return return_value

    def train_one_file(self, filename, cft, is_healthy=False, new_day=True, obs_to_add=0, with_RL=True, train_mod_chan=True, on_stim=False, autosave=False):
        all_res, all_obs = cft.read_cpp_files([filename], use_classifier_result=is_healthy, cut_after_cue=False,
                                              init_in_walk=True, on_stim=on_stim)

        if new_day:
            self.train_nets_new_day(all_obs, all_res, cft)

        self.train_nets(all_obs, all_res, cft, with_RL=with_RL, obs_to_add=obs_to_add, train_mod_chan=train_mod_chan)
        if autosave:
            self.save_obj(filename+str(time.time())+'.pyObj')

    def train_nets_new_day(self, l_obs, l_res, cft):
        if <int>len(l_obs) <= 0:
            print "l_obs empty"
            return -1
        success, l_of_res = self.test(l_obs, l_res, on_modulate_chan=True)
        l_obs_koho = cft.obs_classify_mixed_res(l_obs, l_res, l_of_res[self.name], 0)
        self.simulated_annealing(l_obs, l_obs_koho, l_res, self.tsa_alpha_start, self.tsa_max_iteration, self.tsa_max_accuracy)
        return 0

    def train_unsupervised_one_file(self, filename, cft, is_healthy=False, obs_to_add=-3, train_mod_chan=False, on_stim=False, autosave=False):
        if not train_mod_chan:
            self.mod_chan = range(self.chan_count)

        l_res, l_obs = cft.read_cpp_files([filename], use_classifier_result=is_healthy, cut_after_cue=False,
                                          init_in_walk=True, on_stim=on_stim)
        if l_obs <= 0:
            print "l_obs empty"
        success, l_of_res = self.test(l_obs, l_res)
        l_obs_koho = cft.obs_classify_prev_res(l_obs, l_of_res[self.name], obs_to_add=obs_to_add)
        self.simulated_annealing(l_obs, l_obs_koho, l_res, self.tsa_alpha_start, self.tsa_max_iteration, self.tsa_max_accuracy, over_train_walk=True)

        if train_mod_chan:
            self.mod_chan = cft.get_mod_chan(l_obs)
        if autosave:
            self.save_obj(filename+str(time.time())+'.pyObj')

    def train_unsupervised_on_files(self, initdir, cft, is_healthy=False, obs_to_add=-3,  train_mod_chan=False, on_stim=False, autosave=False):
        if not train_mod_chan:
            self.mod_chan = range(self.weight_count)

        root = Tkinter.Tk()
        root.withdraw()
        file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=initdir,  title="select cpp file to train the classifier", filetypes=[('all files', '.*'), ('text files', '.txt')])
        if file_path == "":
            return -1

        paths = root.tk.splitlist(file_path)

        l_res, l_obs = cft.read_cpp_files(paths, use_classifier_result=is_healthy, cut_after_cue=False,
                                          init_in_walk=True, on_stim=on_stim)
        if <int>len(l_obs) <= 0:
            print "l_obs empty"
            return -2

        success, l_of_res = self.test(l_obs, l_res)
        l_obs_koho = cft.obs_classify_prev_res(l_obs, l_of_res[self.name], obs_to_add=obs_to_add)
        self.simulated_annealing(l_obs, l_obs_koho, l_res, self.tsa_alpha_start, self.tsa_max_iteration, self.tsa_max_accuracy, over_train_walk=True)

        if train_mod_chan:
            self.mod_chan = cft.get_mod_chan(l_obs)
        if autosave:
            self.save_obj(paths[-1]+str(time.time())+'.pyObj')
        return 0

    def test_classifier_on_file(self, cft, initdir, on_modulate_chan=False, gui=False, include_classifier_result=True, save_folder=""):
        root = Tkinter.Tk()
        root.withdraw()
        file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=initdir,  title="select cpp file to test the classifier", filetypes=[('all files', '.*'), ('text files', '.txt')])
        if file_path == "":
            return -1
        paths = root.tk.splitlist(file_path)
        if save_folder == "":
            save_folder = initdir

        for path in paths:
            l_res, l_obs = cft.read_cpp_files([path], use_classifier_result=False, cut_after_cue=False, init_in_walk=True)
            if <int>len(l_obs) > 0:
                success, l_of_res = self.test(l_obs, l_res, on_modulate_chan=on_modulate_chan)
                if include_classifier_result:
                    l_res, l_obs = cft.read_cpp_files([path], use_classifier_result=True, cut_after_cue=False, init_in_walk=True)
                    l_of_res["file_result"] = np.array(l_res).argmax(1)
                cft.plot_result(l_of_res, big_figure=False, dir_path=save_folder, extra_txt=splitext(basename(path))[0]+str(time.time()), gui=gui)
            else:
                print "empty file"