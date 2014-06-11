import numpy as np
import copy
import random as rnd
import math
from itertools import combinations
import pickle
from scipy.stats.mstats import mquantiles
import Tkinter
import tkFileDialog

import kohonen_neuron_c as kn


class brain_state_calculate:
    def __init__(self, weight_count, name='koho', ext_img='.png', save=False, show=False):
        #rnd.seed(42)
        #result for the state
        self.stop = [1, 0]
        self.default_res = [0, 0]
        self.ext_img = ext_img
        self.save = save
        self.show = show

        #params for Test
        self.test_all = False
        self.combination_to_test = 50
        #self.A = np.array([[0.99, 0.01], [0.01, 0.99]])
        self.A = np.array([[0.75, 0.25], [0.1, 0.9]])
        #history length should be a prime number
        self.history_length = 1

        #koho parameters
        self.alpha = 0.01
        self.koho_row = 1
        self.koho_col = 7
        self.koho = []
        #number of neighbor to update in the network
        self.neighbor = 3
        #min winning count to be consider as a good neuron
        self.min_win = 7
        #number of best neurons to keep for calculate distance of obs to the network
        self.dist_count = 3
        self.max_weight = 5
        self.weight_count = weight_count

        #simulated annealing parameters
        #change alpha each X iteration
        self.change_alpha_iteration = 7
        #change alpha by a factor of
        #/!\ should be float
        self.change_alpha_factor = 10.0

        #other parameter
        #store consecutive not a successfull trial used for train_nets
        self.was_bad = 0
        #channel modulated they are all modulated at the beginning
        self.mod_chan = range(self.weight_count)
        self.verbose = True
        self.name = name

        #quantile parameter
        #enable quantile
        self.use_obs_quantile = False
        quantile_step = 0.1
        self.qVec = np.array(np.arange(0.0, 1.0, quantile_step))
        if self.use_obs_quantile:
            self.A = np.array([[0.9, 0.1], [0.1, 0.9]])
            self.weight_count = self.qVec.shape[0]

        #train simulated annealing parameter
        self.tsa_alpha_start = 0.1
        self.tsa_max_iteration = 14
        self.tsa_max_accuracy = 0.99

    def build_networks(self):
        #build the network
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
        l_res, l_obs = cft.read_cpp_files(files, is_healthy=False, cut_after_cue=False)
        l_obs_koho = cft.obs_classify_kohonen(l_obs)
        #train networks
        self.build_networks()
        self.simulated_annealing(l_obs, l_obs_koho, l_res, self.tsa_alpha_start, self.tsa_max_iteration, self.tsa_max_accuracy, over_train_walk=True)
        if train_mod_chan:
            self.mod_chan = cft.get_mod_chan(l_obs)
        else:
            self.mod_chan = range(self.weight_count)

    def init_networks_on_files(self, initdir, cft, train_mod_chan=False):
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
        return 0

    def init_test(self, HMM = True):
        #initilise test for live processing
        self.history = np.array([self.stop])
        #matrix which contain the rank of the result
        self.prevP = np.array(self.stop)
        self.HMM = HMM
        self.raw_res = 0
        self.result = 0

    def test_one_obs(self, obs, on_modulate_chan=True):
        #we transform the obs according to our need
        if on_modulate_chan and not self.use_obs_quantile:
            obs = self.get_only_mod_chan(obs)
        if self.use_obs_quantile:
            obs = self.obs_to_quantiles(obs)
        #test one obs
        dist_res = np.arange(0, len(self.koho), 1.0)
        best_ns = []
        res = copy.copy(self.default_res)

        #find the best distance of the obs to each network
        for k in range(len(self.koho)):
            dist_res[k] = self.koho[k].find_mean_best_dist(obs, self.dist_count)
            #we add extra neurons to best_ns in order to remove null probability
            best_ns += self.koho[k].find_best_X_neurons(obs, self.dist_count+self.dist_count)
        self.raw_res = dist_res.argmin()
        if self.HMM:
            #flatten list
            #best_ns = [item for sublist in best_ns for item in sublist]

            prob_res = self.compute_network_accuracy(best_ns, dist_res, obs)

            #compute result with HMM
            P = np.arange(0, prob_res.shape[0], 1.0)
            for i in range(prob_res.shape[0]):
                P[i]=self.prevP.T.dot(self.A[:, i])*prob_res[i]
            #repair sum(P) == 1
            P = P.T/P.sum()

            #transform in readable result
            rank = P.argmax()
            res[rank] = 1

            #save P.T
            self.prevP = copy.copy(P.T)
        else:
            print "no HMM"
            #transform in readable result
            rank = dist_res.argmin()
            res[rank] = 1

        #use history to smooth change
        self.history = np.vstack((self.history, res))
        if self.history.shape[0] > self.history_length:
            self.history = self.history[1:, :]

        #transform in readable result
        rank = self.history.mean(0).argmax()
        self.result=rank
        return rank

    def test(self, l_obs, l_res, on_modulate_chan=True):
        good = 0

        #matrix which contain the rank of the result
        results_dict={'gnd_truth':[], self.name+'_raw':[], self.name:[]}
        self.init_test()

        for i in range(len(l_obs)):
            self.test_one_obs(l_obs[i], on_modulate_chan)
            results_dict[self.name+'_raw'].append(self.raw_res)
            results_dict[self.name].append(self.result)
            results_dict['gnd_truth'].append(np.array(l_res[i]).argmax())
            if self.result == np.array(l_res[i]).argmax():
                good += 1

        if len(l_obs) > 0:
            return good/float(len(l_obs)), results_dict
        else:
            print('l_obs is empty')
            return 0, {}

    def get_only_mod_chan(self, obs):
        obs_mod = copy.copy(np.array(map(float, obs)))
        #keep only chan that where modulated
        for c in range(obs_mod.shape[0]):
            if c not in self.mod_chan:
                obs_mod[c] = 0
        return obs_mod

    def obs_to_quantiles(self, obs):
        return mquantiles(obs, self.qVec)

    def compute_network_accuracy(self, best_ns, dist_res, obs):
        #we test combination of each best n
        #return an array of probability (prob of the obs to be state X)
        dist_comb = []
        if self.test_all:
            #test all combinations
            all_comb = combinations(best_ns, self.dist_count)
            for c in all_comb:
                l_dist = []
                for n in c:
                    l_dist.append(n.calc_error(obs))
                dist_comb.append(np.array(l_dist).mean())
        else:
            #test some combinations
            for c in range(self.combination_to_test):
                l_dist = []
                for n in self.random_combination(best_ns, self.dist_count):
                    l_dist.append(n.calc_error(obs))
                dist_comb.append(np.array(l_dist).mean())

        prob_res = np.arange(0, len(self.koho), 1.0)
        #sort each dist for combination and find where the result of each network is in the sorted list
        #this give a percentage of accuracy for the network
        dist_comb = np.array(sorted(dist_comb, reverse=True))
        for k in range(len(self.koho)):
            prob = abs(dist_comb-dist_res[k]).argmin()/float(len(dist_comb))
            if prob == 0.0:
                prob = 0.01
            prob_res[k] = prob

        return np.array(prob_res)

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
        #"Random selection from itertools.combinations(iterable, r)"
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(rnd.sample(xrange(n), r))
        return tuple(pool[i] for i in indices)

    def simulated_annealing(self, l_obs, l_obs_koho, l_res, alpha_start, max_iteration, max_success, over_train_walk=False):
        #inspired from simulated annealing, to determine when we should stop learning
        #initialize
        success, lor_trash = self.test(l_obs, l_res, on_modulate_chan=False)
        success -= 0.1
        alpha = alpha_start
        n = 0
        while success <= max_success and n < max_iteration:
            print(success)
            koho_cp = copy.copy(self.koho)
            #train each kohonen network
            for i in range(len(koho_cp)):
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
                if len(l_obs_koho[1]) > 0:
                    while cpt < len(l_obs_koho[0]):
                        koho_cp[1].algo_kohonen(l_obs_koho[1])
                        cpt += len(l_obs_koho[1])
            #compute success of the networks
            success_cp, lor_trash = self.test(l_obs, l_res, on_modulate_chan=False)
            #if we keep the same network for too long we go there
            if math.exp(-abs(success-success_cp)/(alpha*1.0)) in [0.0, 1.0]:
                print('break')
                break
            #simulated annealing criterion to keep or not the trained network
            if success < success_cp or rnd.random() < math.exp(-abs(success-success_cp)/(alpha*1.0)):
                success = copy.copy(success_cp)
                self.koho = copy.copy(koho_cp)

            #learning rate decrease over iteration
            #change learning rate
            if n % self.change_alpha_iteration == 0:
                alpha /= self.change_alpha_factor
            #alpha *= Lambda
            n += 1

    def train_nets(self, l_obs, l_res, cft, with_RL=True, obs_to_add=0, train_mod_chan=True):
        if not train_mod_chan:
            self.mod_chan = range(self.weight_count)

        if len(l_obs) <= 0:
            print "l_obs empty"
            return -1

        #we use l_obs_mod only to classify result
        if with_RL:
            save_koho = copy.copy(self.koho)
        success, l_of_res = self.test(l_obs, l_res, on_modulate_chan=False)
        success, l_of_res_classify = self.test(l_obs, l_res, on_modulate_chan=True)
        l_obs_koho = cft.obs_classify_mixed_res(l_obs, l_res, l_of_res_classify[self.name+'_raw'], obs_to_add)
        self.simulated_annealing(l_obs, l_obs_koho, l_res, self.tsa_alpha_start, self.tsa_max_iteration, self.tsa_max_accuracy)

        # success, l_of_res = self.test(l_obs, l_res, test_mod=False)
        #we look the walk in the raw result
        walk_get = np.nonzero(l_of_res[self.name+'_raw'])[0]
        if walk_get.shape[0] == 0:
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
            if win1 > win2:
                #update l_of_res in case the for loop are not in the else
                l_of_res = copy.copy(l_of_res_new)
                save_koho = copy.copy(self.koho)
                print("better with training --------")
            else:
                self.koho = save_koho
                print("worst with training")

                self.koho[1].alpha = 0.1
                self.koho[0].alpha = 0.1
                walk_get = np.nonzero(l_of_res[self.name+'_raw'])[0]
                walk_expected = np.nonzero(l_of_res['gnd_truth'])[0]
                for i in range(14):
                    #when algo say walk we try to exclude the obs from walk network and include it in rest network
                    if walk_get.shape[0] > 0:
                        for k in range(3):
                            obs_ind = walk_get[rnd.randrange(walk_get.shape[0])]
                            self.koho[0].update_closest_neurons(l_obs[obs_ind])
                            self.koho[1].update_closest_neurons(l_obs[obs_ind], push_away=True)

                    if walk_expected.shape[0] > 0:
                        #when we want walk we try to include the obs in walk and exclude it from rest
                        for k in range(3):
                            obs_ind = walk_expected[rnd.randrange(walk_expected.shape[0])]
                            self.koho[0].update_closest_neurons(l_obs[obs_ind], push_away=True)
                            self.koho[1].update_closest_neurons(l_obs[obs_ind])

                    success, l_of_res_new = self.test(l_obs, l_res, on_modulate_chan=False)
                    win1, win2 = cft.compare_result(l_of_res[self.name], l_of_res_new[self.name], l_of_res['gnd_truth'], True)
                    #if result are better we keep the network
                    if win1 > win2:
                        l_of_res = copy.copy(l_of_res_new)
                        save_koho = copy.copy(self.koho)
                        print("better ---")
                    else:
                        self.koho = save_koho
                        print("worst")

        if train_mod_chan:
            self.mod_chan = cft.get_mod_chan(l_obs)

        return 0

    def train_on_files(self, initdir, cft, is_healthy=False, new_day=True, obs_to_add=0, with_RL=True, train_mod_chan=True, on_stim=False):
        root = Tkinter.Tk()
        root.withdraw()
        file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=initdir,  title="select cpp file to train the classifier", filetypes=[('all files', '.*'), ('text files', '.txt')])
        if file_path == "":
            return -1

        paths = root.tk.splitlist(file_path)

        all_res, all_obs = cft.read_cpp_files(paths, is_healthy=is_healthy, cut_after_cue=False, init_in_walk=True, on_stim=on_stim)

        if new_day:
            self.train_nets_new_day(all_obs, all_res, cft)

        return self.train_nets(all_obs, all_res, cft, with_RL=with_RL, obs_to_add=obs_to_add, train_mod_chan=train_mod_chan)

    def train_one_file(self, filename, cft, is_healthy=False, new_day=True, obs_to_add=0, with_RL=True, train_mod_chan=True, on_stim=False):
        all_res, all_obs = cft.read_cpp_files([filename], is_healthy=is_healthy, cut_after_cue=False, init_in_walk=True, on_stim=on_stim)

        if new_day:
            self.train_nets_new_day(all_obs, all_res, cft)

        self.train_nets(all_obs, all_res, cft, with_RL=with_RL, obs_to_add=obs_to_add, train_mod_chan=train_mod_chan)

    def train_nets_new_day(self, l_obs, l_res, cft):
        if len(l_obs) <= 0:
            print "l_obs empty"
            return -1
        success, l_of_res = self.test(l_obs, l_res)
        l_obs_koho = cft.obs_classify_mixed_res(l_obs, l_res, l_of_res[self.name+'_raw'], 0)
        self.simulated_annealing(l_obs, l_obs_koho, l_res, self.tsa_alpha_start, self.tsa_max_iteration, self.tsa_max_accuracy)
        return 0

    def train_unsupervised_one_file(self, filename, cft, is_healthy=False, obs_to_add=-3, on_stim=False):
        l_res, l_obs = cft.read_cpp_files([filename], is_healthy=is_healthy, cut_after_cue=False, init_in_walk=True, on_stim=on_stim)
        if l_obs <= 0:
            print "l_obs empty"
        success, l_of_res = self.test(l_obs, l_res)
        l_obs_koho = cft.obs_classify_prev_res(l_obs, l_of_res[self.name], obs_to_add=obs_to_add)
        self.simulated_annealing(l_obs, l_obs_koho, l_res, self.tsa_alpha_start, self.tsa_max_iteration, self.tsa_max_accuracy, over_train_walk=True)

    def train_unsupervised_on_files(self, initdir, cft, is_healthy=False, obs_to_add=-3,  train_mod_chan=False, on_stim=False):
        if not train_mod_chan:
            self.mod_chan = range(self.weight_count)

        root = Tkinter.Tk()
        root.withdraw()
        file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=initdir,  title="select cpp file to train the classifier", filetypes=[('all files', '.*'), ('text files', '.txt')])
        if file_path == "":
            return -1

        paths = root.tk.splitlist(file_path)

        l_res, l_obs = cft.read_cpp_files(paths, is_healthy=is_healthy, cut_after_cue=False, init_in_walk=True, on_stim=on_stim)
        if len(l_obs) <= 0:
            print "l_obs empty"
            return -2

        success, l_of_res = self.test(l_obs, l_res)
        l_obs_koho = cft.obs_classify_prev_res(l_obs, l_of_res[self.name], obs_to_add=obs_to_add)
        self.simulated_annealing(l_obs, l_obs_koho, l_res, self.tsa_alpha_start, self.tsa_max_iteration, self.tsa_max_accuracy, over_train_walk=True)

        if train_mod_chan:
            self.mod_chan = cft.get_mod_chan(l_obs)
        return 0