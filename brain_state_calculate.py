import kohonen_neuron_c as kn
import csv
import numpy as np
import copy
import random as rnd
import math
import matplotlib.pyplot as plt
from itertools import combinations
import pickle
from scipy.stats.mstats import mquantiles

class brain_state_calculate:
    def __init__(self, weight_count, name='koho', ext_img='.png', save=False, show=False):
        rnd.seed(42)
        #result for the state
        self.stop = [1, 0]
        self.default_res = [0, 0]
        self.ext_img = ext_img
        self.save = save
        self.show = show

        #params for Test
        self.test_all = False
        self.combination_to_test = 50
        self.A = np.array([[0.99, 0.01], [0.01, 0.99]])
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

    def build_networks(self):
        #build the network
        koho_stop = kn.Kohonen(self.koho_row, self.koho_col, self.weight_count, self.max_weight, self.alpha, self.neighbor, self.min_win, self.ext_img, self.save, self.show, rnd.random())
        koho_walk = kn.Kohonen(self.koho_row, self.koho_col, self.weight_count, self.max_weight, self.alpha, self.neighbor, self.min_win, self.ext_img, self.save, self.show, rnd.random())
        self.koho = [koho_stop, koho_walk]

    def load_networks(self, path):
        print path
        pkl_file = open(path, 'rb')
        dict = pickle.load(pkl_file)
        self.koho = dict['networks']
        self.mod_chan = dict['mod_chan']
        print len(self.koho)

    def save_networks(self, dir_name, date):
        #save networks
        print 'Saving network'
        dict={'networks': self.koho, 'mod_chan': self.mod_chan}
        with open(dir_name + 'koho_networks_' + date, 'wb') as my_file:
            my_pickler = pickle.Pickler(my_file)
            my_pickler.dump(dict)

    def init_networks(self, l_obs, l_res, cft):
        l_obs_koho = cft.obs_classify_kohonen(l_obs)
        #train networks
        self.mod_chan = cft.get_mod_chan(l_obs)
        self.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.95)

    def init_test(self):
        #initilise test for live processing
        self.history = np.array([self.stop])
        #matrix which contain the rank of the result
        self.prevP = np.array(self.stop)
        self.HMM=True
        self.raw_res=0
        self.result=0

    def test_one_obs(self, obs, on_modulate_chan=True):
        if on_modulate_chan:
            obs = self.get_only_mod_chan(obs)
        #test one obs
        dist_res = []
        best_ns = []
        res = copy.copy(self.default_res)

        #find the best distance of the obs to each network
        for k in self.koho:
            dist_res.append(k.find_mean_best_dist(obs, self.dist_count))
            #we add extra neurons to best_ns in order to remove null probability
            best_ns.append(k.find_best_X_neurons(obs, self.dist_count+self.dist_count))
        self.raw_res=np.array(dist_res).argmin()
        if self.HMM:
            #flatten list
            best_ns = [item for sublist in best_ns for item in sublist]

            prob_res = self.compute_network_accuracy(best_ns, dist_res, obs)

            #compute result with HMM
            P = []
            for k in range(prob_res.shape[0]):
                P.append(self.prevP.T.dot(self.A[:, k])*prob_res[k])
            #repair sum(P) == 1
            P = np.array(P).T/sum(P)

            #transform in readable result
            rank = P.argmax()
            res[rank] = 1

            #save P.T
            self.prevP = copy.copy(P.T)
        else:
            #transform in readable result
            rank = np.array(dist_res).argmin()
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
        results = []
        raw_res = []
        self.init_test()

        for i in range(len(l_obs)):
            self.test_one_obs(l_obs[i], on_modulate_chan)
            results_dict[self.name+'_raw'].append(self.raw_res)
            results_dict[self.name].append(self.result)
            raw_res.append(copy.copy(self.raw_res))
            results.append(copy.copy(self.result))
            if self.result == np.array(l_res[i]).argmax():
                good += 1

        if len(l_obs) > 0:
            results_dict['gnd_truth'] = np.array(l_res).argmax(1)
            return good/float(len(l_obs)), results_dict
        else:
            print ('l_obs is empty')
            return 0, {}

    def get_only_mod_chan(self, obs):
        obs_mod = copy.copy(np.array(obs))
        #keep only chan that where modulated
        for c in range(obs_mod.shape[0]):
            if c not in self.mod_chan:
                obs_mod[c] = 0
        return obs_mod

    @staticmethod
    def obs_to_quantiles(obs):
        #9 quantiles from 10% to 90% step 10%
        qVec=np.array(np.arange(0.1, 1, 0.1))
        return mquantiles(obs, qVec)

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

        prob_res = []
        #sort each dist for combination and find where the result of each network is in the sorted list
        #this give a percentage of accuracy for the network
        dist_comb = np.array(sorted(dist_comb, reverse=True))
        for k in range(len(self.koho)):
            prob = abs(dist_comb-dist_res[k]).argmin()/float(len(dist_comb))
            if prob == 0.0:
                prob = 0.01
            prob_res.append(prob)

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
                print 'break'
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

    def train_nets(self, l_obs, l_res, cft, with_RL=True, obs_to_add=0):
        #we use l_obs_mod only to classify result
        if with_RL:
            save_koho = copy.copy(self.koho)
        success, l_of_res = self.test(l_obs, l_res, on_modulate_chan=False)
        success, l_of_res_classify = self.test(l_obs, l_res, on_modulate_chan=True)
        l_obs_koho = cft.obs_classify_mixed_res(l_obs, l_res, l_of_res_classify[self.name], obs_to_add)
        self.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99)

        # success, l_of_res = self.test(l_obs, l_res, test_mod=False)
        #we look the walk in the raw result
        walk_get = np.nonzero(l_of_res[self.name+'_raw'])[0]
        if walk_get.shape[0] == 0:
            self.was_bad += 1
            if self.was_bad > 1:
                l_obs_koho = cft.obs_classify_kohonen(l_obs)
                self.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99)
                success, l_of_res = self.test(l_obs, l_res)
        else:
            self.was_bad = 0

        if with_RL:
            success, l_of_res_new = self.test(l_obs, l_res, on_modulate_chan=False)

            win1, win2 = cft.compare_result(l_of_res[self.name], l_of_res_new[self.name], l_of_res['gnd_truth'], True)
            if win1 > win2:
                #update l_of_res in case the for loop are not in the else
                l_of_res = l_of_res_new
                print "better with training --------"
            else:
                self.koho = save_koho
                print "worst with training"

                self.koho[1].alpha = 0.1
                self.koho[0].alpha = 0.1
                walk_get = np.nonzero(l_of_res[self.name+'_raw'])[0]
                walk_expected = np.nonzero(l_of_res['gnd_truth'])[0]
                for i in range(14):
                    #when algo say walk we try to exclude the obs from walk network and include it in rest network
                    save_koho = copy.copy(self.koho)

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
                        l_of_res = l_of_res_new
                        print "better ---"
                    else:
                        self.koho = save_koho
                        print "worst"

    def train_on_files(self, paths, cft, is_healthy=False, new_day=True, train_mod_chan=True):
        all_obs = []
        all_res = []
        for filename in paths:
            l_res, l_obs = cft.convert_one_cpp_file(filename, is_healthy=is_healthy, cut_after_cue=True, init_in_walk=False)
            all_obs += l_obs
            all_res += l_res

        if new_day:
            success, l_of_res = self.test(all_obs, all_res)
            l_obs_koho = cft.obs_classify_mixed_res(all_obs, all_res, l_of_res[self.name], 0)
            self.simulated_annealing(all_obs, l_obs_koho, all_res, 0.1, 14, 0.99)

        self.train_nets(self, all_obs, all_res, cft, with_RL=True, obs_to_add=0)

        if train_mod_chan:
            self.mod_chan = cft.get_mod_chan(all_obs)


class cpp_file_tools:
    def __init__(self, chan_count, group_chan, ext_img='.png', save=False, show=False):
         #group channel by
        self.group_chan = group_chan
        #result for the state
        self.stop = [1, 0]
        self.walk = [0, 1]
        self.ext_img = ext_img
        self.save = save
        self.show = show
        #params for file converter
        self.first_chan = 7
        self.chan_count = chan_count

    def convert_one_cpp_file(self, filename, is_healthy=False, cut_after_cue=False, init_in_walk=True):
        l_obs = []
        l_res = []
        #read 'howto file reading.txt' to understand
        if is_healthy:
            #col 4
            stop = ['1', '-4', '0']
            walk = ['2']
            init = ['-2']
            #when we read contusion file
            if init_in_walk:
                walk += init
            else:
                #when we read SCI file
                stop += init
        else:
            #col 6
            stop = ['0', '3', '4']
            walk = ['1', '2']

        csvfile = open(filename, 'rb')
        file = csv.reader(csvfile, delimiter=' ', quotechar='"')
        prevState=stop[0]
        #grab expected result in file and convert, grab input data
        for row in file:
            if len(row) > self.first_chan and row[0] != '0':
                #if rat is healthy walk state are in col 4 otherwise in col 6 see 'howto file reading file'
                if is_healthy:
                    ratState = row[3]
                else:
                    ratState = row[5]

                #add brain state to l_obs and convert number to float
                brain_state = self.convert_brain_state(row[self.first_chan:self.chan_count+self.first_chan])
                #'-1' added to ignore time where the rat is in the air added by 'add_ground_truth'
                if row[5] != '-1':
                    if row[5] in ['0', '3', '4'] and prevState in walk and cut_after_cue:
                            break

                    if ratState in stop:
                        #we don't take after the cue cause the rat reach the target
                        l_res.append(self.stop)
                        l_obs.append(brain_state)
                    elif ratState in walk:
                        l_res.append(self.walk)
                        l_obs.append(brain_state)

                    if row[5] in ['1', '2']:
                        prevState = walk[0]

        return l_res, l_obs

    def convert_cpp_file(self, dir_name, date, files, is_healthy=False, file_core_name='healthyOutput_', cut_after_cue=False, init_in_walk=True):
        #convert cpp file to list of obs and list of res
        l_obs = []
        l_res = []
        #read 'howto file reading.txt' to understand
        for f in files:
            filename = dir_name+date+file_core_name+str(f)+'.txt'
            l_res_tmp, l_obs_tmp = self.convert_one_cpp_file(filename, is_healthy, cut_after_cue, init_in_walk)
            l_obs += l_obs_tmp
            l_res += l_res_tmp
        return l_res, l_obs

    def convert_brain_state(self, obs):
        #convert what we read in the file to correct brain state
        obs_converted = np.array(range(len(obs)/self.group_chan))
        #convert obs from string to float
        obs_converted = map(float, obs_converted)
        #sum chan X by X (X=self.group_chan)
        res = 0.0
        for i in range(len(obs)):
            if i % self.group_chan == 0:
                obs_converted[i/self.group_chan] = res
                res = 0.0
            res += float(obs[i])
        return np.array(obs_converted)

    def get_mod_chan(self, l_obs):
        #return the chan where a neuron is active (modulated chan)
        l_obs = np.array(l_obs)
        mod_chan = l_obs.sum(0).nonzero()[0]
        return mod_chan

    def obs_classify(self, l_obs, l_res):
        #classify obs using the cue
        l_obs_stop = []
        l_obs_walk = []
        for i in range(len(l_res)):
            if l_res[i] == self.stop:
                l_obs_stop.append(l_obs[i])
            elif l_res[i] == self.walk:
                l_obs_walk.append(l_obs[i])
        return [l_obs_stop, l_obs_walk]

    def obs_classify_good_res(self, l_obs, l_res, l_calc_res, obs_to_add=0):
        #add obs only if the network give the good answer
        l_obs_stop = []
        l_obs_walk = []
        for i in range(1, len(l_res)-1):
            if l_res[i] == self.stop and l_calc_res[i] == self.stop.index(1):
                l_obs_stop.append(l_obs[i])
                #when we change state and this is a good idea brain state before and after should be same state
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.stop, l_obs_stop)

            elif l_res[i] == self.walk and l_calc_res[i] == self.walk.index(1):
                l_obs_walk.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_bad_res(self, l_obs, l_res, l_calc_res, obs_to_add=0):
        #add obs only if the network give the bad answer
        l_obs_stop = []
        l_obs_walk = []
        for i in range(1, len(l_res)-1):
            if l_res[i] == self.stop and l_calc_res[i] == self.walk.index(1):
                l_obs_stop.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.stop, l_obs_stop)
            elif l_res[i] == self.walk and l_calc_res[i] == self.stop.index(1):
                l_obs_walk.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_mixed_res(self, l_obs, l_res, l_calc_res, obs_to_add=0):
        #add obs to stop when no cue and to walk only if the network give the right answer
        l_obs_stop = []
        l_obs_walk = []
        #list_of_res
        #0 = res expected
        #1 = res calculate before HMM
        #2 = res calculate after HMM
        for i in range(1, len(l_res)-1):
            if l_res[i] == self.stop:
                l_obs_stop.append(l_obs[i])

            elif l_res[i] == self.walk and l_calc_res[i] == self.walk.index(1):
                l_obs_walk.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_prev_res(self, l_obs, l_calc_res, obs_to_add=0):
        #we class obs using only the previous result no ground truth involved here
        #we need ground truth to call test
        l_obs_stop = []
        l_obs_walk = []
        #when obs_to add is <0 we remove obs
        obs_to_remove = []
        for i in range(1, len(l_obs)-1):
            if l_calc_res[i] == self.stop.index(1):
                l_obs_stop.append(l_obs[i])
            elif l_calc_res[i] == self.walk.index(1):
                l_obs_walk.append(l_obs[i])
                #when state change
                if l_calc_res[i] != l_calc_res[i+1]:
                    if obs_to_add > 0:
                        for n in range(i-obs_to_add, i):
                            if 0 < n < len(l_obs):
                                l_obs_walk.append(l_obs[n])
                        for n in range(i, i+obs_to_add):
                            if 0 < n < len(l_obs):
                                l_obs_walk.append(l_obs[n])
                    elif obs_to_add < 0:
                        for n in range(i, i+abs(obs_to_add)):
                            if 0 < n < len(l_obs):
                                obs_to_remove.append(l_obs[n])
                        for n in range(i-abs(obs_to_add), i):
                            if 0 < n < len(l_obs):
                                obs_to_remove.append(l_obs[n])
        #remove obs when obs_to_add <0
        if len(obs_to_remove) > 0:
            tmp_l = []
            for obs in l_obs_walk:
                to_add = True
                for obs_r in obs_to_remove:
                    if (obs_r == obs).all():
                        to_add = False
                        break
                if to_add:
                    tmp_l.append(obs)
            l_obs_walk = tmp_l
        return [l_obs_stop, l_obs_walk]

    def obs_classify_kohonen(self, l_obs, acceptance_factor=0.0):
        print '###### classify with kohonen ######'
        while True:
            #while the network don't give 2 classes
            n = 0
            while True:
                net = kn.Kohonen(12, 7, l_obs[0].shape[0], 5, 0.1, 3, 2, self.ext_img, False, False)

                for i in range(10):
                    net.algo_kohonen(l_obs, False)

                #create two group of neurons
                net.evaluate_neurons(l_obs)
                net.group_neuron_into_x_class(2)
                n+=1
                if len(net.groups) == 2:
                    break
                elif n > 4:
                    #when we still don't have a valid number of class after many trials we raise an exception
                    raise Exception("error the network can't converge for that number of class")
                else:
                    print len(net.groups), len(net.good_neurons)

            #test the networks to know which group is stop and which is walk
            dict_res = {}
            for gp in net.groups:
                dict_res[gp.number] = []

            for obs in l_obs:
                gp = net.find_best_group(obs)
                dict_res[gp.number].append(obs)

            #stop have more observation than walk
            keys = dict_res.keys()
            print keys
            if len(keys) == 2:
                if len(dict_res[keys[0]]) > len(dict_res[keys[1]]):
                    stop = keys[0]
                    walk = keys[1]
                else:
                    stop = keys[1]
                    walk = keys[0]

                l_obs_koho = [dict_res[stop], dict_res[walk]]
                nb_stop = len(dict_res[stop])
                nb_walk = len(dict_res[walk])
                print 'nb stop', nb_stop, 'nb_walk', nb_walk, nb_walk/float(nb_stop)
                if acceptance_factor > 0 and (acceptance_factor < nb_walk/float(nb_stop) < 1.5) or (nb_walk + nb_stop < 150 and nb_walk > 20):
                    return l_obs_koho
                elif acceptance_factor == 0:
                    return l_obs_koho
            else:
                return [[], []]

    @staticmethod
    def add_extra_obs(l_obs, l_res, obs_to_add, calculate_res, i, res_expected, l_obs_state):
        #when the brain state change we add value before or after the observed state
        obs_to_remove=[]
        if 1 < i < len(l_res)-1:
            if calculate_res[i-1] != calculate_res[i]:
                if obs_to_add > 0:
                    for n in range(i-obs_to_add, i):
                        if 0 < n < len(l_res) and l_res[n] == res_expected:
                            l_obs_state.append(l_obs[n])
                    for n in range(i, i+obs_to_add):
                        if 0 < n < len(l_res) and l_res[n] == res_expected:
                            l_obs_state.append(l_obs[n])
                elif obs_to_add < 0:
                    for n in range(i-abs(obs_to_add), i):
                        if 0 < n < len(l_res):
                            obs_to_remove.append(l_obs[n])
                    for n in range(i, i+abs(obs_to_add)):
                        if 0 < n < len(l_res):
                            obs_to_remove.append(l_obs[n])
        if len(obs_to_remove) > 0:
            tmp_l = []
            for obs in l_obs_state:
                to_add = True
                for obs_r in obs_to_remove:
                    if (obs_r == obs).all():
                        to_add = False
                        break
                if to_add:
                    tmp_l.append(obs)
            l_obs_state = tmp_l

    #classify the given result
    @staticmethod
    def class_result(l_res, l_expected_res):
        walk_before_cue = []
        walk_after_cue = []
        current_walk = 0
        for i in range(len(l_res)):
            #when we are at the end of the walk or cue change and we walk
            if (l_res[i] != l_res[i-1] or l_expected_res[i] != l_expected_res[i-1]) and l_res[i] == 0:
                if l_expected_res[i-1] == 0:
                    walk_before_cue.append(current_walk)
                else:
                    walk_after_cue.append(current_walk)
                current_walk = 0

            if l_res[i] == 1:
               current_walk += 1

        return np.array(walk_before_cue), np.array(walk_after_cue)

    #classify the given result
    def compare_result(self, l_res1, l_res2, l_expected_res, no_perfect=False):
        w_before_cue1, w_after_cue1 = self.class_result(l_res1, l_expected_res)
        w_before_cue2, w_after_cue2 = self.class_result(l_res2, l_expected_res)
        block_length = 0.1
        min_walk = 3/block_length
        long_walk = 1/block_length
        short_walk = 0.2/block_length
        win_point1 = 0
        win_point2 = 0
        success_rate1 = 0
        success_rate2 = 0

        all_w1 = np.hstack((w_before_cue1, w_after_cue1))
        all_w2 = np.hstack((w_before_cue2, w_after_cue2))

        long_w1 = w_after_cue1[w_after_cue1 > long_walk]
        long_w2 = w_after_cue2[w_after_cue2 > long_walk]
        short_w1 = w_after_cue1[w_after_cue1 < short_walk]
        short_w2 = w_after_cue2[w_after_cue2 < short_walk]

        #good training have one long walk
        #who has less long walk but at least one
        if 0 < long_w1.shape[0] < long_w2.shape[0]:
            win_point1 += 1
        elif 0 < long_w2.shape[0] < long_w1.shape[0]:
            win_point2 += 1
        elif long_w1.shape[0] < 1 and long_w1.shape[0] < 1:
            win_point1 -= 1
            win_point2 -= 1
        else:
            win_point1 += 1
            win_point2 += 2

        #who has less short walk
        if short_w1.shape[0] < short_w2.shape[0]:
            win_point1 += 1
        elif short_w2.shape[0] < short_w1.shape[0]:
            win_point2 += 1
        else:
            win_point1 += 1
            win_point2 += 1

        #before cue fav short walk
        #init mean cause array.mean() return none if array is empty
        if w_before_cue1.shape[0] > 0:
            wbc1_mean = w_before_cue1.mean()
        else:
            wbc1_mean = 0

        if w_before_cue2.shape[0] > 0:
            wbc2_mean = w_before_cue2.mean()
        else:
            wbc2_mean = 0

        if wbc1_mean < wbc2_mean:
            win_point1 += 1
        elif wbc2_mean < wbc1_mean:
            win_point2 += 1
        else:
            win_point1 += 1
            win_point2 += 1

        #during cue fav long walk
        #init mean cause array.mean() return none if array is empty
        if w_after_cue1.shape[0] > 0:
            wdc1_mean = w_after_cue1.mean()
        else:
            wdc1_mean = 0

        if w_after_cue2.shape[0] > 0:
            wdc2_mean = w_after_cue2.mean()
        else:
            wdc2_mean = 0

        if wdc1_mean > wdc2_mean:
            win_point1 += 1
        elif wdc2_mean > wdc1_mean:
            win_point2 += 1
        else:
            win_point1 += 1
            win_point2 += 1

        #who has the longest walk
        #init max cause array.max() return none if array is empty
        if all_w1.shape[0] > 0:
            all_w1_max = all_w1.max()
        else:
            all_w1_max = 0
        if all_w2.shape[0] > 0:
            all_w2_max = all_w2.max()
        else:
            all_w2_max = 0

        if all_w1_max > all_w2_max:
            win_point1 += 1
        elif all_w2_max > all_w1_max:
            win_point2 += 1
        else:
            win_point1 += 1
            win_point2 += 1

        #less walk time before cue
        if w_before_cue1.sum() < w_before_cue2.sum():
            win_point1 += 1
        elif w_before_cue2.sum() < w_before_cue1.sum():
            win_point2 += 1
        else:
            win_point1 += 1
            win_point2 += 1

        #no walk before cue is good
        if w_before_cue1.shape[0] == 0:
            win_point1 += 1
        if w_before_cue2.shape[0] == 0:
            win_point2 += 1

        #at least min_walk of walk
        if all_w1.sum() > min_walk:
            win_point1 += 1
        if all_w2.sum() > min_walk:
            win_point2 += 1

        if no_perfect:
            return win_point1, win_point2
        else:
            #his this trial perfect (no walk before cue, at least X second of walk)
            if w_before_cue1.shape[0] == 0:
                success_rate1 = min(1, all_w1.sum() / float(min_walk))
            if w_before_cue2.shape[0] == 0:
                success_rate2 = min(1, all_w2.sum() / float(min_walk))

            return win_point1, win_point2, success_rate1, success_rate2

    def success_rate(self, l_res, l_expected_res):
        block_length = 0.1
        min_walk = 3/block_length

        w_before_cue, w_after_cue = self.class_result(l_res, l_expected_res)
        if w_before_cue.shape[0] == 0:
            return min(1, w_after_cue.sum()/float(min_walk))
        else:
            return 0

    def plot_result(self, list_of_res, extra_txt=''):
        plt.figure(figsize=(10, 14))

        cpt = 0
        color=['b', 'r', 'g', 'm', 'c', 'y', 'k']
        for key in list_of_res:
            plt.subplot(len(list_of_res.keys()), 1, cpt)
            plt.plot(list_of_res[key], color[cpt%len(color)], label=key)
            plt.ylabel(key, rotation=0)
            plt.ylim(-0.2, 1.2)
            for i in range(len(list_of_res['gnd_truth'])):
                if list_of_res['gnd_truth'][i-1] != list_of_res['gnd_truth'][i]:
                    plt.vlines(i, -0.2, len(list_of_res)*1.2+0.2, 'b', '--')
            cpt += 1

        plt.tight_layout()

        if self.save:
            plt.savefig('tmp_fig/'+'GMM_vs_kohonen' + extra_txt + self.ext_img, dpi=100)
            plt.savefig('tmp_fig/'+'GMM_vs_kohonen' + extra_txt + '.eps', dpi=100)
        if not self.show:
            plt.close()

    def show_fig(self):
        plt.show()