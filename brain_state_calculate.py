import kohonen_neuron as kn
import csv
import numpy as np
import copy
import random as rnd
import math
import matplotlib.pyplot as plt
from itertools import combinations
import pickle

class brain_state_calculate:
    def __init__(self, chan_count, group_chan, ext_img='.png', save=False, show=False):
        rnd.seed(42)
        #group channel by
        self.group_chan = group_chan
        #result for the state
        self.stop = [1, 0]
        self.walk = [0, 1]
        self.default_res = [0, 0]
        self.ext_img = ext_img
        self.save = save
        self.show = show
        #params for file converter
        self.first_chan = 7
        self.chan_count = chan_count

        #params for Test
        self.test_all = False
        self.combination_to_test = 50
        self.A = np.array([[0.99, 0.01], [0.01, 0.99]])
        #history length should be a prime number
        self.history_length = 1

        #koho parameters
        self.alpha = 0.01
        self.koho_row = 6
        self.koho_col = 7
        self.koho = []
        #number of neighbor to update in the network
        self.neighbor = 3
        #min winning count to be consider as a good neuron
        self.min_win = 7
        #number of best neurons to keep for calculate distance of obs to the network
        self.dist_count = 5
        self.max_weight = 5
        self.weight_count = self.chan_count/self.group_chan

        #simulated annealing parameters
        #change alpha each X iteration
        self.change_alpha_iteration = 7
        #change alpha by a factor of
        #/!\ should be float
        self.change_alpha_factor = 10.0

    def build_networks(self):
        #build the network
        koho_stop = kn.Kohonen(self.koho_row, self.koho_col, self.weight_count, self.max_weight, self.alpha, self.neighbor, self.min_win, self.ext_img, self.save, self.show, rnd.random())
        koho_walk = kn.Kohonen(self.koho_row, self.koho_col, self.weight_count, self.max_weight, self.alpha, self.neighbor, self.min_win, self.ext_img, self.save, self.show, rnd.random())
        self.koho = [koho_stop, koho_walk]

    def load_networks(self, path):
        print path
        pkl_file = open(path, 'rb')
        self.koho = pickle.load(pkl_file)
        print len(self.koho)

    def init_test(self):
        self.history = np.array([self.stop])
        #matrix which contain the rank of the result
        self.prevP = np.array(self.stop)
        self.HMM=True

    def test_obs(self, obs):
        dist_res = []
        best_ns = []
        res = copy.copy(self.default_res)

        #find the best distance of the obs to each network
        for k in self.koho:
            dist_res.append(k.find_mean_best_dist(obs, self.dist_count))
            #we add extra neurons to best_ns in order to remove null probability
            best_ns.append(k.find_best_X_neurons(obs, self.dist_count+self.dist_count))

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
        return rank

    def convert_file(self, dir_name, date, files, is_healthy=False, file_core_name='healthyOutput_'):
        l_obs = []
        l_res = []
        #read 'howto file reading.txt' to understand
        if is_healthy:
            #col 4
            stop = ['1', '-4', '0']
            walk = ['2', '-2']
        else:
            #col 6
            stop = ['0', '3', '4']
            walk = ['1', '2']

        for f in files:
            filename = dir_name+date+file_core_name+str(f)+'.txt'
            csvfile = open(filename, 'rb')
            file = csv.reader(csvfile, delimiter=' ', quotechar='"')
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
                        if ratState in stop:
                            l_res.append(self.stop)
                            l_obs.append(brain_state)
                        elif ratState in walk:
                            l_res.append(self.walk)
                            l_obs.append(brain_state)

            # derivative of l_obs
            # l_obs_d = []
            # l_obs_d.append([])
            # for i in range(1, len(l_obs)):
            #     l_obs_d.append(np.array(l_obs[i])-np.array(l_obs[i-1]))
        return l_res, l_obs

    def convert_brain_state(self, obs):
        #convert what we read in the file to correct brain state
        obs_converted = []
        #convert obs from string to float
        obs = map(float, obs)
        #sum chan X by X (X=self.group_chan)
        res = 0
        for i in range(len(obs)):
            if i % self.group_chan == 0:
                obs_converted.append(res)
                res = 0
            res += obs[i]
        return obs_converted

    def obs_classify(self, l_obs, l_res):
        l_obs_stop = []
        l_obs_walk = []
        for i in range(len(l_res)):
            if l_res[i] == self.stop:
                l_obs_stop.append(l_obs[i])
            elif l_res[i] == self.walk:
                l_obs_walk.append(l_obs[i])
        return [l_obs_stop, l_obs_walk]

    def obs_classify_good_res(self, l_obs, l_res, obs_to_add):
        #add obs only if the network give the good answer
        l_obs_stop = []
        l_obs_walk = []
        success, list_of_res = self.test(l_obs, l_res, True)
        for i in range(1, len(l_res)-1):
            if l_res[i] == self.stop and list_of_res[2][i] == self.stop.index(1):
                l_obs_stop.append(l_obs[i])
                #when we change state and this is a good idea brain state before and after should be same state
                self.add_extra_obs(l_obs, l_res, obs_to_add, list_of_res, i, self.stop, l_obs_stop)

            elif l_res[i] == self.walk and list_of_res[2][i] == self.walk.index(1):
                l_obs_walk.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, list_of_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_bad_res(self, l_obs, l_res, obs_to_add):
        #add obs only if the network give the bad answer
        l_obs_stop = []
        l_obs_walk = []
        success, list_of_res = self.test(l_obs, l_res, True)
        for i in range(1, len(l_res)-1):
            if l_res[i] == self.stop and list_of_res[2][i] == self.walk.index(1):
                l_obs_stop.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, list_of_res, i, self.stop, l_obs_stop)
            elif l_res[i] == self.walk and list_of_res[2][i] == self.stop.index(1):
                l_obs_walk.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, list_of_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_mixed_res(self, l_obs, l_res, obs_to_add):
        #add obs to stop when no cue and to walk only if the network give the right answer
        l_obs_stop = []
        l_obs_walk = []
        success, list_of_res = self.test(l_obs, l_res, True)
        #list_of_res
        #0 = res expected
        #1 = res calculate before HMM
        #2 = res calculate after HMM
        for i in range(1, len(l_res)-1):
            if l_res[i] == self.stop:
                l_obs_stop.append(l_obs[i])

            elif l_res[i] == self.walk and list_of_res[1][i] == self.walk.index(1):
                l_obs_walk.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, list_of_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_prev_res(self, l_obs, l_res, obs_to_add):
        #we class obs using only the previous result no ground truth involved here
        #we need ground truth to call test
        l_obs_stop = []
        l_obs_walk = []
        success, list_of_res = self.test(l_obs, l_res, True)
        #list_of_res
        #0 = res expected
        #1 = res calculate before HMM
        #2 = res calculate after HMM
        for i in range(1, len(l_obs)-1):
            if list_of_res[2][i] == self.stop.index(1):
                l_obs_stop.append(l_obs[i])
            elif list_of_res[2][i] == self.walk.index(1):
                l_obs_walk.append(l_obs[i])
                if list_of_res[2][i-1] != list_of_res[2][i]:
                    for n in range(i-obs_to_add, i):
                        if n > 0:
                            l_obs_walk.append(l_obs[n])
                if list_of_res[2][i] != list_of_res[2][i+1]:
                    for n in range(i, i+obs_to_add):
                        if n > 0:
                            l_obs_walk.append(l_obs[n])

        return [l_obs_stop, l_obs_walk]

    @staticmethod
    def obs_classify_kohonen(l_obs, acceptance_factor=0.3):
        print '###### classify with kohonen ######'
        while True:
            #while the network don't give 2 classes
            n = 0
            while True:
                net = kn.Kohonen(20, 20, 32, 5, 0.1, 3, 2, '.png', False, False)
                for i in range(10):
                    net.algo_kohonen(l_obs, False)

                #create two group of neurons
                net.evaluate_neurons(l_obs)
                net.group_neuron_into_x_class(2)
                n+=1
                if len(net.groups) == 2:
                    break
                elif n > 4:
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
                print nb_stop, nb_walk, nb_walk/float(nb_stop)
                if (acceptance_factor < nb_walk/float(nb_stop) < 1.5) or (nb_walk + nb_stop < 150 and nb_walk > 20):
                    return l_obs_koho
            else:
                return [[], []]

    #TODO handle empty chan_mod with kohonen classify
    def obs_classify_mod_chan(self, l_obs, l_res, chan_mod):
        l_obs = np.array(l_obs)
        #keep only chan that where modulated
        for c in range(l_obs.shape[1]):
            if c not in chan_mod:
                l_obs[:, c] = 0

        success, l_of_res = self.test(l_obs, l_res, True)
        obs_stop = []
        obs_walk = []
        for i in range(l_obs.shape[0]):
            if l_res[i] == [1, 0]:
                obs_stop.append(l_obs[i, :])
            elif l_of_res[1][i] == 1 and l_res[i] == [0, 1]:
                obs_walk.append(l_obs[i, :])
        return [obs_stop, obs_walk]

    @staticmethod
    def get_mod_chan(l_obs):
        l_obs = np.array(l_obs)
        return l_obs.sum(0).nonzero()[0]

    @staticmethod
    def add_extra_obs(l_obs, l_res, obs_to_add, list_of_res, i, res_expected, l_obs_state):
        #when the brain state change we add value before or after to the observed state
        if list_of_res[2][i-1] != list_of_res[2][i]:
            for n in range(i-obs_to_add, i):
                if 0 < n < len(l_res) and l_res[n] == res_expected:
                    l_obs_state.append(l_obs[n])
        elif list_of_res[2][i+1] != list_of_res[2][i]:
            for n in range(i, i+obs_to_add):
                if 0 < n < len(l_res) and l_res[n] == res_expected:
                    l_obs_state.append(l_obs[n])

    def compute_network_accuracy(self, best_ns, dist_res, obs):
        #we test combination of each best n
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

    def test(self, l_obs, l_res, HMM=False, verbose=False):
        good = 0

        history = np.array([self.stop])
        #matrix which contain the rank of the result
        results = []
        raw_res = []
        prevP = np.array(self.stop)

        prob_res = copy.copy(self.default_res)
        for i in range(len(l_obs)):
            dist_res = []
            best_ns = []
            res = copy.copy(self.default_res)

            #find the best distance of the obs to each network
            for k in self.koho:
                dist_res.append(k.find_mean_best_dist(l_obs[i], self.dist_count))
                #we add extra neurons to best_ns in order to remove null probability
                best_ns.append(k.find_best_X_neurons(l_obs[i], self.dist_count+self.dist_count))
            raw_res.append(np.array(dist_res).argmin())

            if HMM:
                #flatten list
                best_ns = [item for sublist in best_ns for item in sublist]

                prob_res = self.compute_network_accuracy(best_ns, dist_res, l_obs[i])

                #compute result with HMM
                P = []
                for k in range(prob_res.shape[0]):
                    P.append(prevP.T.dot(self.A[:, k])*prob_res[k])
                #repair sum(P) == 1
                P = np.array(P).T/sum(P)

                #transform in readable result
                rank = P.argmax()
                res[rank] = 1

                #save P.T
                prevP = copy.copy(P.T)
                #prevP = np.array(copy.copy(res))
            else:
                #transform in readable result
                rank = np.array(dist_res).argmin()
                res[rank] = 1

            #use history to smooth change
            history = np.vstack((history, res))
            if history.shape[0] > self.history_length:
                history = history[1:, :]

            res = copy.copy(self.default_res)
            #transform in readable result
            rank = history.mean(0).argmax()
            res[rank] = 1
            results.append(res)
            if res == l_res[i]:
                good += 1
            if verbose:
                print(np.array(res).argmax(), np.array(l_res[i]).argmax(), np.array(prob_res).argmax(), prob_res, prevP)

        if len(l_obs) > 0:
            if verbose:
                print good/float(len(l_obs))
            return good/float(len(l_obs)), [np.array(l_res).argmax(1), np.array(raw_res), np.array(results).argmax(1)]
        else:
            print ('l_obs is empty')
            return 0, []

    def plot_result(self, list_of_res, extra_txt=''):
        plt.figure()
        for i in range(len(list_of_res)):
            plt.plot(list_of_res[i]+i*1.2)
        plt.ylim(-0.2, len(list_of_res)*1.2+0.2)
        cue_start = np.array(list_of_res[0]).argmax()
        plt.vlines(cue_start, -0.2, len(list_of_res)*1.2+0.2, 'b', '--')
        if self.save:
            plt.savefig('tmp_fig/'+'GMM_vs_kohonen' + extra_txt + self.ext_img, bbox_inches='tight')
        if not self.show:
            plt.close()

    def simulated_annealing(self, l_obs, l_obs_koho, l_res, alpha_start, max_iteration, max_success, verbose=False):
        #inspired from simulated annealing, to determine when we should stop learning
        #initialize
        success, lor_trash = self.test(l_obs, l_res)
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
            #compute success of the networks
            success_cp, lor_trash = self.test(l_obs, l_res)

            if verbose:
                print '---'
                print n
                print alpha
                print success_cp * 100
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

    @staticmethod
    def random_combination(iterable, r):
        #"Random selection from itertools.combinations(iterable, r)"
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(rnd.sample(xrange(n), r))
        return tuple(pool[i] for i in indices)

    def save_networks(self, dir_name, date):
        #save networks
        with open(dir_name + 'koho_networks_' + date, 'wb') as my_file:
            my_pickler = pickle.Pickler(my_file)
            my_pickler.dump(self.koho)

    #classify the given result
    @staticmethod
    def class_result(l_res, l_expected_res):
        block_length = 0.1
        long_walk = 2/block_length
        short_walk = 2
        long_walk_count = 0
        short_walk_count = 0
        longest_walk = 0
        current_walk = 0
        current_stop = 0
        #give the percentage of walk before cue
        walk_before_cue_count = 0
        total_walk = 0
        walk_after_cue_count = 0
        walk_expected = 0

        for i in range(len(l_res)):
            if l_expected_res[i] == 1:
                walk_expected += 1
                if l_res[i] == 1:
                    walk_after_cue_count += 1

            if l_res[i] == 1 and l_expected_res[i] == 0:
                walk_before_cue_count += 1

            if l_expected_res[i-1] == 0 and l_expected_res[i] == 1:
                walk_before_cue_count /= float(i)

            if l_res[i] != l_res[i-1]:
                if current_walk > longest_walk:
                    longest_walk = current_walk
                if current_walk > long_walk:
                    long_walk_count += 1
                if 0 < current_walk < short_walk:
                    short_walk_count += 1
                if current_walk > 0:
                    total_walk += current_walk
                current_walk = 0
                current_stop = 0
            else:
                if l_res[i] == 0:
                    current_stop += 1
                else:
                    current_walk += 1
        walk_after_cue_count /= float(walk_expected)
        training = []
        #perfect trial no walk before cue and at least 2 second of walk
        if walk_before_cue_count == 0 and total_walk > 2/block_length and long_walk_count > 0:
            training.append("train_prev_res")
        #collapse only walk or only rest
        if total_walk > len(l_res)-2 or total_walk < 2:
            training.append("train_kohonen")
        #to many walk before cue
        if walk_before_cue_count > 0.0:
            training.append("train_stop")
        #not enough walk after cue
        if walk_after_cue_count < 0.2:
            training.append("train_walk")
        #to many short walk
        if short_walk_count > 7:
            training.append("train_mixed_res")
        return training


#rules
# if we have too many walk before cue -> train only stop, make walk away
# if we have not enough walk during cue -> add extra result to walk and train with mixed_res
#                                       -> new network
# if we have too many short walk
# if we have perfect Trial train with good or don't touch network
# if we have nearly perfect trial train on mistake
# if collapse -> kohonen learning
# if only walk -> train stop
# if only rest -> train walk

# when new day migrate all the network
#give percentage of confidence for both state
#loop until we have a success trial
#use random seed

#rebuild network
#migrate network use neighbor length of network and algo koho with no decrease neighbor
#koho[].neighbor=max([self.koho_col,self.koho_row])
#10 times do -> koho[].algo_kohonen(l_obs,False)
#train with all file in
#train with this file

#when test only use chan previously modulate set other to 0