import matplotlib.pyplot as plt
import brain_state_calculate as bsc
import numpy as np
import random as rnd
import copy
import random as rnd
import pickle

class ChangeObs:
    def __init__(self, l_obs):
        rnd.seed(42)
        #wich col we should move
        self.move_chan = []
        #where we should move the col
        self.move_chan_to = []
        #how much we should modulate the given col
        self.value_modulate = []
        #param for mean modulation
        mu = 0
        sigma = 2

        l_obs = np.array(l_obs)
        #index of modulated channel
        self.mod_chan = l_obs.sum(0).nonzero()[0]
        #number of channel
        self.nbchan = len(l_obs[0])
        #params for number of chan to move
        #35% of modulated chan lost or gain par day with 28% of std
        mean_move = 0.35 * self.mod_chan.shape[0]
        std_move = 0.28 * self.mod_chan.shape[0]
        change_x_chan = 0
        while change_x_chan < 1:
            change_x_chan = self.f2i(rnd.gauss(mean_move, std_move))

        for i in range(change_x_chan):
            self.move_chan.append(self.f2i(rnd.uniform(0, self.mod_chan.shape[0]-1)))
            self.move_chan_to.append(self.f2i(rnd.uniform(0, self.nbchan-1)))

        for i in range(self.nbchan):
            self.value_modulate.append(self.f2i(rnd.gauss(mu, sigma)))
        print self.mod_chan
        print self.move_chan
        print self.move_chan_to
        print self.value_modulate

    def change(self, l_obs):
        l_obs = np.array(l_obs)
        save_obs=copy.copy(l_obs)
        for c in range(l_obs.shape[1]):
            if c in self.mod_chan:
                l_obs[:, c] = l_obs[:, c]+self.value_modulate[c]
            if c in self.move_chan:
                ind = self.move_chan.index(c)
                move_to = self.move_chan_to[ind]
                tmp = copy.copy(l_obs[:, move_to])
                l_obs[:, move_to] = l_obs[:, c]
                l_obs[:, c] = tmp
        #we allow burst count to be negative in order to avoid all value set to zero after X "day"
        #l_obs[l_obs < 0] = 0
        return l_obs

    @staticmethod
    def f2i(number):
        return int(round(number, 0))

    @staticmethod
    def expand_walk(l_res, extend_before, extend_after):
        start_after = []
        for i in range(len(l_res)-1):
            if l_res[i] != l_res[i+1]:
                if l_res[i] == [1, 0]:
                    for n in range(i-extend_before, i+1):
                        if 0 < n < len(l_res):
                            l_res[n] = [0, 1]
                else:
                    start_after.append(i)
        for i in start_after:
            for n in range(i, i+extend_after):
                if 0 < n < len(l_res):
                    l_res[n] = [0, 1]
        return l_res

def obs_to_hist(l_obs):
    l_hist = []
    for obs in l_obs:
        hist, bins = np.histogram(obs)
        l_hist.append(hist)
    return l_hist

#In this script we train the kohonen network using another kohonen network
#learning is totally unsupervised
#####################
######  START  ######
dir_name = '../RT_classifier/BMIOutputs/0423_r600/'
save_obj = False
ext_img = '.png'
save = True
show = False
verbose = True
HMM = True
files0423 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
files = []
for i in range(10):
    files += files0423
#file to use for the network building
first_train = 3
my_bsc = bsc.brain_state_calculate(32, 1, 'koho', ext_img, save, show)
my_bsc.build_networks()

##build one koho network and class obs with unsupervised learning
l_res, l_obs = my_bsc.convert_cpp_file(dir_name, 't_0423', files[0:first_train-1], False,cut_after_cue=False)
l_obs_koho = my_bsc.obs_classify_kohonen(l_obs, 0.0)

#build and train networks
my_bsc.build_networks()
my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.99)

#test network
l_res, l_obs = my_bsc.convert_cpp_file(dir_name, 't_0423', files[first_train-1:first_train], False,cut_after_cue=False)
success, l_of_res = my_bsc.test(l_obs, l_res)
#add gmm result to the plot (use test for healthy data)
#l_res_gmm, l_obs_trash = my_bsc.convert_file(dir_name, 't_0423', files[first_train-1:first_train], True)
l_res_gnd_truth, l_obs_trash = my_bsc.convert_cpp_file(dir_name, 't_0423', files[first_train-1:first_train], False,cut_after_cue=False)
l_of_res.append(np.array(l_res_gnd_truth).argmax(1))
my_bsc.plot_result(l_of_res, '_file_'+str(files[first_train-1:first_train])+'_')

#test all the trial and train the network between each trial
chg_obs = []
change_every = len(files0423)
mod_chan = my_bsc.get_mod_chan(l_obs)
my_bsc.save_networks('', '20140505')
rnd.seed(42)
sr_dict = {'r600': {'0': {'koho_RL': [], 'l_of_res': []}}}
for i in range(first_train, len(files)):
    print '### ### ### ### ### ### ### ### ###'
    print files[i:i+1]
    l_res, l_obs = my_bsc.convert_cpp_file(dir_name, 't_0423', files[i:i+1], False, cut_after_cue=False)

    #change the value
    for chg in chg_obs:
        l_obs = chg.change(l_obs)

    #prepare to change the value
    if i % change_every == 0:
        chg_obs.append(ChangeObs(l_obs))
        l_obs = chg_obs[-1].change(l_obs)
        print 'change obs:'+str(len(chg_obs))
        sr_dict['r600'][str(len(chg_obs))] = {'koho_RL': [], 'l_of_res':[]}

    extend_before = ChangeObs.f2i(rnd.gauss(0.4/0.1, 0.5))
    extend_after = ChangeObs.f2i(rnd.uniform(10, 30))
    l_res = ChangeObs.expand_walk(l_res, extend_before, extend_after)

    #test and plot
    success, l_of_res = my_bsc.test(l_obs, l_res)
    #l_res_gmm, l_obs_trash = my_bsc.convert_file(dir_name, 't_0423', files[i:i+1], True)
    l_res_gnd_truth, l_obs_trash = my_bsc.convert_cpp_file(dir_name, 't_0423', files[i:i+1], False,cut_after_cue=False)
    #l_of_res.append(np.array(l_res_gmm).argmax(1))
    #l_of_res.append((np.array(l_of_res[2])+np.array(l_of_res[3])) > 1)
    l_of_res.append(np.array(l_res_gnd_truth).argmax(1))
    print success
    sr_dict['r600'][str(len(chg_obs))]['l_of_res'].append(l_of_res)
    my_bsc.plot_result(l_of_res, '_file_'+str(len(chg_obs))+'_'+str(files[i:i+1]))

    my_bsc.train_nets(l_obs, l_res, with_RL=False)
    sr_dict['r600'][str(len(chg_obs))]['koho_RL'].append(my_bsc.success_rate(l_of_res[2], l_of_res[-1]))
    my_bsc.get_mod_chan(l_obs)


if show:
    plt.show()

with open('success_rate_simulated_r600_v2', 'wb') as my_file:
    my_pickler = pickle.Pickler(my_file)
    my_pickler.dump(sr_dict)
print('###############')
print('####  END  ####')

#expand walk with random for before cue