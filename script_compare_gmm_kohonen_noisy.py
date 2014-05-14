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

#classify the given result
def compare_result(l_res1, l_res2, l_expected_res, no_perfect=False):
    w_before_cue1, w_after_cue1 = class_result(l_res1, l_expected_res)
    w_before_cue2, w_after_cue2 = class_result(l_res2, l_expected_res)
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

def success_rate(l_res, l_expected_res):
    block_length = 0.1
    min_walk = 3/block_length

    w_before_cue, w_after_cue = class_result(l_res, l_expected_res)
    if w_before_cue.shape[0] == 0:
        return min(1, w_after_cue.sum()/float(min_walk))
    else:
        return 0

#classify the given result
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


def train(l_obs, l_res, my_bsc):
    HMM = True
    verbose = False
    save_koho = copy.copy(my_bsc.koho)
    success, l_of_res = my_bsc.test(l_obs, l_res, HMM, verbose)

    l_obs_koho = my_bsc.obs_classify_mixed_res(l_obs, l_res, 0)
    success, l_of_res_new = my_bsc.test(l_obs, l_res, HMM, verbose)
    my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99, verbose)

    win1, win2 = compare_result(l_of_res[2], l_of_res_new[2], l_of_res[0])
    if win1 > win2:
        l_of_res = l_of_res_new
        print "better with training --------"
    else:
        my_bsc.koho = save_koho
        print "worst with training"

    my_bsc.koho[1].alpha = 0.1

    for i in range(l_of_res[2].shape[0]):
        #try to push away each obs where walk decoded
        if l_of_res[1][i] == 1:
            save_koho = copy.copy(my_bsc.koho)
            my_bsc.koho[0].update_closest_neurons(l_obs[i], push_away=False)
            my_bsc.koho[1].update_closest_neurons(l_obs[i], push_away=True)

            success, l_of_res_new = my_bsc.test(l_obs, l_res, HMM, verbose)
            win1, win2 = compare_result(l_of_res[2], l_of_res_new[2], l_of_res[0])
            if win1 > win2:
                l_of_res = l_of_res_new
                print "better with go away ---"
            else:
                print "worst with go away"
                my_bsc.koho = save_koho
        #try to bring closer each obs where no walk decoded and walk cue
        elif l_of_res[1][i] == 0 and l_of_res[0][i] == 1:
            save_koho = copy.copy(my_bsc.koho)
            my_bsc.koho[0].update_closest_neurons(l_obs[i], push_away=True)
            my_bsc.koho[1].update_closest_neurons(l_obs[i], push_away=False)
            success, l_of_res_new = my_bsc.test(l_obs, l_res, HMM, verbose)
            win1, win2 = compare_result(l_of_res[2], l_of_res_new[2], l_of_res[0])
            if win1 > win2:
                print "better with come here --"
                l_of_res = l_of_res_new
            else:
                print "worst with come here"
                my_bsc.koho = save_koho


    return my_bsc

def train_nets(l_obs, l_obs_mod, l_res, my_bsc, mod_chan):
    #we use l_obs_mod only to classify result
    HMM = True
    verbose = False
    save_koho = copy.copy(my_bsc.koho)
    success, l_of_res = my_bsc.test(l_obs_mod, l_res, HMM, verbose)

    l_obs_koho = my_bsc.obs_classify_mod_chan(l_obs, l_res, mod_chan, 0)
    my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99, verbose)
    success, l_of_res_new = my_bsc.test(l_obs_mod, l_res, HMM, verbose)

    win1, win2 = compare_result(l_of_res[2], l_of_res_new[2], l_of_res[0], True)
    if win1 > win2:
        l_of_res = l_of_res_new
        print "better with training --------"
    else:
        my_bsc.koho = save_koho
        print "worst with training"

        my_bsc.koho[1].alpha = 0.1
        my_bsc.koho[0].alpha = 0.1
        walk_get = np.nonzero(l_of_res[1])[0]
        walk_expected = np.nonzero(l_of_res[0])[0]
        if walk_get.shape[0] == 0:
            my_bsc.was_bad += 1
            if my_bsc.was_bad > 2:
                l_obs_koho = my_bsc.obs_classify_kohonen(l_obs)
                my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99, verbose)
                success, l_of_res_new = my_bsc.test(l_obs_mod, l_res, HMM, verbose)
        else:
            my_bsc.was_bad = 0
            for i in range(40):
                #when we say walk
                obs_ind = walk_get[rnd.randrange(walk_get.shape[0])]
                save_koho = copy.copy(my_bsc.koho)
                my_bsc.koho[0].update_closest_neurons(l_obs[obs_ind], push_away=False)
                my_bsc.koho[1].update_closest_neurons(l_obs[obs_ind], push_away=True)

                #when we want walk
                obs_ind = walk_expected[rnd.randrange(walk_expected.shape[0])]
                my_bsc.koho[0].update_closest_neurons(l_obs[obs_ind], push_away=True)
                my_bsc.koho[1].update_closest_neurons(l_obs[obs_ind], push_away=False)

                success, l_of_res_new = my_bsc.test(l_obs_mod, l_res, False, verbose)
                win1, win2 = compare_result(l_of_res[1], l_of_res_new[1], l_of_res[0], True)
                if win1 > win2:
                    l_of_res = l_of_res_new
                    print "better ---"
                else:
                    my_bsc.koho = save_koho
                    print "worst"
    return my_bsc

def get_only_mod_chan(l_obs, mod_chan):
    l_obs_mod = copy.copy(np.array(l_obs))
    #keep only chan that where modulated
    for c in range(l_obs_mod.shape[1]):
        if c not in mod_chan:
            l_obs_mod[:, c] = 0
    return l_obs_mod

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
my_bsc = bsc.brain_state_calculate(32, 1, ext_img, save, show)
my_bsc.build_networks()

##build one koho network and class obs with unsupervised learning
l_res, l_obs = my_bsc.convert_cpp_file(dir_name, 't_0423', files[0:first_train-1], False,cut_after_cue=False)
l_obs_koho = my_bsc.obs_classify_kohonen(l_obs, 0.0)

#build and train networks
my_bsc.build_networks()
my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.99, verbose)

#test network
l_res, l_obs = my_bsc.convert_cpp_file(dir_name, 't_0423', files[first_train-1:first_train], False,cut_after_cue=False)
success, l_of_res = my_bsc.test(l_obs, l_res, HMM, verbose)
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
sr_dict = {'r600': {'0': {'koho_RL': []}}}
for i in range(first_train, len(files)):
    print '### ### ### ### ### ### ### ### ###'
    print files[i:i+1]
    l_res, l_obs = my_bsc.convert_cpp_file(dir_name, 't_0423', files[i:i+1], False,cut_after_cue=False)

    #change the value
    for chg in chg_obs:
        l_obs = chg.change(l_obs)

    #prepare to change the value
    if i % change_every == 0:
        chg_obs.append(ChangeObs(l_obs))
        l_obs = chg_obs[-1].change(l_obs)
        print 'change obs:'+str(len(chg_obs))
        sr_dict['r600'][str(len(chg_obs))] = {'koho_RL': []}

    extend_before = ChangeObs.f2i(rnd.gauss(0.4/0.1, 0.5))
    extend_after = ChangeObs.f2i(rnd.uniform(10, 30))
    l_res = ChangeObs.expand_walk(l_res, extend_before, extend_after)

    l_obs_mod = get_only_mod_chan(l_obs, mod_chan)

    #test and plot
    success, l_of_res = my_bsc.test(l_obs_mod, l_res, HMM)
    #l_res_gmm, l_obs_trash = my_bsc.convert_file(dir_name, 't_0423', files[i:i+1], True)
    l_res_gnd_truth, l_obs_trash = my_bsc.convert_cpp_file(dir_name, 't_0423', files[i:i+1], False,cut_after_cue=False)
    #l_of_res.append(np.array(l_res_gmm).argmax(1))
    #l_of_res.append((np.array(l_of_res[2])+np.array(l_of_res[3])) > 1)
    l_of_res.append(np.array(l_res_gnd_truth).argmax(1))
    print success
    my_bsc.plot_result(l_of_res, '_file_'+str(len(chg_obs))+'_'+str(files[i:i+1]))

    #when we modulate chan means new day so more learning
    # if i % change_every == 0:
    #     # l_obs_koho = my_bsc.obs_classify_mixed_res(l_obs, l_res, 0)
    #     l_obs_koho = my_bsc.obs_classify_mod_chan(l_obs, l_res, mod_chan)
    #     my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99, verbose)

    my_bsc = train_nets(l_obs, l_obs_mod, l_res, my_bsc, mod_chan)
    sr_dict['r600'][str(len(chg_obs))]['koho_RL'].append(success_rate(l_of_res[2], l_of_res[-1]))
    #my_bsc = train(l_obs, l_res, my_bsc)
    #train networks using previous result
    #l_obs_koho = my_bsc.obs_classify_mixed_res(l_obs, l_res, 0)
    #l_obs_koho = my_bsc.obs_classify_mod_chan(l_obs, l_res, mod_chan)
    #my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.01, 7, 0.99, verbose)
    # my_bsc.koho[0].plot_network('_'+str(len(chg_obs))+'_'+str(files[i:i+1])+'_rest')
    # my_bsc.koho[1].plot_network('_'+str(len(chg_obs))+'_'+str(files[i:i+1])+'_walk')
    # my_bsc.koho[0].plot_network_dist('_'+str(len(chg_obs))+'_'+str(files[i:i+1])+'_rest')
    # my_bsc.koho[1].plot_network_dist('_'+str(len(chg_obs))+'_'+str(files[i:i+1])+'_walk')
    # plt.figure()
    # plt.imshow(np.vstack((np.array(l_obs).T, np.array(l_res)[:,1]*5)),interpolation='none')
    # plt.colorbar()
    # plt.savefig('tmp_fig/obs_'+'_file_'+str(len(chg_obs))+'_'+str(files[i:i+1])+ext_img)
    mod_chan = my_bsc.get_mod_chan(l_obs)


if show:
    plt.show()

with open('success_rate_simulated_r600', 'wb') as my_file:
    my_pickler = pickle.Pickler(my_file)
    my_pickler.dump(sr_dict)
print('###############')
print('####  END  ####')

#expand walk with random for before cue