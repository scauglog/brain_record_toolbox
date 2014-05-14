import brain_state_calculate as bsc
from collections import OrderedDict
from collections import Counter
import numpy as np
import copy
import random as rnd
import signal_processing as sp
import pickle

#we retrain the network only if networks is too different
def train(files, rat, date, my_bsc, min_obs, min_obs_train, obs_to_add, start, mod_chan):
    koho_win = 0
    GMM_win = 0
    koho_pts = 0
    GMM_pts = 0
    koho_perfect = 0
    GMM_perfect = 0
    for n in range(start, len(files[rat][date])-1):
        l_res, l_obs = my_bsc.convert_cpp_file(dir_name, '12'+date, files[rat][date][n:n+1], False, 'SCIOutput_')
        if len(l_obs) > min_obs and np.array(l_obs).sum() > 0:
            success, l_of_res = my_bsc.test(l_obs, l_res, HMM, verbose)
            l_res_GMM, l_obs = my_bsc.convert_cpp_file(dir_name, '12'+date, files[rat][date][n:n+1], True, 'SCIOutput_')
            l_of_res.append(np.array(l_res_GMM).argmax(1))
            l_of_res.append((np.array(l_of_res[2])+np.array(l_of_res[3])) > 1)
            my_bsc.plot_result(l_of_res, rat+'_'+date+'_'+str(n))

            l_obs_koho = my_bsc.obs_classify_mixed_res(l_obs, l_res, obs_to_add)
            dict_count = Counter(l_of_res[1])
            #we rely on raw res (before HMM) to define if we should reclass the network
            if dict_count[0] < min_obs_train or dict_count[1] < min_obs_train:
                print('retrain network')
                # l_obs_koho = my_bsc.obs_classify_kohonen(l_obs)
                l_obs_koho = my_bsc.obs_classify(l_obs, l_res)
                my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 1, True)

            #train networks
            my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.7, True)
            # my_bsc.koho[0].alpha = 0.01
            # for k in range(10):
            #     my_bsc.koho[0].algo_kohonen(l_obs_koho[0])

            tmp_koho_pts, tmp_GMM_pts, tmp_perfect_koho, tmp_perfect_GMM = compare_result(l_of_res[2], l_of_res[3], l_of_res[0])
            koho_pts += tmp_koho_pts
            GMM_pts += tmp_GMM_pts
            koho_perfect += tmp_perfect_koho
            GMM_perfect += tmp_perfect_GMM
            if tmp_koho_pts > tmp_GMM_pts:
                koho_win += 1
            elif tmp_GMM_pts > tmp_koho_pts:
                GMM_win += 1
            else:
                koho_win += 1
                GMM_win += 1
    return koho_win, koho_pts, koho_perfect, GMM_win, GMM_pts, GMM_perfect

#we retrain the network each day
def train2(files, rat, date, my_bsc, min_obs, min_obs_train, obs_to_add, start, mod_chan):
    new_date = True
    koho_win = 0
    GMM_win = 0
    koho_pts = 0
    GMM_pts = 0
    koho_perfect = 0
    GMM_perfect = 0
    for n in range(start, len(files[rat][date])-1):
        #get obs
        l_res, l_obs = my_bsc.convert_cpp_file(dir_name, '12'+date, files[rat][date][n:n+1], False, 'SCIOutput_')
        if len(l_obs) > min_obs and np.array(l_obs).sum() > 0:
            #get res
            success, l_of_res = my_bsc.test(l_obs, l_res, HMM, verbose)
            l_res_GMM, l_obs = my_bsc.convert_cpp_file(dir_name, '12'+date, files[rat][date][n:n+1], True, 'SCIOutput_')
            l_of_res.append(np.array(l_res_GMM).argmax(1))
            l_of_res.append((np.array(l_of_res[2])+np.array(l_of_res[3])) > 1)
            my_bsc.plot_result(l_of_res, rat+'_'+date+'_'+str(n))

            if new_date and len(l_obs) > min_obs:
                new_date = False
                dict_count = Counter(l_of_res[1])
                #we rely on raw res (before HMM) to define if we should reclass the network
                if dict_count[0] < min_obs_train or dict_count[1] < min_obs_train:
                    print('retrain network')
                    #l_obs_koho = my_bsc.obs_classify_kohonen(l_obs)
                    l_obs_koho = my_bsc.obs_classify(l_obs, l_res)
                else:
                    l_obs_koho = my_bsc.obs_classify_mixed_res(l_obs, l_res, obs_to_add)
                l_obs_koho = my_bsc.obs_classify_mod_chan(l_obs, l_res, mod_chan)
                mod_chan = my_bsc.get_mod_chan(l_obs)
                my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.9, True)

            l_obs_koho = my_bsc.obs_classify_mixed_res(l_obs, l_res, obs_to_add)
            #train networks
            my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 7, 0.9, True)
            # my_bsc.koho[0].alpha = 0.01
            # for k in range(10):
            #     my_bsc.koho[0].algo_kohonen(l_obs_koho[0])

            tmp_koho_pts, tmp_GMM_pts, tmp_perfect_koho, tmp_perfect_GMM = compare_result(l_of_res[2], l_of_res[3], l_of_res[0])
            koho_pts += tmp_koho_pts
            GMM_pts += tmp_GMM_pts
            koho_perfect += tmp_perfect_koho
            GMM_perfect += tmp_perfect_GMM
            if tmp_koho_pts > tmp_GMM_pts:
                koho_win += 1
            elif tmp_GMM_pts > tmp_koho_pts:
                GMM_win += 1
            else:
                koho_win += 1
                GMM_win += 1
    return koho_win, koho_pts, koho_perfect, GMM_win, GMM_pts, GMM_perfect

def get_only_mod_chan(l_obs, mod_chan):
    l_obs_mod = copy.copy(np.array(l_obs))
    #keep only chan that where modulated
    for c in range(l_obs_mod.shape[1]):
        if c not in mod_chan:
            l_obs_mod[:, c] = 0
    return l_obs_mod

#train the network each day with mixed res and the new day with mod_chan
def train3(files, rat, date, my_bsc, my_bsc2, min_obs, min_obs_train, obs_to_add, start, mod_chan):
    new_date = True
    koho_win = 0
    GMM_win = 0
    koho_pts = 0
    GMM_pts = 0
    sr_dict={ "koho_RL": [], 'GMM_online': [], 'koho': [], 'GMM_offline_RL': [], 'GMM_offline': []}
    koho_success_rate = []
    GMM_success_rate = []
    my_sp=sp.Signal_processing()
    #for each file of the day (=date)
    for n in range(start, len(files[rat][date])-1):
        #get obs
        l_res, l_obs = my_bsc.convert_cpp_file(dir_name, '12'+date, files[rat][date][n:n+1], False, 'SCIOutput_')
        #if the trial is too short or have no neuron modulated we don't train
        if len(l_obs) > min_obs and np.array(l_obs).sum() > 0:
            l_obs_mod = get_only_mod_chan(l_obs, mod_chan)

            #test and plot
            success, l_of_res = my_bsc.test(l_obs_mod, l_res, HMM)
            success2, l_of_res2 = my_bsc2.test(l_obs_mod, l_res, HMM)
            l_res_gnd_truth, l_obs_trash = my_bsc.convert_cpp_file(dir_name, '12'+date, files[rat][date][n:n+1], True, 'SCIOutput_')
            l_res_gmm_rl = np.array(my_sp.load_m('GMM_RL_matrix_result/12'+date+'_resGMMwithRL_'+str(files[rat][date][n])+'.mat','res'))
            l_res_gmm = np.array(my_sp.load_m('GMM_RL_matrix_result/12'+date+'_resGMM_'+str(files[rat][date][n])+'.mat','res'))
            l_res_gmm_rl -= 2
            l_res_gmm -= 2
            #>1 cause l_res_gmm is uint so when -2 on 0 -> overflow and become 255
            l_res_gmm_rl[l_res_gmm_rl > 1] = 0
            l_res_gmm[l_res_gmm > 1] = 0
            l_of_res += l_of_res2
            size_trial=l_of_res[0].shape[0]
            l_of_res.append(l_res_gmm_rl[1, 0:size_trial])
            l_of_res.append(l_res_gmm[1, 0:size_trial])
            l_of_res.append(np.array(l_res_gnd_truth).argmax(1))
            l_of_res.append(l_res_gmm_rl[0, 0:size_trial])
            print success
            my_bsc.plot_result(l_of_res, '_fileRLonWorst_'+rat+'_'+date+'_'+str(files[rat][date][n:n+1]))

            #when new day first learn with mod_chan
            if new_date:
                new_date = False
                l_obs_koho = my_bsc.obs_classify_mod_chan(l_obs, l_res, mod_chan, 0)
                #l_obs_koho = my_bsc.obs_classify_mod_chan(l_obs, l_res, mod_chan)
                my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99, verbose)
                my_bsc2.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99, verbose)

            #in any case training
            my_bsc = train_nets(l_obs, l_obs_mod, l_res, my_bsc, mod_chan)
            my_bsc2 = train_nets2(l_obs, l_obs_mod, l_res, my_bsc2, mod_chan)
            #we update the modulated channel
            mod_chan = my_bsc.get_mod_chan(l_obs)

            #we compare our result with the GMM result to see wich method is better
            # tmp_koho_pts, tmp_GMM_pts, tmp_success_rate_koho, tmp_sucess_rate_GMM = compare_result(l_of_res[2], l_of_res[-2], l_of_res[0])
            # koho_pts += tmp_koho_pts
            # GMM_pts += tmp_GMM_pts
            sr_dict['koho_RL'].append(success_rate(l_of_res[2], l_of_res[0]))
            sr_dict['koho'].append(success_rate(l_of_res[5], l_of_res[0]))
            sr_dict['GMM_offline_RL'].append(success_rate(l_of_res[6], l_of_res[0]))
            sr_dict['GMM_offline'].append(success_rate(l_of_res[7], l_of_res[0]))
            sr_dict['GMM_online'].append(success_rate(l_of_res[8], l_of_res[0]))
            # if tmp_koho_pts > tmp_GMM_pts:
            #     koho_win += 1
            # elif tmp_GMM_pts > tmp_koho_pts:
            #     GMM_win += 1
            # else:
            #     koho_win += 1
            #     GMM_win += 1

    return sr_dict

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
            for i in range(42):
                if walk_get.shape[0] > 0:
                    #when we say walk
                    obs_ind = walk_get[rnd.randrange(walk_get.shape[0])]
                    save_koho = copy.copy(my_bsc.koho)
                    my_bsc.koho[0].update_closest_neurons(l_obs[obs_ind], push_away=False)
                    my_bsc.koho[1].update_closest_neurons(l_obs[obs_ind], push_away=True)

                if walk_expected.shape[0] > 0:
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

def train_nets2(l_obs, l_obs_mod, l_res, my_bsc, mod_chan):
    #we use l_obs_mod only to classify result
    HMM = True
    verbose = False
    # save_koho = copy.copy(my_bsc.koho)
    # success, l_of_res = my_bsc.test(l_obs_mod, l_res, HMM, verbose)

    l_obs_koho = my_bsc.obs_classify_mod_chan(l_obs, l_res, mod_chan, 0)
    my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99, verbose)

    success, l_of_res = my_bsc.test(l_obs_mod, l_res, HMM, verbose)

    walk_get = np.nonzero(l_of_res[1])[0]
    if walk_get.shape[0] == 0:
        my_bsc.was_bad += 1
        if my_bsc.was_bad > 2:
            l_obs_koho = my_bsc.obs_classify_kohonen(l_obs)
            my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99, verbose)
    else:
        my_bsc.was_bad = 0

    return my_bsc

base_dir = '../RT_classifier/BMIOutputs/BMISCIOutputs/'
# files = {'r31': OrderedDict([
#              ('03', range(1, 25)+range(52, 58)),
#              ('04', range(1, 45)),
#              ('06', range(78, 113)),
#              ('07', range(27, 51)),
#              ('10', range(6, 31)),
#              ('11', range(1, 16)),
#              ('12', range(1, 27)),
#              ('13', range(63, 89)),
#              ('14', range(1, 23))]),
#          'r32': OrderedDict([
#              ('03', range(25, 52)),
#              ('04', range(45, 83)),
#              ('06', range(42, 78)),
#              ('07', range(51, 82)),
#              ('10', range(31, 69)),
#              ('11', range(1, 36)),
#              ('12', range(27, 54)),
#              ('13', range(32, 63))]),
#          'r34': OrderedDict([
#              ('06', range(1, 42)),
#              ('07', range(1, 27)),
#              ('11', range(1, 31)),
#              ('12', range(54, 87)),
#              ('13', range(1, 32)),
#              ('14', range(23, 48))])
#          }

files = {'r32': OrderedDict([
             ('03', range(25, 52)),
             ('04', range(45, 83)),
             ('06', range(42, 78)),
             ('07', range(51, 82)),
             ('10', range(31, 69)),
             ('11', range(1, 36)),
             ('12', range(27, 54)),
             ('13', range(32, 63))]),
         }
# files = {'r32': OrderedDict([
#              ('03', range(25, 52))]),
#          }

#####################
######  START  ######
save_obj = True
ext_img = '.png'
save = True
show = False
HMM = True
verbose = False
number_of_chan = 128
group_chan_by = 4
my_bsc = bsc.brain_state_calculate(number_of_chan, group_chan_by)
my_bsc2 = bsc.brain_state_calculate(number_of_chan, group_chan_by)
#number of obs for stop and start we should have to train the network with kohonen
min_obs_train = 10
#number of obs we should have to test the file
min_obs = 10
obs_to_add = 0
my_bsc.save = save
my_bsc.show = show

sr={}
for rat in files.keys():
    init_networks = True
    my_bsc.build_networks()
    my_bsc2.build_networks()
    koho_win_r = 0
    koho_pts_r = 0
    koho_perfect_r = 0
    GMM_win_r = 0
    GMM_pts_r = 0
    GMM_perfect_r = 0
    mod_chan = []
    sr[rat]={}
    for date in files[rat].keys():
        dir_name = base_dir + 'Dec' + date + '/' + rat + '/'
        print '---------- ' + rat + ' ' + date + ' ----------'
        if init_networks:
            init_networks = False
            ##build one koho network and class obs with unsupervised learning
            l_res, l_obs = my_bsc.convert_cpp_file(dir_name, '12'+date, files[rat][date][0:5], False, 'SCIOutput_')
            l_obs_koho = my_bsc.obs_classify_kohonen(l_obs)
            #train networks
            my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.95, verbose)
            my_bsc2.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.95, verbose)
            start = 5
            mod_chan = my_bsc.get_mod_chan(l_obs)
        else:
            start = 0
        sr[rat][date] = train3(files, rat, date, my_bsc, my_bsc2, min_obs, min_obs_train, obs_to_add, start, mod_chan)


with open('success_rate_SCI_r32', 'wb') as my_file:
    my_pickler = pickle.Pickler(my_file)
    my_pickler.dump(sr)

print('###############')
print('####  END  ####')