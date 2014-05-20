import brain_state_calculate as bsc
from collections import OrderedDict
import numpy as np
import signal_processing as sp
import time
import pickle
import random as rnd
from scipy.stats.mstats import mquantiles

def shuffle_obs(l_obs):
    rnd.shuffle(l_obs)

#train the network each day with mixed res and the new day with mod_chan
def train3(files, rat, date, my_bsc_RL, my_bsc_no_RL, my_cft, min_obs, obs_to_add, start):
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
        l_res, l_obs = my_cft.convert_cpp_file(dir_name, '12'+date, files[rat][date][n:n+1], False, 'SCIOutput_', cut_after_cue=True)
        #shuffle_obs(l_obs)
        l_obs=obs_to_hist(l_obs)
        #if the trial is too short or have no neuron modulated we don't train
        if len(l_obs) > min_obs and np.array(l_obs).sum() > 0:

            #test and plot
            success, l_of_res = my_bsc_RL.test(l_obs, l_res)
            success2, l_of_res2 = my_bsc_no_RL.test(l_obs, l_res)
            l_res_gnd_truth, l_obs_trash = my_cft.convert_cpp_file(dir_name, '12'+date, files[rat][date][n:n+1], True, 'SCIOutput_', cut_after_cue=True)
            l_res_gmm_rl = np.array(my_sp.load_m('GMM_RL_matrix_result/12'+date+'_resGMMwithRL_'+str(files[rat][date][n])+'.mat','res'))
            l_res_gmm = np.array(my_sp.load_m('GMM_RL_matrix_result/12'+date+'_resGMM_'+str(files[rat][date][n])+'.mat','res'))
            l_res_gmm_rl -= 2
            l_res_gmm -= 2
            #>1 cause l_res_gmm is uint so when -2 on 0 -> overflow and become 255
            l_res_gmm_rl[l_res_gmm_rl > 1] = 0
            l_res_gmm[l_res_gmm > 1] = 0
            l_of_res.update(l_of_res2)
            size_trial = l_of_res['gnd_truth'].shape[0]
            l_of_res['GMM_RL'] = l_res_gmm_rl[1, 0:size_trial]
            l_of_res['GMM'] = l_res_gmm[1, 0:size_trial]
            l_of_res['GMM_online'] = np.array(l_res_gnd_truth).argmax(1)
            l_of_res['GMM_gnd_truth'] = l_res_gmm_rl[0, 0:size_trial]
            print success
            my_cft.plot_result(l_of_res, '_fileRLonWorst_'+rat+'_'+date+'_'+str(files[rat][date][n:n+1]))
            #when new day first learn with mod_chan
            if new_date:
                new_date = False
                try:
                    l_obs_koho = my_cft.obs_classify_mixed_res(l_obs, l_res, l_of_res[my_bsc_RL.name], obs_to_add)
                    my_bsc_RL.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99)
                    l_obs_koho = my_cft.obs_classify_mixed_res(l_obs, l_res, l_of_res2[my_bsc_no_RL.name], obs_to_add)
                    my_bsc_no_RL.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99)
                except ValueError:
                    print 'go to the next trial'


            #in any case training
            start_time = time.time()
            try:
                my_bsc_RL.train_nets(l_obs, l_res, my_cft, obs_to_add=obs_to_add)
                end_time = time.time()
                print '######### time RL : '+str(end_time-start_time)+' #########'
                start_time = time.time()
                my_bsc_no_RL.train_nets(l_obs, l_res, my_cft, obs_to_add=obs_to_add, with_RL=False)
                end_time = time.time()
                print '######### time without RL = '+str(end_time-start_time)+' #########'
                #we update the modulated channel
                my_bsc_RL.mod_chan = my_cft.get_mod_chan(l_obs)
                my_bsc_no_RL.mod_chan = my_cft.get_mod_chan(l_obs)
                #we compare our result with the GMM result to see wich method is better
                # tmp_koho_pts, tmp_GMM_pts, tmp_success_rate_koho, tmp_sucess_rate_GMM = compare_result(l_of_res[2], l_of_res[-2], l_of_res[0])
                # koho_pts += tmp_koho_pts
                # GMM_pts += tmp_GMM_pts
                sr_dict['koho_RL'].append(my_cft.success_rate(l_of_res[my_bsc_RL.name], l_of_res['gnd_truth']))
                sr_dict['koho'].append(my_cft.success_rate(l_of_res[my_bsc_no_RL.name], l_of_res['gnd_truth']))
                sr_dict['GMM_offline_RL'].append(my_cft.success_rate(l_of_res['GMM_RL'], l_of_res['GMM_gnd_truth']))
                sr_dict['GMM_offline'].append(my_cft.success_rate(l_of_res['GMM'], l_of_res['GMM_gnd_truth']))
                sr_dict['GMM_online'].append(my_cft.success_rate(l_of_res['GMM_online'], l_of_res['gnd_truth']))
                # if tmp_koho_pts > tmp_GMM_pts:
                #     koho_win += 1
                # elif tmp_GMM_pts > tmp_koho_pts:
                #     GMM_win += 1
                # else:
                #     koho_win += 1
                #     GMM_win += 1
            except ValueError:
                print 'goto the next trial'


    return sr_dict

def obs_to_hist(l_obs):
    l_hist = []
    qVec=np.array(np.arange(0.1,1,0.1))
    for obs in l_obs:
        l_hist.append(mquantiles(obs,qVec))
    return l_hist

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
#              ('03', range(25, 45))]),
#          }

#####################
######  START  ######
hist=True
save_obj = True
ext_img = '.png'
save = True
show = False
HMM = True
verbose = False
number_of_chan = 128
group_chan_by = 4
my_bsc_RL = bsc.brain_state_calculate(number_of_chan/group_chan_by, 'koho_RL')
my_cft = bsc.cpp_file_tools(number_of_chan, group_chan_by, ext_img, save, show)
my_bsc_no_RL = bsc.brain_state_calculate(number_of_chan/group_chan_by, 'koho')
#number of obs for stop and start we should have to train the network with kohonen
min_obs_train = 10
#number of obs we should have to test the file
min_obs = 10
obs_to_add = 0
my_bsc_RL.save = save
my_bsc_RL.show = show

sr={}
global_time_start = time.time()
for rat in files.keys():
    init_networks = True
    if hist:
        my_bsc_RL.weight_count = 9
        my_bsc_RL.A = np.array([[0.90, 0.10], [0.10, 0.90]])
        my_bsc_no_RL.weight_count = 9
        my_bsc_no_RL.A = np.array([[0.90, 0.10], [0.10, 0.90]])
    my_bsc_RL.build_networks()
    my_bsc_no_RL.build_networks()
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
            l_res, l_obs = my_cft.convert_cpp_file(dir_name, '12'+date, files[rat][date][0:5], False, 'SCIOutput_', cut_after_cue=True)
            #shuffle_obs(l_obs)
            l_obs = obs_to_hist(l_obs)
            l_obs_koho = my_cft.obs_classify_kohonen(l_obs)
            #train networks
            my_bsc_RL.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.95)
            my_bsc_no_RL.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.95)
            start = 5
            my_bsc_RL.mod_chan = my_cft.get_mod_chan(l_obs)
            my_bsc_no_RL.mod_chan = my_cft.get_mod_chan(l_obs)
        else:
            start = 0
        sr[rat][date] = train3(files, rat, date, my_bsc_RL, my_bsc_no_RL, my_cft, min_obs, obs_to_add, start)


with open('success_rate_SCI_r32_hist', 'wb') as my_file:
    my_pickler = pickle.Pickler(my_file)
    my_pickler.dump(sr)
delta_time = time.time()-global_time_start
print("#####  "+str(delta_time)+'  #####')
print('###############')
print('####  END  ####')
