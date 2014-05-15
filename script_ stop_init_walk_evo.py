import brain_state_calculate as bsc
from collections import OrderedDict
import numpy as np
import signal_processing as sp
import time
import pickle

#train the network each day with mixed res and the new day with mod_chan
def train3(files, rat, date, my_bsc_RL, my_bsc_no_RL, min_obs, min_obs_train, obs_to_add, start, mod_chan):
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
        l_res, l_obs = my_bsc_RL.convert_cpp_file(dir_name, '12'+date, files[rat][date][n:n+1], False, 'SCIOutput_')
        #if the trial is too short or have no neuron modulated we don't train
        if len(l_obs) > min_obs and np.array(l_obs).sum() > 0:

            #test and plot
            success, l_of_res = my_bsc_RL.test(l_obs, l_res)
            success2, l_of_res2 = my_bsc_no_RL.test(l_obs, l_res)
            l_res_gnd_truth, l_obs_trash = my_bsc_RL.convert_cpp_file(dir_name, '12'+date, files[rat][date][n:n+1], True, 'SCIOutput_')
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
            my_bsc_RL.plot_result(l_of_res, '_fileRLonWorst_'+rat+'_'+date+'_'+str(files[rat][date][n:n+1]))

            #when new day first learn with mod_chan
            if new_date:
                new_date = False
                l_obs_koho = my_bsc_RL.obs_classify_mod_chan(l_obs, l_res, 0)
                #l_obs_koho = my_bsc.obs_classify_mod_chan(l_obs, l_res, mod_chan)
                my_bsc_RL.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99)
                my_bsc_no_RL.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99)

            #in any case training
            start_time=time.time()
            my_bsc_RL.train_nets(l_obs, l_res)
            end_time=time.time()
            print '######### time RL : '+str(end_time-start_time)+' #########'
            start_time=time.time()
            my_bsc_no_RL.train_nets(l_obs, l_res, with_RL=False)
            end_time=time.time()
            print '######### time without RL = '+str(end_time-start_time)+' #########'
            #we update the modulated channel
            my_bsc_RL.get_mod_chan(l_obs)
            my_bsc_no_RL.get_mod_chan(l_obs)

            #we compare our result with the GMM result to see wich method is better
            # tmp_koho_pts, tmp_GMM_pts, tmp_success_rate_koho, tmp_sucess_rate_GMM = compare_result(l_of_res[2], l_of_res[-2], l_of_res[0])
            # koho_pts += tmp_koho_pts
            # GMM_pts += tmp_GMM_pts
            sr_dict['koho_RL'].append(my_bsc_RL.success_rate(l_of_res[my_bsc_RL.name], l_of_res['gnd_truth']))
            sr_dict['koho'].append(my_bsc_RL.success_rate(l_of_res[my_bsc_no_RL.name], l_of_res['gnd_truth']))
            sr_dict['GMM_offline_RL'].append(my_bsc_RL.success_rate(l_of_res['GMM_RL'], l_of_res['GMM_gnd_truth']))
            sr_dict['GMM_offline'].append(my_bsc_RL.success_rate(l_of_res['GMM'], l_of_res['GMM_gnd_truth']))
            sr_dict['GMM_online'].append(my_bsc_RL.success_rate(l_of_res['GMM_online'], l_of_res['gnd_truth']))
            # if tmp_koho_pts > tmp_GMM_pts:
            #     koho_win += 1
            # elif tmp_GMM_pts > tmp_koho_pts:
            #     GMM_win += 1
            # else:
            #     koho_win += 1
            #     GMM_win += 1

    return sr_dict


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
files = {'r32': OrderedDict([
             ('03', range(25, 45))]),
         }

#####################
######  START  ######
save_obj = True
ext_img = '.png'
save = False
show = True
HMM = True
verbose = False
number_of_chan = 128
group_chan_by = 4
my_bsc_RL = bsc.brain_state_calculate(number_of_chan, group_chan_by, 'koho_RL')
my_bsc_no_RL = bsc.brain_state_calculate(number_of_chan, group_chan_by, 'koho')
#number of obs for stop and start we should have to train the network with kohonen
min_obs_train = 10
#number of obs we should have to test the file
min_obs = 10
obs_to_add = 0
my_bsc_RL.save = save
my_bsc_RL.show = show

sr={}
for rat in files.keys():
    init_networks = True
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
            l_res, l_obs = my_bsc_RL.convert_cpp_file(dir_name, '12'+date, files[rat][date][0:5], False, 'SCIOutput_')
            l_obs_koho = my_bsc_RL.obs_classify_kohonen(l_obs)
            #train networks
            my_bsc_RL.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.95)
            my_bsc_no_RL.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.95)
            start = 5
            mod_chan = my_bsc_RL.get_mod_chan(l_obs)
        else:
            start = 0
        sr[rat][date] = train3(files, rat, date, my_bsc_RL, my_bsc_no_RL, min_obs, min_obs_train, obs_to_add, start, mod_chan)


with open('success_rate_SCI_r32_v2', 'wb') as my_file:
    my_pickler = pickle.Pickler(my_file)
    my_pickler.dump(sr)

print('###############')
print('####  END  ####')