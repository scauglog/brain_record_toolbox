import matplotlib.pyplot as plt
import brain_state_calculate as bsc
import numpy as np

#In this script we train the kohonen network using another kohonen network
#learning is totally unsupervised
#####################
######  START  ######
from cpp_file_tools import cpp_file_tools

dir_name = '../RT_classifier/BMIOutputs/0423_r600/'
save_obj = False
ext_img = '.png'
save = True
show = False
files0423 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

my_bsc = bsc.brain_state_calculate(32, 'koho_RL', ext_img, save, show)
my_cft = cpp_file_tools(32, 1, ext_img, save, show)
my_bsc2 = bsc.brain_state_calculate(32, 'koho', ext_img, save, show)

##build one koho network and class obs with unsupervised learning
l_res, l_obs = my_bsc.cft.convert_cpp_file(dir_name, 't_0423', files0423[0:5], False)
#use training dataset (not working)
# sp = signal_processing.Signal_processing()
# l_obs = sp.load_m(dir_name+'trainSet140423.mat', 'BrainAct')
l_obs_koho = my_bsc.cft.obs_classify_kohonen(l_obs, 0.0)

#build and train networks
my_bsc.build_networks()
my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.95)
my_bsc2.build_networks()
my_bsc2.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.95)

#test all the trial and train the network (unsupervised) between each trial
results_koho = []
results_koho_retrained = []
results_GMM = []
for i in range(5, len(files0423)):
    #test
    print '### ### ### ### ### ### ### ### ###'
    print files0423[i-1:i]
    l_res, l_obs = my_cft.convert_cpp_file(dir_name, 't_0423', files0423[i - 1:i], False, cut_after_cue=False,
                                           init_in_walk=True)
    success, l_of_res = my_bsc.test(l_obs, l_res)
    success2, l_of_res2 = my_bsc2.test(l_obs, l_res)
    l_res_gmm, l_obs_trash = my_cft.convert_cpp_file(dir_name, 't_0423', files0423[i - 1:i], True, cut_after_cue=False,
                                                     init_in_walk=True)
    l_of_res.update(l_of_res2)
    l_of_res['GMM']=np.array(l_res_gmm).argmax(1)
    print success, success2
    results_koho_retrained.append(success)
    results_koho.append(success2)
    my_cft.plot_result(l_of_res, '_koho_unsupervised'+str(files0423[i:i+1])+'_')
    good = 0
    #calculate GMM success rate
    for i in range(l_of_res['gnd_truth'].shape[0]):
        if l_of_res['gnd_truth'][i] == l_of_res['GMM'][i]:
            good += 1
    results_GMM.append(good/float(l_of_res['gnd_truth'].shape[0]))
    #train networks using previous result
    l_obs_koho = my_cft.obs_classify_prev_res(l_obs, l_of_res[my_bsc.name], -3)
    my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.1, 14, 0.99, over_train_walk=True)

plt.figure()
plt.plot(results_koho, label='kohonen')
plt.plot(results_koho_retrained, label='kohonen retrained')
plt.plot(results_GMM, label='GMM')
plt.ylim(0, 1)
plt.legend(loc='lower center')
plt.ylabel('% of good answer')
plt.xlabel('trial')
plt.savefig('GMM_vs_kohonen_unsupervised.png', bbox_inches='tight')
if show:
    plt.show()
print('###############')
print('####  END  ####')