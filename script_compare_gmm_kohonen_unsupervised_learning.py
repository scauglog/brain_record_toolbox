import matplotlib.pyplot as plt
import brain_state_calculate as bsc
import numpy as np

#In this script we train the kohonen network using another kohonen network
#learning is totally unsupervised
#####################
######  START  ######
dir_name = '../RT_classifier/BMIOutputs/0423_r600/'
save_obj = False
ext_img = '.png'
save = False
show = True
verbose = True
HMM = True
files0423 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

my_bsc = bsc.brain_state_calculate(32, 1, ext_img, save, show)
my_bsc.build_networks()

##build one koho network and class obs with unsupervised learning
l_res, l_obs = my_bsc.convert_file(dir_name, 't_0423', files0423[0:5], False)
#use training dataset (not working)
# sp = signal_processing.Signal_processing()
# l_obs = sp.load_m(dir_name+'trainSet140423.mat', 'BrainAct')
l_obs_koho = my_bsc.obs_classify_kohonen(l_obs, 0.0)

#build and train networks
my_bsc.build_networks()
my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.95, verbose)

#test network
l_res, l_obs = my_bsc.convert_file(dir_name, 't_0423', files0423[5:6], False)
success, l_of_res = my_bsc.test(l_obs, l_res, HMM, verbose)
#add gmm result to the plot (use test for healthy data)
l_res_gmm, l_obs_trash = my_bsc.convert_file(dir_name, 't_0423', files0423[5:6], True)
l_of_res.append(np.array(l_res_gmm).argmax(1))
my_bsc.plot_result(l_of_res, '_file_'+str(files0423[6:7])+'_')

#test all the trial and train the network (unsupervised) between each trial
for i in range(5, len(files0423)):
    l_res, l_obs = my_bsc.convert_file(dir_name, 't_0423', files0423[i-1:i], False)
    #train networks using previous result
    l_obs_koho = my_bsc.obs_classify_prev_res(l_obs, l_res, 5)
    my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.01, 7, 0.99, verbose)
    # #reinforcement training for stop
    # my_bsc.koho[0].alpha = 0.01
    # for n in range(10):
    #     my_bsc.koho[0].algo_kohonen(l_obs_koho[0])

    #test
    print '### ### ### ### ### ### ### ### ###'
    print files0423[i:i+1]
    l_res, l_obs = my_bsc.convert_file(dir_name, 't_0423', files0423[i:i+1], False)
    success, l_of_res = my_bsc.test(l_obs, l_res, HMM)
    l_res_gmm, l_obs_trash = my_bsc.convert_file(dir_name, 't_0423', files0423[i:i+1], True)
    l_of_res.append(np.array(l_res_gmm).argmax(1))
    print success
    l_of_res.append((np.array(l_of_res[2])+np.array(l_of_res[3])) > 1)
    my_bsc.plot_result(l_of_res, '_file_'+str(files0423[i:i+1])+'_')
if show:
    plt.show()
print('###############')
print('####  END  ####')