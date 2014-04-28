import matplotlib.pyplot as plt
import brain_state_calculate as bsc
import numpy as np
import kohonen_neuron as kn
from collections import Counter
import signal_processing

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

net = kn.Kohonen(20, 20, 32, 5, 0.1, 3, 2, '.png', False, False)
for i in range(10):
    net.algo_kohonen(l_obs, False)

#create two group of neurons
net.evaluate_neurons(l_obs)
net.group_neuron_into_x_class(2)

#test the networks to know which group is stop and which is walk
l_res_trash, l_obs = my_bsc.convert_file(dir_name, 't_0423', files0423[5:6], False)
kn_res = []
for obs in l_obs:
    gp = net.find_best_group(obs)
    kn_res.append(gp.number)

#stop have more observation than walk
ct = Counter(kn_res)
keys = ct.keys()
print keys
if ct[keys[0]] > ct[keys[1]]:
    stop = keys[0]
    walk = keys[1]
else:
    stop = keys[1]
    walk = keys[0]

#some visual feedback of the network
if show:
    plt.plot(kn_res)
    plt.plot(np.array(l_res).argmax(1))
    plt.show()

##we train two network using result given by the previous network
#first classify obs using the koho network
l_res, l_obs = my_bsc.convert_file(dir_name, 't_0423', files0423[0:5], False)
kn_res = []
for obs in l_obs:
    gp = net.find_best_group(obs)
    kn_res.append(gp.number)

obs_stop = []
obs_walk = []
for i in range(len(kn_res)):
    if kn_res[i] == stop:
        obs_stop.append(l_obs[i])
    else:
        obs_walk.append(l_obs[i])

l_obs_koho = [obs_stop, obs_walk]

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