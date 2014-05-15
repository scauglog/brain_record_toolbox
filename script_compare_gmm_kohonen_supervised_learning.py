import brain_state_calculate as bsc
import numpy as np

#in this script we train the kohonen networks using ground truth
#####################
######  START  ######
dir_name = '../RT_classifier/BMIOutputs/0423_r600/'
save_obj = False
ext_img = '.png'
save = False
show = False
files0423 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
my_bsc = bsc.brain_state_calculate(32, 1, 'koho', ext_img, save, show)
my_bsc.build_networks()
print ('--------- Train ---------')
l_res, l_obs = my_bsc.convert_cpp_file(dir_name, 't_0423', files0423[0:5], False)
l_obs_koho = my_bsc.obs_classify(l_obs, l_res)
#train networks
my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.95, True)
#test
l_res, l_obs = my_bsc.convert_cpp_file(dir_name, 't_0423', files0423[5:6], False)
success, l_of_res = my_bsc.test(l_obs, l_res, True)
l_res_gmm, l_obs_trash = my_bsc.convert_cpp_file(dir_name, 't_0423', files0423[5:6], True)
l_of_res.append(np.array(l_res_gmm).argmax(1))
my_bsc.plot_result(l_of_res, '_file_'+str(files0423[5:6])+'_')
print '--------- end ---------'

print('--------- Train again and test all ---------')
for i in range(5, len(files0423)):
    #train again
    l_res, l_obs = my_bsc.convert_cpp_file(dir_name, 't_0423', files0423[i-1:i], False)
    l_obs_koho = my_bsc.obs_classify_mixed_res(l_obs, l_res, 5)
    #train networks
    my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.01, 7, 0.99, True)
    my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.01, 7, 0.7, True)
    my_bsc.koho[0].alpha = 0.01
    for n in range(10):
        my_bsc.koho[0].algo_kohonen(l_obs_koho[0])

    #test all
    print '### ### ### ### ### ### ### ### ###'
    print files0423[i:i+1]
    l_res, l_obs = my_bsc.convert_cpp_file(dir_name, 't_0423', files0423[i:i+1], False)
    success, l_of_res = my_bsc.test(l_obs, l_res, True)
    l_res_gmm, l_obs_trash = my_bsc.convert_cpp_file(dir_name, 't_0423', files0423[i:i+1], True)
    l_of_res.append(np.array(l_res_gmm).argmax(1))
    print success
    l_of_res.append((np.array(l_of_res[2])+np.array(l_of_res[3])) > 1)
    my_bsc.plot_result(l_of_res, 'file_'+str(files0423[i:i+1])+'_')

print '--------- end Train SCI ---------'

print('###############')
print('####  END  ####')