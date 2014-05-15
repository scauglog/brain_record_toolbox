import matplotlib.pyplot as plt
import pickle
import brain_state_calculate as bsc

#####################
######  START  ######
save_obj = True
ext_img = '.png'
save = False
show = False
##### r32
#11/26
files1126 = [32, 33, 34, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 53, 54, 55, 56, 57, 58, 59, 60]
#11/27
files1127 = [57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 86]
#12/03 SCI
files1203 = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

##### r31
#11/26
#files1126 = [88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123]
#11/27
#files1127 = [91, 92, 93, 94, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 114, 115, 116]
#12/03
#files1203 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 52, 53, 54, 55, 56, 57]

##### 0442_r600
#files0422 = [2,3,4,5,6,7,8]
my_bsc = bsc.brain_state_calculate(128, 4, 'koho')
my_bsc.build_networks()
print ('--------- Train healthy ---------')
l_res, l_obs = my_bsc.convert_cpp_file('1127', files1127[0:20], True)
l_obs_koho = my_bsc.obs_classify(l_obs, l_res)
#train networks
my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.8, True)
#test healthy
l_res, l_obs = my_bsc.convert_cpp_file('1127', files1127[20:22], True)
success, l_of_res = my_bsc.test(l_obs, l_res, True)
my_bsc.plot_result(l_of_res)
print '--------- end ---------'

print('--------- Train SCI ---------')
l_res, l_obs = my_bsc.convert_cpp_file('1203', files1203[0:5], False)
l_obs_koho = my_bsc.obs_classify(l_obs, l_res)
#train networks
my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.10, 14, 0.7, True)
for i in range(5, len(files1203)):
    l_res, l_obs = my_bsc.convert_cpp_file('1203', files1203[i-1:i], False)
    l_obs_koho = my_bsc.obs_classify_mixed_res(l_obs, l_res, 5)
    #train networks
    my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.01, 7, 0.7, True)
    # my_bsc.simulated_annealing(l_obs, l_obs_koho, l_res, 0.01, 7, 0.7, True)
    # my_bsc.koho[0].alpha = 0.01
    # for n in range(10):
    #     my_bsc.koho[0].algo_kohonen(l_obs_koho[0])

    #test SCI
    print '### ### ### ### ### ### ### ### ###'
    print files1203[i:i+1]
    l_res, l_obs = my_bsc.convert_cpp_file('1203', files1203[i:i+1], False)
    success, l_of_res = my_bsc.test(l_obs, l_res, True)
    print success
    my_bsc.plot_result(l_of_res)

plt.show()
print '--------- end Train SCI ---------'

#save networks
dir_name = ''
if save_obj:
    with open(dir_name + 'koho_networks', 'wb') as my_file:
        my_pickler = pickle.Pickler(my_file)
        my_pickler.dump(my_bsc.koho)

print('###############')
print('####  END  ####')