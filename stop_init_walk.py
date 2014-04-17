import kohonen_neuron as kn
import csv
import numpy as np
import copy
import random as rnd
import math
import matplotlib.pyplot as plt
from itertools import combinations
import pickle
import time

def convert_file(date, files, init_tail, isHealthy=False):
    l_obs = []
    l_res = []
    #read 'howto file reading' to understand
    if isHealthy:
        #col 4
        stop = ['1']
        init = ['']
        walk = ['2', '-2']
    else:
        #col 6
        stop = ['0', '3', '4']
        init = ['-2']
        walk = ['1', '2']

    for f in files:
        filename = 'r32/'+date+'healthyOutput_'+str(f)+'.txt'
        csvfile = open(filename, 'rb')
        file = csv.reader(csvfile, delimiter=' ', quotechar='"')
        #grab expected result in file and convert, grab input data
        for row in file:
            if len(row) > 7 and row[0] != '0':
                #if rat is healthy walk state are in col 4 otherwise in col 6 see 'howto file reading file'
                if isHealthy:
                    ratState = row[3]
                else:
                    ratState = row[5]

                if ratState in stop:
                    l_res.append([1, 0, 0])
                    l_obs.append(map(float, row[7:128+7]))
                elif ratState in init:
                    l_res.append([0, 1, 0])
                    l_obs.append(map(float, row[7:128+7]))
                elif ratState in walk:
                    l_res.append([0, 0, 1])
                    l_obs.append(map(float, row[7:128+7]))
        #set to init state the state before walk (currently deactivated)
        if isHealthy and False:
            for i in range(1, len(l_res)):
                if l_res[i] == [0, 0, 1] and l_res[i-1] == [1, 0, 0]:
                    for j in range(i-init_tail, i):
                        l_res[j] = [0, 1, 0]

        l_obs_d=[]
        l_obs_d.append([])
        for i in range(1, len(l_obs)):
            l_obs_d.append(np.array(l_obs[i])-np.array(l_obs[i-1]))
    return (l_res, l_obs)

def test(l_obs, l_res, koho, dist_count, print_res=True, return_res=False):
    good = 0
    test_all = True
    # A = np.array([[0.9, 0.1, 0], [0.1, 0.75, 0.15], [0.025, 0.025, 0.9]])
    # A = np.array([[0.665, 0.0193, 0.3152], [0.1369, 0.2697, 0.5934], [0.2628, 0.0981, 0.6392]])
    # A = np.array([[0.80, 0.0, 0.20], [0, 1, 0], [0.20, 0.0, 0.80]])
    A = np.array([[0.90, 0.0, 0.10], [0, 1, 0], [0.10, 0.0, 0.90]])

    #history length should be an odd number
    history_length = 3
    history = np.array([[1, 0, 0]])
    results = []
    prevP = np.array([1, 0, 0])
    raw_res = []
    for i in range(len(l_obs)):
        start = time.time()
        dist_res = []
        best_ns = []
        #find the best distance of the obs to each network
        for k in koho:
            dist_res.append(k.find_mean_best_dist(l_obs[i], dist_count))
            best_ns.append(k.find_best_X_neurons(l_obs[i], dist_count))
        #flatten list
        best_ns = [item for sublist in best_ns for item in sublist]

        #we test combination of each best n
        dist_comb = []
        if test_all:
            #test all combinations
            all_comb = combinations(best_ns, dist_count)
            for c in all_comb:
                l_dist = []
                for n in c:
                    l_dist.append(n.calc_error(l_obs[i]))
                dist_comb.append(np.array(l_dist).mean())
        else:
            #test some combinations
            for c in range(100):
                l_dist = []
                for n in random_combination(best_ns, dist_count):
                    l_dist.append(n.calc_error(l_obs[i]))
                dist_comb.append(np.array(l_dist).mean())

        prob_res = []
        #sort each dist for combination and find where the result of each network is in the sorted list
        #this give a percentage of accuracy for the network
        dist_comb = np.array(sorted(dist_comb, reverse=True))
        for k in range(len(koho)):
            prob_res.append(abs(dist_comb-dist_res[k]).argmin()/float(len(dist_comb)))
        prob_res = np.array(prob_res)
        raw_res.append(np.array(dist_res).argmax())

        #compute result with HMM
        P = []
        for k in range(prob_res.shape[0]):
            P.append(prevP.T.dot(A[:, k])*prob_res[k])
        #repair sum(P) == 1
        P = np.array(P).T/sum(P)

        #transform in readable result
        rank = P.argmax()
        res = [0, 0, 0]
        res[rank] = 1

        #save res and P.T
        results.append(res)
        prevP = P.T

        if res == l_res[i]:
            good += 1
        if print_res:
            print(np.array(res).argmax(), np.array(l_res[i]).argmax(), np.array(prob_res).argmax(), prob_res, prevP, (start-time.time()))
    plt.figure()
    plt.plot(np.array(l_res).argmax(1)+0.2)
    plt.plot(np.array(results).argmax(1))
    plt.plot(np.array(raw_res)-0.2)
    plt.ylim(-1, 3)
    plt.show()
    if len(l_obs) > 0:
        if return_res:
            return good/float(len(l_obs)), results
        else:
            return good/float(len(l_obs))
    else:
        print ('l_obs is empty')
    return good

def test_raw(l_obs, l_res, koho, dist_count):
    good = 0
    #history length should be an odd number
    history_length = 3
    history = np.array([[1, 0, 0]])
    results = []
    for i in range(len(l_obs)):
        dist_res = []
        #find the distance of the obs to each network
        for k in koho:
            dist_res.append(k.find_mean_best_dist(l_obs[i], dist_count))

        #transform result in array 0 or 1
        rank = dist_res.index(min(dist_res))
        res = [0, 0, 0]
        res[rank] = 1

        #use history to smooth change
        history = np.vstack((history, res))
        if history.shape[0] > history_length:
            history = history[1:, :]

        #transform result in array 0 or 1
        rank = history.argmax()
        res = [0, 0, 0]
        res[rank] = 1
        results.append(res)
        if res == l_res[i]:
            good += 1
    if len(l_obs) > 0:
        return good/float(len(l_obs))
    else:
        print ('l_obs is empty')
    return good

def simulated_annealing(koho, l_obs, l_obs_koho, dist_count, max_success, max_iteration, alpha_start):
    #inspired from simulated annealing, to determine when we should stop learning
    #initialize
    success = test_raw(l_obs, l_res, koho, dist_count)-0.1
    #learning coefficient of networks
    alpha = alpha_start
    #change alpha each X iteration
    change_alpha_iteration = 7
    #change alpha by a factor of
    #/!\ should be float
    change_alpha_factor = 10.0
    #factor to change alpha
    # Lambda = 0.9
    n = 0
    while success <= max_success and n < max_iteration:
        koho_cp = copy.copy(koho)
        #train each kohonen network
        for i in range(len(koho_cp)):
            #update learning coefficient
            koho_cp[i].alpha = alpha
            #no neighbor decrease for the first iteration
            if n == 0:
                koho_cp[i].algo_kohonen(l_obs_koho[i], False)
            else:
                koho_cp[i].algo_kohonen(l_obs_koho[i])
        #compute success of the networks
        success_cp = test_raw(l_obs, l_res, koho_cp, dist_count)

        print '---'
        print n
        print alpha
        print success_cp
        print math.exp(-(success-success_cp)/(alpha*1.0))
        #if we keep the same network for too long we go there
        if math.exp(-abs(success-success_cp)/(alpha*1.0)) in [0.0, 1.0]:
            print 'break'
            break
        #simulated annealing criterion to keep or not the trained network
        if success < success_cp or rnd.random() < math.exp(-abs(success-success_cp)/(alpha*1.0)):
            success = copy.copy(success_cp)
            koho = copy.copy(koho_cp)

        #learning rate decrease over iteration
        #change learning rate
        if n % change_alpha_iteration == 0:
            alpha /= change_alpha_factor
        n += 1

def obs_classify(l_obs, l_res):
    l_obs_stop = []
    l_obs_init = []
    l_obs_walk = []
    for i in range(len(l_res)):
        if l_res[i] == [1, 0, 0]:
            l_obs_stop.append(l_obs[i])
        elif l_res[i] == [0, 1, 0]:
            l_obs_init.append(l_obs[i])
        elif l_res[i] == [0, 0, 1]:
            l_obs_walk.append(l_obs[i])
    return [l_obs_stop, l_obs_init, l_obs_walk]

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(rnd.sample(xrange(n), r))
    return tuple(pool[i] for i in indices)
#####################
######  START  ######
save_obj = True
#koho_parameter
alpha = 0.01
koho_row = 6
koho_col = 7
#number of neighbor to update in the network
neighbor = 5
#min winning count to be consider as a good neuron
min_win = 7
#number of best neurons to keep for calculate distance of obs to the network
dist_count = 5
#end koho parameter
ext_img = '.png'
save = False
show = False
#number of record to set to init before walk
init_tail = 5
#A = np.matrix([[0.45, 0.45, 0.1], [0.1547, 0.2119, 0.6333], [0.25, 0.05, 0.7]])
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

#build the network
koho_stop = kn.Kohonen(koho_row, koho_col, 128, 5, alpha, neighbor, min_win, ext_img, save, show)
koho_init = kn.Kohonen(koho_row, koho_col, 128, 5, alpha, neighbor, min_win, ext_img, save, show)
koho_walk = kn.Kohonen(koho_row, koho_col, 128, 5, alpha, neighbor, min_win, ext_img, save, show)
koho = [koho_stop, koho_init, koho_walk]

print ('--------- Train healthy ---------')
l_res, l_obs = convert_file('1127', files1127[0:20], init_tail, True)
l_obs_koho = obs_classify(l_obs, l_res)
#train networks
simulated_annealing(koho, l_obs, l_obs_koho, dist_count, 0.80, 14, 0.1)
#test healthy
l_res, l_obs = convert_file('1127', files1127[20:22], init_tail, True)
print test(l_obs, l_res, koho, dist_count)
print '--------- end ---------'

print('--------- Train SCI ---------')
l_res, l_obs = convert_file('1203', files1203[0:10], init_tail, False)
l_obs_koho = obs_classify(l_obs, l_res)
#train networks
simulated_annealing(koho, l_obs, l_obs_koho, dist_count, 0.70, 14, 0.1)
l_res, l_obs = convert_file('1203', files1203[10:11], init_tail, False)
l_obs_koho = obs_classify(l_obs, l_res)
#train networks
simulated_annealing(koho, l_obs, l_obs_koho, dist_count, 0.70, 14, 0.1)


#test SCI
l_res, l_obs = convert_file('1203', files1203[11:13], init_tail, False)
print test(l_obs, l_res, koho, dist_count)
print '--------- end Train SCI ---------'

#save networks
dir_name = ''
if save_obj:
    with open(dir_name + 'koho_networks', 'wb') as my_file:
        my_pickler = pickle.Pickler(my_file)
        my_pickler.dump(koho)

print('###############')
print('####  END  ####')