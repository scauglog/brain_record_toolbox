import csv
import numpy as np
import copy
import random as rnd
import math
import pickle
import time

def convert_file(date, files, init_tail, isHealthy=False):
    l_obs = []
    l_res = []
    if isHealthy:
        stop = ['1']
        init = ['']
        walk = ['2', '-2']
    else:
        stop = ['0', '3', '4']
        init = ['-2']
        walk = ['1', '2']

    for f in files:
        filename = 'r31/'+date+'healthyOutput_'+str(f)+'.txt'
        csvfile = open(filename, 'rb')
        file = csv.reader(csvfile, delimiter=' ', quotechar='"')
        #grab expected result in file and convert, grab input data
        for row in file:
            if len(row) > 7 and row[0] != '0':
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

def test(l_obs, l_res, koho, dist_count, print_res=True):
    good = 0
    history_length = 3
    history = np.array([[1, 0, 0]])
    dtime=[]
    for i in range(len(l_obs)):
        dist_res = []
        start_t = time.time()
        for k in koho:
            dist_res.append(k.find_mean_best_dist(l_obs[i], dist_count))

        rank = dist_res.index(min(dist_res))
        res = [0, 0, 0]
        res[rank] = 1
        history = np.vstack((history, res))

        if history.shape[0] > history_length:
            history = history[1:, :]

        rank = history.mean(0).argmax()
        res = [0, 0, 0]
        res[rank] = 1
        end_t = time.time()
        dtime.append(end_t-start_t)
        if res == l_res[i]:
            good += 1
        if print_res:
            print(res, l_res[i], dist_res)
    print dtime
    print np.array(dtime).mean()
    if len(l_obs) > 0:
        return good/float(len(l_obs))
    else:
        print ('l_obs is empty')
    return good

def simulated_annealing(koho, l_obs, l_obs_koho, dist_count, max_success, max_iteration):
    success = test(l_obs, l_res, koho, dist_count, False)
    alpha = 0.1
    # Lambda = 0.9
    n = 0
    while success <= max_success and n < max_iteration:
        koho_cp = copy.copy(koho)
        for i in range(len(koho_cp)):
            koho_cp[i].alpha = alpha
            #no neighbor decrease for the first iteration
            if n == 0:
                koho_cp[i].algo_kohonen(l_obs_koho[i], False)
            else:
                koho_cp[i].algo_kohonen(l_obs_koho[i])
        success_cp = test(l_obs, l_res, koho_cp, dist_count, False)
        print '---'
        print n
        print alpha
        print success_cp
        print math.exp(-(success-success_cp)/(alpha*1.0))
        if math.exp(-abs(success-success_cp)/(alpha*1.0)) in [0.0, 1.0]:
            print 'break'
            break
        if success < success_cp or rnd.random() < math.exp(-abs(success-success_cp)/(alpha*1.0)):
            success = copy.copy(success_cp)
            koho = copy.copy(koho_cp)

        if n % 7 == 0:
            alpha /= 10.0
        # alpha *= Lambda
        n += 1
        #Temp *= Lambda

def obs_classify(l_obs,l_res):
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

#####################
######  START  ######
neighbor = 2
ext_img = '.png'
save = False
show = False
init_tail = 5
#number of best neurons to keep for calculate distance of obs to the network
dist_count = 3
save_obj = False
#A = np.matrix([[0.45, 0.45, 0.1], [0.1547, 0.2119, 0.6333], [0.25, 0.05, 0.7]])
##### r32
#11/26
# files1126 = [32, 33, 34, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 53, 54, 55, 56, 57, 58, 59, 60]
#11/27
# files1127 = [57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 86]
#12/03 SCI
# files1203 = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

##### r31
#11/26
files1126 = [88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123]
#11/27
files1127 = [91, 92, 93, 94, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 114, 115, 116]
#12/03
files1203 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 52, 53, 54, 55, 56, 57]

dir_name = ''
with open(dir_name + 'koho_networks_v1', 'rb') as my_file:
    koho = pickle.load(my_file)

l_res, l_obs = convert_file('1203', files1203[25:27], init_tail, False)
l_obs_koho = obs_classify(l_obs, l_res)
simulated_annealing(koho, l_obs, l_obs_koho, dist_count, 0.70, 42)

l_res, l_obs = convert_file('1203', files1203[28:29], init_tail, False)
print test(l_obs, l_res, koho, dist_count)

if save_obj:
    with open(dir_name + 'koho_networks_v2', 'wb') as my_file:
        my_pickler = pickle.Pickler(my_file)
        my_pickler.dump(koho)