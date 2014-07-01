#!python
#cython: boundscheck=False
#cython: wraparound=False

import csv
import numpy as np
import kohonen_neuron_c as kn
from matplotlib import pyplot as plt
import settings

cdef class cpp_file_tools:
    def __init__(self, int chan_count, int group_chan, ext_img='.png', save=False, show=False, settings_path="cppfileSettings.yaml", ion=True):
        self.chan_count = chan_count
        self.group_chan = group_chan
        self.ext_img = ext_img
        self.save = save
        self.show = show
        self.load_settings(settings_path)
        if ion:
            plt.ion()

    def load_settings(self, settings_path):
        cftset = settings.Settings(settings_path).get()

        #result for the state
        self.stop = cftset['stop']
        self.stop_index = self.stop.index(1)
        self.walk = cftset['walk']
        self.walk_index = self.walk.index(1)
        #params for file converter
        self.first_chan = cftset['first_chan']

        #kohonen classifier parameter
        self.kc_col = cftset['kc_col']
        self.kc_row = cftset['kc_row']
        self.kc_max_weight = cftset['kc_max_weight']
        self.kc_alpha = cftset['kc_alpha']
        self.kc_neighbor = cftset['kc_neighbor']
        self.kc_min_win = cftset['kc_min_win']

        #file convert
        self.stop_healthy = cftset['stop_healthy']
        self.init_healthy = cftset['init_healthy']
        self.walk_healthy = cftset['walk_healthy']

        self.stop_SCI = cftset['stop_SCI']
        self.init_SCI = cftset['init_SCI']
        self.walk_SCI = cftset['walk_SCI']

        self.cue_col = cftset['cue_col']
        self.result_col = cftset['result_col']

        self.block_length = cftset['block_length']

    def convert_one_cpp_file(self, filename, use_classifier_result=False, cut_after_cue=False, init_in_walk=True, on_stim=False):
        cdef np.ndarray[DTYPE_t, ndim=1] brain_state
        #if is healthy the gnd truth is on col 4 else it's on col 6
        l_obs = []
        l_res = []
        #read 'howto file reading.txt' to understand
        if use_classifier_result:
            #col 4
            stop = self.stop_healthy
            walk = self.walk_healthy
            init = self.init_healthy

        else:
            #col 6
            stop = self.stop_SCI
            walk = self.walk_SCI
            init = self.init_SCI

        #when we read contusion file
        if init_in_walk:
            walk += init
        else:
            #when we read SCI file
            stop += init

        csv_file = open(filename, 'rb')
        file = csv.reader(csv_file, delimiter=' ', quotechar='"')
        prevState = stop[0]
        #grab expected result in file and convert, grab input data
        for row in file:
            if <int>len(row) > self.first_chan and row[0] != '0':
                #if rat is healthy walk state are in col 4 otherwise in col 6 see 'howto file reading file'
                if use_classifier_result:
                    ratState = row[self.result_col]
                else:
                    ratState = row[self.cue_col]

                #add brain state to l_obs and convert number to float
                brain_state = self.convert_brain_state(row[self.first_chan:self.chan_count+self.first_chan])
                #'-1' added to ignore time where the rat is in the air added by 'add_ground_truth'
                if row[self.cue_col] != '-1':
                    #cut after cue
                    if row[self.cue_col] in self.stop_SCI and prevState in walk and cut_after_cue:
                        break
                    #don't select stim off
                    if row[4] == '0' and on_stim:
                        continue

                    if ratState in stop:
                        #we don't take after the cue cause the rat reach the target
                        l_res.append(self.stop)
                        l_obs.append(brain_state)
                    elif ratState in walk:
                        l_res.append(self.walk)
                        l_obs.append(brain_state)

                    if row[self.cue_col] in self.walk_SCI:
                        prevState = walk[0]

        return l_res, l_obs

    def convert_cpp_file(self, dir_name, date, files, use_classifier_result=False, file_core_name='healthyOutput_',
                         cut_after_cue=False, init_in_walk=True):
        files = self.convert_to_filename_list(dir_name, date, files, file_core_name)
        return self.read_cpp_files(files, use_classifier_result, cut_after_cue, init_in_walk)

    def read_cpp_files(self, files, use_classifier_result=False, cut_after_cue=False, init_in_walk=True, on_stim=False):
        #convert cpp file to list of obs and list of res
        l_obs = []
        l_res = []
        #read 'howto file reading.txt' to understand
        for f in files:
            l_res_tmp, l_obs_tmp = self.convert_one_cpp_file(f, use_classifier_result, cut_after_cue, init_in_walk, on_stim)
            l_obs += l_obs_tmp
            l_res += l_res_tmp
        return l_res, l_obs

    def convert_to_filename_list(self, dir_name, date, files, file_core_name):
        list=[]
        for f in files:
            list.append(dir_name+date+file_core_name+str(f)+'.txt')
        return list

    cpdef np.ndarray convert_brain_state(self, object obs):
        cdef np.ndarray[DTYPE_t, ndim=1] obs_converted
        cdef double res
        cdef int i
        #convert what we read in the file to correct brain state
        obs_converted = <np.ndarray[DTYPE_t, ndim=1]>np.arange(0, len(obs)/self.group_chan, 1.0)
        #sum chan X by X (X=self.group_chan)
        res = 0.0
        for i in range(<int>len(obs)):
            res += <double>float(obs[i])
            if (i+1) % self.group_chan == 0:
                obs_converted[i/self.group_chan] = res
                res = 0.0
        return obs_converted

    def get_mod_chan(self, l_obs):
        #return the chan where a neuron is active (modulated chan)
        l_obs = np.array(l_obs)
        mod_chan = l_obs.sum(0).nonzero()[0]
        return mod_chan

    def obs_classify(self, l_obs, l_res):
        #classify obs using the cue
        cdef int i
        l_obs_stop = []
        l_obs_walk = []
        for i in range(<int>len(l_res)):
            if l_res[i] == self.stop:
                l_obs_stop.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[i])
            elif l_res[i] == self.walk:
                l_obs_walk.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[i])
        return [l_obs_stop, l_obs_walk]

    def obs_classify_good_res(self, l_obs, l_res, l_calc_res, obs_to_add=0):
        #add obs only if the network give the good answer
        cdef int i
        l_obs_stop = []
        l_obs_walk = []
        for i in range(1, <int>len(l_res)-1):
            if l_res[i] == self.stop and l_calc_res[i] == self.stop_index:
                l_obs_stop.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[i])
                #when we change state and this is a good idea brain state before and after should be same state
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.stop, l_obs_stop)

            elif l_res[i] == self.walk and l_calc_res[i] == self.walk_index:
                l_obs_walk.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_bad_res(self, l_obs, l_res, l_calc_res, obs_to_add=0):
        #add obs only if the network give the bad answer
        cdef int i
        l_obs_stop = []
        l_obs_walk = []
        for i in range(1, <int>len(l_res)-1):
            if l_res[i] == self.stop and l_calc_res[i] == self.walk_index:
                l_obs_stop.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.stop, l_obs_stop)
            elif l_res[i] == self.walk and l_calc_res[i] == self.stop_index:
                l_obs_walk.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_mixed_res(self, l_obs, l_res, l_calc_res, int obs_to_add=0):
        #add obs to stop when no cue and to walk only if the network give the right answer
        cdef int i
        l_obs_stop = []
        l_obs_walk = []
        #list_of_res
        #0 = res expected
        #1 = res calculate before HMM
        #2 = res calculate after HMM
        for i in range(1, <int>len(l_res)-1):
            if l_res[i] == self.stop:
                l_obs_stop.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[i])

            elif l_res[i] == self.walk and <int>l_calc_res[i] == self.walk_index:
                l_obs_walk.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_prev_res(self, l_obs, l_calc_res, int obs_to_add=0):
        #we class obs using only the previous result no ground truth involved here
        #we need ground truth to call test
        cdef int i, n
        l_obs_stop = []
        l_obs_walk = []
        #when obs_to add is <0 we remove obs
        obs_to_remove = []
        for i in range(1, <int>len(l_obs)-1):
            if <int>l_calc_res[i] == self.stop_index:
                l_obs_stop.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[i])
            elif <int>l_calc_res[i] == self.walk_index:
                l_obs_walk.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[i])
                #when state change
                if <int>l_calc_res[i] != <int>l_calc_res[i+1]:
                    if obs_to_add > 0:
                        for n in range(i-obs_to_add, i):
                            if 0 < n < <int>len(l_obs):
                                l_obs_walk.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[n])
                        for n in range(i, i+obs_to_add):
                            if 0 < n < <int>len(l_obs):
                                l_obs_walk.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[n])
                    elif obs_to_add < 0:
                        for n in range(i, i+<int>abs(obs_to_add)):
                            if 0 < n < <int>len(l_obs):
                                obs_to_remove.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[n])
                        for n in range(i-<int>abs(obs_to_add), i):
                            if 0 < n < <int>len(l_obs):
                                obs_to_remove.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[n])
        #remove obs when obs_to_add <0
        if <int>len(obs_to_remove) > 0:
            tmp_l = []
            for obs in l_obs_walk:
                to_add = True
                for obs_r in obs_to_remove:
                    if (obs_r == obs).all():
                        to_add = False
                        break
                if to_add:
                    tmp_l.append(obs)
            l_obs_walk = tmp_l
        return [l_obs_stop, l_obs_walk]

    def obs_classify_kohonen(self, l_obs):
        print('###### classify with kohonen ######')
        cdef int n, i, stop, walk, nb_stop, nb_walk
        cdef kn.Kohonen net
        cdef np.ndarray[DTYPE_t, ndim=1] obs
        cdef kn.Group_neuron gp

        while True:
            #while the network don't give 2 classes
            n = 0
            while True:
                net = kn.Kohonen(self.kc_col, self.kc_row, <int>l_obs[0].shape[0], self.kc_max_weight, self.kc_alpha, self.kc_neighbor, self.kc_min_win, self.ext_img, False, False)

                for i in range(10):
                    net.algo_kohonen(l_obs, False)

                #create two group of neurons
                net.evaluate_neurons(l_obs)
                net.group_neuron_into_x_class(2)
                n+=1
                if <int>len(net.groups) == 2:
                    break
                elif n > 4:
                    #when we still don't have a valid number of class after many trials we raise an exception
                    raise Exception("error the network can't converge for that number of class")
                else:
                    print(len(net.groups), len(net.good_neurons))

            #test the networks to know which group is stop and which is walk
            dict_res = {}
            for gp in net.groups:
                dict_res[<int>gp.number] = []

            for obs in l_obs:
                gp = net.find_best_group(<np.ndarray[DTYPE_t, ndim=1]>obs)
                dict_res[<int>gp.number].append(<np.ndarray[DTYPE_t, ndim=1]>obs)

            #stop have more observation than walk
            keys = dict_res.keys()
            print(keys)
            if <int>len(keys) == 2:
                if <int>len(dict_res[keys[0]]) > <int>len(dict_res[keys[1]]):
                    stop = <int>keys[0]
                    walk = <int>keys[1]
                else:
                    stop = <int>keys[1]
                    walk = <int>keys[0]

                l_obs_koho = [dict_res[stop], dict_res[walk]]
                nb_stop = <int>len(dict_res[stop])
                nb_walk = <int>len(dict_res[walk])
                print('nb stop', nb_stop, 'nb_walk', nb_walk, nb_walk/float(nb_stop))
                return l_obs_koho
            else:
                return [[], []]

    @staticmethod
    def add_extra_obs(l_obs, l_res, int obs_to_add, calculate_res, int i, res_expected, l_obs_state):
        #when the brain state change we add value before or after the observed state
        cdef int n
        cdef np.ndarray[DTYPE_t, ndim=1] obs
        obs_to_remove=[]
        if 1 < i < <int>len(l_res)-1:
            if <int>calculate_res[i-1] != <int>calculate_res[i]:
                if obs_to_add > 0:
                    for n in range(i-obs_to_add, i):
                        if 0 < n < <int>len(l_res) and l_res[n] == res_expected:
                            l_obs_state.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[n])
                    for n in range(i, i+obs_to_add):
                        if 0 < n < <int>len(l_res) and l_res[n] == res_expected:
                            l_obs_state.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[n])
                elif obs_to_add < 0:
                    for n in range(i-<int>abs(obs_to_add), i):
                        if 0 < n < <int>len(l_res):
                            obs_to_remove.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[n])
                    for n in range(i, i+<int>abs(obs_to_add)):
                        if 0 < n < <int>len(l_res):
                            obs_to_remove.append(<np.ndarray[DTYPE_t, ndim=1]>l_obs[n])
        if <int>len(obs_to_remove) > 0:
            tmp_l = []
            for obs in l_obs_state:
                to_add = True
                for obs_r in obs_to_remove:
                    if (obs_r == obs).all():
                        to_add = False
                        break
                if to_add:
                    tmp_l.append(obs)
            l_obs_state = tmp_l

    #classify the given result
    @staticmethod
    def class_result(l_res, l_expected_res):
        cdef int current_walk, i
        walk_before_cue = []
        walk_after_cue = []
        current_walk = 0
        for i in range(1, <int>len(l_res)):
            #when we are at the end of the walk or cue change and we walk
            if (<int>l_res[i] != <int>l_res[i-1] or <int>l_expected_res[i] != <int>l_expected_res[i-1]) or i+1==len(l_res):
                if current_walk != 0:
                    if l_expected_res[i-1] == 0:
                        walk_before_cue.append(current_walk)
                    else:
                        walk_after_cue.append(current_walk)
                current_walk = 0

            if l_res[i] == 1:
               current_walk += 1
        return np.array(walk_before_cue), np.array(walk_after_cue)

    #classify the given result
    def compare_result(self, l_res1, l_res2, l_expected_res, no_perfect=False):
        cdef double min_walk, long_walk, short_walk, success_rate1, success_rate2
        cdef int win_point1, win_point2, all_w1_max, all_w2_max
        w_before_cue1, w_after_cue1 = self.class_result(l_res1, l_expected_res)
        w_before_cue2, w_after_cue2 = self.class_result(l_res2, l_expected_res)
        min_walk = 3/self.block_length
        long_walk = 1/self.block_length
        short_walk = 0.2/self.block_length
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

        ##good training have one long walk
        ##who has less long walk but at least one
        if 0 < <int>long_w1.shape[0] < <int>long_w2.shape[0]:
           win_point1 += 1
        elif 0 < <int>long_w2.shape[0] < <int>long_w1.shape[0]:
           win_point2 += 1
        elif <int>long_w1.shape[0] < 1 and <int>long_w2.shape[0] < 1:
           win_point1 -= 1
           win_point2 -= 1
        else:
           win_point1 += 1
           win_point2 += 2

        ##who has less short walk
        if <int>short_w1.shape[0] < <int>short_w2.shape[0]:
           win_point1 += 1
        elif <int>short_w2.shape[0] < <int>short_w1.shape[0]:
           win_point2 += 1
        else:
           win_point1 += 1
           win_point2 += 1

        cdef double wbc1_mean, wbc2_mean, wdc1_mean, wdc2_mean
        #before cue fav short walk
        #init mean cause array.mean() return none if array is empty
        if <int>w_before_cue1.shape[0] > 0:
            wbc1_mean = <double>w_before_cue1.mean()
        else:
            wbc1_mean = 0

        if <int>w_before_cue2.shape[0] > 0:
            wbc2_mean = <double>w_before_cue2.mean()
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
        if <int>w_after_cue1.shape[0] > 0:
           wdc1_mean = <double>w_after_cue1.mean()
        else:
           wdc1_mean = 0

        if <int>w_after_cue2.shape[0] > 0:
           wdc2_mean = <double>w_after_cue2.mean()
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
        if <int>all_w1.shape[0] > 0:
           all_w1_max = <int>all_w1.max()
        else:
           all_w1_max = 0
        if <int>all_w2.shape[0] > 0:
           all_w2_max = <int>all_w2.max()
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
        if <int>w_before_cue1.sum() < <int>w_before_cue2.sum():
            win_point1 += 1
        elif <int>w_before_cue2.sum() < <int>w_before_cue1.sum():
            win_point2 += 1
        else:
            win_point1 += 1
            win_point2 += 1

        #no walk before cue is good
        if <int>w_before_cue1.shape[0] == 0:
            win_point1 += 1
        if <int>w_before_cue2.shape[0] == 0:
            win_point2 += 1

        #at least min_walk of walk
        if <int>all_w1.sum() > min_walk:
            win_point1 += 1
        if <int>all_w2.sum() > min_walk:
            win_point2 += 1

        if no_perfect:
            return win_point1, win_point2
        else:
            #his this trial perfect (no walk before cue, at least X second of walk)
            if <int>w_before_cue1.shape[0] == 0:
                success_rate1 = min(1, all_w1.sum() / float(min_walk))
            if <int>w_before_cue2.shape[0] == 0:
                success_rate2 = min(1, all_w2.sum() / float(min_walk))

            return win_point1, win_point2, success_rate1, success_rate2

    cpdef double success_rate(self, object l_res, object l_expected_res):
        #if there is no walk when we want rest and at least X second of walk
        cdef double min_walk
        min_walk = 3/self.block_length
        w_before_cue, w_after_cue = self.class_result(l_res, l_expected_res)
        if <int>w_before_cue.shape[0] == 0:
            return min(1.0, w_after_cue.sum()/float(min_walk))
        else:
            return 0

    cpdef double accuracy(self, l_res, l_expected_res):
        #mean of (%success when we want walk + %success when we want rest)
        cdef double walk_success, walk_total, rest_sucess, rest_total
        cdef int i
        walk_success = 0.0
        walk_total = 0.0
        rest_success = 0.0
        rest_total = 0.0
        for i in range(<int>len(l_res)):
            if <int>l_expected_res[i] == self.walk_index:
                walk_total += 1
                if <int>l_res[i] == self.walk_index:
                    walk_success += 1
            elif <int>l_expected_res[i] == self.stop_index:
                rest_total += 1
                if <int>l_res[i] == self.stop_index:
                    rest_success += 1
        if walk_total > 0 and rest_total > 0:
            return (walk_success/walk_total+rest_success/rest_total)/2
        elif walk_total > 0 and rest_total < 0:
            return (walk_success/walk_total)/2
        elif rest_total > 0 and walk_total < 0:
            return (rest_success/rest_total)/2
        else:
            return 0

    def plot_result(self, list_of_res, extra_txt='', dir_path='', big_figure=True, gui=False):
        if not gui or self.show:
            if big_figure:
                plt.figure(figsize=(10, 14))
            else:
                plt.figure()

        cpt = 1
        color = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
        for key in list_of_res:
            plt.subplot(len(list_of_res.keys()), 1, cpt)
            plt.plot(list_of_res[key], color[cpt%len(color)], label=key)
            plt.ylabel(key, rotation=0)
            plt.ylim(-0.2, 1.2)
            for i in range(len(list_of_res['gnd_truth'])):
                if list_of_res['gnd_truth'][i-1] != list_of_res['gnd_truth'][i]:
                    plt.vlines(i, -0.2, len(list_of_res)*1.2+0.2, 'b', '--')
            cpt += 1

        plt.tight_layout()

        if self.save:
            plt.savefig(dir_path + 'result' + extra_txt + self.ext_img, dpi=100)

        if self.show:
            plt.show()
        else:
            if gui:
                plt.clf()
            else:
                plt.close()

    def plot_obs(self, l_obs, l_res, extra_txt='', dir_path='', gui=False):
        if not gui or self.show:
            plt.figure()
        obs = np.vstack((np.array(l_obs).T,np.array(l_res).argmax(1).T*4,np.array(l_res).argmax(1).T*4))
        plt.imshow(obs, interpolation='none')

        if self.save:
            plt.savefig(dir_path+'obs'+extra_txt+self.ext_img)

        if self.show:
            plt.show()
        else:
            if gui:
                plt.clf()
            else:
                plt.close()

    def show_fig(self):
        plt.show()

    def close_fig(self):
        plt.close()