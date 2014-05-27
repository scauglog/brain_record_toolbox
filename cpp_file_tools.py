import csv
import numpy as np
import kohonen_neuron_c as kn
from matplotlib import pyplot as plt


class cpp_file_tools:
    def __init__(self, chan_count, group_chan, ext_img='.png', save=False, show=False):
        #group channel by
        self.group_chan = group_chan
        #result for the state
        self.stop = [1, 0]
        self.stop_index = self.stop.index(1)
        self.walk = [0, 1]
        self.walk_index = self.walk.index(1)
        self.ext_img = ext_img
        self.save = save
        self.show = show
        #params for file converter
        self.first_chan = 7
        self.chan_count = chan_count

        #kohonen classifier parameter
        self.kc_col = 12
        self.kc_row = 7
        self.kc_max_weight = 5
        self.kc_alpha = 0.1
        self.kc_neighbor = 3
        self.kc_min_win = 2

        #file convert
        self.stop_healthy = ['1', '-4', '0']
        self.init_healthy = ['2']
        self.walk_healthy = ['-2']

        self.stop_SCI = ['0', '3', '4']
        self.init_SCI = ['1', '2']
        self.walk_SCI = []

    def convert_one_cpp_file(self, filename, is_healthy=False, cut_after_cue=False, init_in_walk=True):
        #if is healthy the gnd truth is on col 4 else it's on col 6
        l_obs = []
        l_res = []
        #read 'howto file reading.txt' to understand
        if is_healthy:
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
            if len(row) > self.first_chan and row[0] != '0':
                #if rat is healthy walk state are in col 4 otherwise in col 6 see 'howto file reading file'
                if is_healthy:
                    ratState = row[3]
                else:
                    ratState = row[5]

                #add brain state to l_obs and convert number to float
                brain_state = self.convert_brain_state(row[self.first_chan:self.chan_count+self.first_chan])
                #'-1' added to ignore time where the rat is in the air added by 'add_ground_truth'
                if row[5] != '-1':
                    if row[5] in ['0', '3', '4'] and prevState in walk and cut_after_cue:
                            break

                    if ratState in stop:
                        #we don't take after the cue cause the rat reach the target
                        l_res.append(self.stop)
                        l_obs.append(brain_state)
                    elif ratState in walk:
                        l_res.append(self.walk)
                        l_obs.append(brain_state)

                    if row[5] in ['1', '2']:
                        prevState = walk[0]

        return l_res, l_obs

    def convert_cpp_file(self, dir_name, date, files, is_healthy=False, file_core_name='healthyOutput_', cut_after_cue=False, init_in_walk=True):
        #if is healthy gnd truth is on col 4 else on col 6

        files = self.convert_to_filename_list(dir_name, date, files, file_core_name)
        return self.read_cpp_files(files, is_healthy, cut_after_cue, init_in_walk)

    def read_cpp_files(self, files, is_healthy=False, cut_after_cue=False, init_in_walk=True):
        #if is healthy gnd truth is on col 4 else on col 6
        #convert cpp file to list of obs and list of res
        l_obs = []
        l_res = []
        #read 'howto file reading.txt' to understand
        for f in files:
            l_res_tmp, l_obs_tmp = self.convert_one_cpp_file(f, is_healthy, cut_after_cue, init_in_walk)
            l_obs += l_obs_tmp
            l_res += l_res_tmp
        return l_res, l_obs

    def convert_to_filename_list(self, dir_name, date, files, file_core_name):
        list=[]
        for f in files:
            list.append(dir_name+date+file_core_name+str(f)+'.txt')
        return list

    def convert_brain_state(self, obs):
        #convert what we read in the file to correct brain state
        obs_converted = np.arange(0, len(obs)/self.group_chan, 1.0)
        #sum chan X by X (X=self.group_chan)
        res = 0.0
        for i in range(len(obs)):
            res += float(obs[i])
            if i % self.group_chan == 0:
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
        l_obs_stop = []
        l_obs_walk = []
        for i in range(len(l_res)):
            if l_res[i] == self.stop:
                l_obs_stop.append(l_obs[i])
            elif l_res[i] == self.walk:
                l_obs_walk.append(l_obs[i])
        return [l_obs_stop, l_obs_walk]

    def obs_classify_good_res(self, l_obs, l_res, l_calc_res, obs_to_add=0):
        #add obs only if the network give the good answer
        l_obs_stop = []
        l_obs_walk = []
        for i in range(1, len(l_res)-1):
            if l_res[i] == self.stop and l_calc_res[i] == self.stop_index:
                l_obs_stop.append(l_obs[i])
                #when we change state and this is a good idea brain state before and after should be same state
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.stop, l_obs_stop)

            elif l_res[i] == self.walk and l_calc_res[i] == self.walk_index:
                l_obs_walk.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_bad_res(self, l_obs, l_res, l_calc_res, obs_to_add=0):
        #add obs only if the network give the bad answer
        l_obs_stop = []
        l_obs_walk = []
        for i in range(1, len(l_res)-1):
            if l_res[i] == self.stop and l_calc_res[i] == self.walk_index:
                l_obs_stop.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.stop, l_obs_stop)
            elif l_res[i] == self.walk and l_calc_res[i] == self.stop_index:
                l_obs_walk.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_mixed_res(self, l_obs, l_res, l_calc_res, obs_to_add=0):
        #add obs to stop when no cue and to walk only if the network give the right answer
        l_obs_stop = []
        l_obs_walk = []
        #list_of_res
        #0 = res expected
        #1 = res calculate before HMM
        #2 = res calculate after HMM
        for i in range(1, len(l_res)-1):
            if l_res[i] == self.stop:
                l_obs_stop.append(l_obs[i])

            elif l_res[i] == self.walk and l_calc_res[i] == self.walk_index:
                l_obs_walk.append(l_obs[i])
                self.add_extra_obs(l_obs, l_res, obs_to_add, l_calc_res, i, self.walk, l_obs_walk)

        return [l_obs_stop, l_obs_walk]

    def obs_classify_prev_res(self, l_obs, l_calc_res, obs_to_add=0):
        #we class obs using only the previous result no ground truth involved here
        #we need ground truth to call test
        l_obs_stop = []
        l_obs_walk = []
        #when obs_to add is <0 we remove obs
        obs_to_remove = []
        for i in range(1, len(l_obs)-1):
            if l_calc_res[i] == self.stop_index:
                l_obs_stop.append(l_obs[i])
            elif l_calc_res[i] == self.walk_index:
                l_obs_walk.append(l_obs[i])
                #when state change
                if l_calc_res[i] != l_calc_res[i+1]:
                    if obs_to_add > 0:
                        for n in range(i-obs_to_add, i):
                            if 0 < n < len(l_obs):
                                l_obs_walk.append(l_obs[n])
                        for n in range(i, i+obs_to_add):
                            if 0 < n < len(l_obs):
                                l_obs_walk.append(l_obs[n])
                    elif obs_to_add < 0:
                        for n in range(i, i+abs(obs_to_add)):
                            if 0 < n < len(l_obs):
                                obs_to_remove.append(l_obs[n])
                        for n in range(i-abs(obs_to_add), i):
                            if 0 < n < len(l_obs):
                                obs_to_remove.append(l_obs[n])
        #remove obs when obs_to_add <0
        if len(obs_to_remove) > 0:
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

    def obs_classify_kohonen(self, l_obs, acceptance_factor=0.0):
        print '###### classify with kohonen ######'
        while True:
            #while the network don't give 2 classes
            n = 0
            while True:
                net = kn.Kohonen(self.kc_col, self.kc_row, l_obs[0].shape[0], self.kc_max_weight, self.kc_alpha, self.kc_neighbor, self.kc_min_win, self.ext_img, False, False)

                for i in range(10):
                    net.algo_kohonen(l_obs, False)

                #create two group of neurons
                net.evaluate_neurons(l_obs)
                net.group_neuron_into_x_class(2)
                n+=1
                if len(net.groups) == 2:
                    break
                elif n > 4:
                    #when we still don't have a valid number of class after many trials we raise an exception
                    raise Exception("error the network can't converge for that number of class")
                else:
                    print len(net.groups), len(net.good_neurons)

            #test the networks to know which group is stop and which is walk
            dict_res = {}
            for gp in net.groups:
                dict_res[gp.number] = []

            for obs in l_obs:
                gp = net.find_best_group(obs)
                dict_res[gp.number].append(obs)

            #stop have more observation than walk
            keys = dict_res.keys()
            print keys
            if len(keys) == 2:
                if len(dict_res[keys[0]]) > len(dict_res[keys[1]]):
                    stop = keys[0]
                    walk = keys[1]
                else:
                    stop = keys[1]
                    walk = keys[0]

                l_obs_koho = [dict_res[stop], dict_res[walk]]
                nb_stop = len(dict_res[stop])
                nb_walk = len(dict_res[walk])
                print 'nb stop', nb_stop, 'nb_walk', nb_walk, nb_walk/float(nb_stop)
                if acceptance_factor > 0 and (acceptance_factor < nb_walk/float(nb_stop) < 1.5) or (nb_walk + nb_stop < 150 and nb_walk > 20):
                    return l_obs_koho
                elif acceptance_factor == 0:
                    return l_obs_koho
            else:
                return [[], []]

    @staticmethod
    def add_extra_obs(l_obs, l_res, obs_to_add, calculate_res, i, res_expected, l_obs_state):
        #when the brain state change we add value before or after the observed state
        obs_to_remove=[]
        if 1 < i < len(l_res)-1:
            if calculate_res[i-1] != calculate_res[i]:
                if obs_to_add > 0:
                    for n in range(i-obs_to_add, i):
                        if 0 < n < len(l_res) and l_res[n] == res_expected:
                            l_obs_state.append(l_obs[n])
                    for n in range(i, i+obs_to_add):
                        if 0 < n < len(l_res) and l_res[n] == res_expected:
                            l_obs_state.append(l_obs[n])
                elif obs_to_add < 0:
                    for n in range(i-abs(obs_to_add), i):
                        if 0 < n < len(l_res):
                            obs_to_remove.append(l_obs[n])
                    for n in range(i, i+abs(obs_to_add)):
                        if 0 < n < len(l_res):
                            obs_to_remove.append(l_obs[n])
        if len(obs_to_remove) > 0:
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
        walk_before_cue = []
        walk_after_cue = []
        current_walk = 0
        for i in range(len(l_res)):
            #when we are at the end of the walk or cue change and we walk
            if (l_res[i] != l_res[i-1] or l_expected_res[i] != l_expected_res[i-1]) and l_res[i] == 0:
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
        w_before_cue1, w_after_cue1 = self.class_result(l_res1, l_expected_res)
        w_before_cue2, w_after_cue2 = self.class_result(l_res2, l_expected_res)
        block_length = 0.1
        min_walk = 3/block_length
        long_walk = 1/block_length
        short_walk = 0.2/block_length
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

        #good training have one long walk
        #who has less long walk but at least one
        if 0 < long_w1.shape[0] < long_w2.shape[0]:
            win_point1 += 1
        elif 0 < long_w2.shape[0] < long_w1.shape[0]:
            win_point2 += 1
        elif long_w1.shape[0] < 1 and long_w1.shape[0] < 1:
            win_point1 -= 1
            win_point2 -= 1
        else:
            win_point1 += 1
            win_point2 += 2

        #who has less short walk
        if short_w1.shape[0] < short_w2.shape[0]:
            win_point1 += 1
        elif short_w2.shape[0] < short_w1.shape[0]:
            win_point2 += 1
        else:
            win_point1 += 1
            win_point2 += 1

        #before cue fav short walk
        #init mean cause array.mean() return none if array is empty
        if w_before_cue1.shape[0] > 0:
            wbc1_mean = w_before_cue1.mean()
        else:
            wbc1_mean = 0

        if w_before_cue2.shape[0] > 0:
            wbc2_mean = w_before_cue2.mean()
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
        if w_after_cue1.shape[0] > 0:
            wdc1_mean = w_after_cue1.mean()
        else:
            wdc1_mean = 0

        if w_after_cue2.shape[0] > 0:
            wdc2_mean = w_after_cue2.mean()
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
        if all_w1.shape[0] > 0:
            all_w1_max = all_w1.max()
        else:
            all_w1_max = 0
        if all_w2.shape[0] > 0:
            all_w2_max = all_w2.max()
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
        if w_before_cue1.sum() < w_before_cue2.sum():
            win_point1 += 1
        elif w_before_cue2.sum() < w_before_cue1.sum():
            win_point2 += 1
        else:
            win_point1 += 1
            win_point2 += 1

        #no walk before cue is good
        if w_before_cue1.shape[0] == 0:
            win_point1 += 1
        if w_before_cue2.shape[0] == 0:
            win_point2 += 1

        #at least min_walk of walk
        if all_w1.sum() > min_walk:
            win_point1 += 1
        if all_w2.sum() > min_walk:
            win_point2 += 1

        if no_perfect:
            return win_point1, win_point2
        else:
            #his this trial perfect (no walk before cue, at least X second of walk)
            if w_before_cue1.shape[0] == 0:
                success_rate1 = min(1, all_w1.sum() / float(min_walk))
            if w_before_cue2.shape[0] == 0:
                success_rate2 = min(1, all_w2.sum() / float(min_walk))

            return win_point1, win_point2, success_rate1, success_rate2

    def success_rate(self, l_res, l_expected_res):
        #if there is no walk when we want rest and at least X second of walk
        block_length = 0.1
        min_walk = 3/block_length
        w_before_cue, w_after_cue = self.class_result(l_res, l_expected_res)
        if w_before_cue.shape[0] == 0:
            return min(1.0, w_after_cue.sum()/float(min_walk))
        else:
            return 0

    def accuracy(self, l_res, l_expected_res):
        #mean of (%success when we want walk + %success when we want rest)
        walk_success = 0.0
        walk_total = 0.0
        rest_success = 0.0
        rest_total = 0.0
        for i in range(len(l_res)):
            if l_expected_res[i] == self.walk_index:
                walk_total += 1
                if l_res[i] == self.walk_index:
                    walk_success += 1
            elif l_expected_res[i] == self.stop_index:
                rest_total += 1
                if l_res[i] == self.stop_index:
                    rest_success += 1
        return (walk_success/walk_total+rest_success/rest_total)/2

    def plot_result(self, list_of_res, extra_txt='', dir_path=''):
        plt.figure(figsize=(10, 14))

        cpt = 0
        color=['b', 'r', 'g', 'm', 'c', 'y', 'k']
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
            plt.savefig(dir_path+'' + extra_txt + self.ext_img, dpi=100)
        if not self.show:
            plt.close()

    def show_fig(self):
        plt.show()