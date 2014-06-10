import scipy.io
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.signal import butter, filtfilt
from scipy import stats
import copy
import kohonen_neuron_c as nn
import random as rnd
import csv


class Signal_processing:
    def __init__(self, save=False, show=False, img_ext='.png'):
        self.save_img = save
        self.show = show
        self.img_ext = img_ext

    #filter a multichannel signal

    #load matlab matrix
    def load_m(self, filename, var_name):
        return scipy.io.loadmat(filename)[var_name]

    #filter multichannel filter : subtract col mean -> butterworth -> zscore
    def signal_mc_filtering(self, signal, lowcut, highcut, fs):
        print('\n### signal filtering ###')
        #mean of each column
        mean = signal.mean(0)
        #duplicate matrix for each row
        mean = np.tile(mean, (signal.shape[0], 1))
        signal -= mean

        #apply butterworth filter and zscore
        for chan in range(signal.shape[0]):
            signal[chan] = self.butter_bandpass_filter(signal[chan], lowcut, highcut, fs, 10)
            signal[chan] = stats.zscore(signal[chan])

        return signal

    def plot_signal(self, signal):
        plt.figure('signal after filtering')
        plt.suptitle('')
        plt.plot(range(signal.shape[0]), signal)
        if self.save_img:
            plt.savefig('signal_after_filtering' + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    #find spikes by threshold
    def find_spikes(self, signal, a_spike, b_spike, tresh):
        print('\n find spike')
        cpt = 0
        last_spike = 0
        list_spike = []
        spikes_values = []
        spikes_time = []
        for i in signal:
            #list store x (x=size of the spike length) values of signal
            list_spike.append(i)
            if len(list_spike) > (a_spike + b_spike):
                del list_spike[0]
                #if list_spike[b_spike] < tresh and (cpt - last_spike) > a_spike and list_spike[b_spike]<list_spike[b_spike+1]:
                if list_spike[b_spike] < tresh and (cpt - last_spike) > a_spike:
                    spikes_values.append(copy.copy(list_spike))
                    spikes_time.append(cpt - a_spike)
                    last_spike = cpt
            cpt += 1
        #can't use directly np.array because it raise an error for first spike (empty array)
        spikes_values = np.array(spikes_values)
        return spikes_values, spikes_time

    #use moving average to smooth spikes
    def smooth_spikes(self, spikes_values, window_size):
        s = []
        window = np.ones(int(window_size)) / float(window_size)
        #window=[1/4.0,2/4.0,1/4.0]
        for val in spikes_values:
            s.append(np.convolve(val, window, 'same'))
        return np.array(s)

    #class each spike in the correct cluster and return a list containing class number same size as number of spike
    def classify_spikes(self, spikes_values, spikes_time, clusters, threshold_template):
        spikes_classes = []
        classed_count = 0
        for i in range(len(spikes_time)):
            best_clus = self.find_best_cluster(spikes_values[i], clusters, threshold_template)
            if best_clus is not None:
                best_clus.add_spike(spikes_values[i], spikes_time[i])
                classed_count += 1
                spikes_classes.append(best_clus.number)
            else:
                spikes_classes.append(-1)
        print ('spike classified: ' + str(classed_count))
        print ('spike unclassified: ' + str(len(spikes_values) - classed_count))
        return spikes_classes

    #find the best cluster (minimal distance between spike and template) for an observation
    def find_best_cluster(self, obs, clusters, threshold_template):
        best_dist = threshold_template
        best_clus = None
        for clus in clusters:
            #compute dist
            dist = clus.dist(obs)
            #check if dist is better than previous
            if dist < best_dist:
                best_clus = clus
                best_dist = dist
        return best_clus

    #plot spike found after threshold some of them can be not a spike
    def plot_spikes(self, spikes_values, spike_count, extra_text=''):
        s = copy.copy(spikes_values).tolist()
        if spike_count > len(s):
            spike_count = len(s)

        plt.figure()
        plt.suptitle('spikes find' + extra_text)
        for i in range(spike_count):
            r = rnd.randrange(len(s))
            value = s.pop(r)
            plt.plot(range(len(value)), value)

        if self.save_img:
            plt.savefig('spikes_find' + extra_text + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    #plot signal and below a line same height as the cluster the spike belongs to
    def plot_signal_spikes_classified(self, spikes_time, spikes_classes, extra_text=''):
        plt.figure()
        graph_x = [0]
        graph_y = [0]
        for i in range(len(spikes_time)):
            graph_x.append(spikes_time[i])
            graph_y.append(0)
            graph_x.append(spikes_time[i])
            graph_y.append(spikes_classes[i])
            graph_x.append(spikes_time[i])
            graph_y.append(0)
        plt.plot(graph_x, graph_y)

        if self.save_img:
            plt.savefig('signal_spikes' + extra_text + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    #useful to show plot without importing matplotlib
    def show_plot(self):
        if self.show:
            plt.show()

    #find the pattern of spikes using mode
    def find_spike_template_mode(self, spikes_values):
        print('\n## find mode ##')
        spike_mode = []
        for col in spikes_values.transpose():
            values, med = np.histogram(col, bins=15)
            tmp = values.argmax()
            spike_mode.append(med[tmp])

        return spike_mode

    #find patterns of spikes using kohonen network
    def find_spike_template_kohonen(self, spikes_values, col, row, weight_count, max_weight, alpha, neighbor, min_win,
                                    dist_treshold):
        print('\n## kohonen ##')

        koho_map = nn.Kohonen(col, row, weight_count, max_weight, alpha, neighbor, min_win, self.img_ext, self.save_img,
                         self.show)
        koho_map.algo_kohonen(spikes_values)

        print('# find best neurons #')
        koho_map.evaluate_neurons(spikes_values)

        print('# group neurons #')
        koho_map.group_neurons(dist_treshold)
        #koho_map.find_cluster_center(spikes_values,20)
        return koho_map

    #compute parameters for butterworth filter
    def butter_bandpass(self, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    #filter signal using Butterworth filter and filtfilt
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order)
        y = filtfilt(b, a, data)
        return y

    #usefull to load csv file without using import csv
    def load_csv(self, filename):
        csvfile = open(filename, 'rb')
        return csv.reader(csvfile, delimiter=',', quotechar='"')

    #put exported vicon data (csv file) into a dictionary
    def vicon_extract(self, data, vicon_dict={}):
        events = ['Foot Strike', 'Foot Off', 'Event']
        context = ['Right', 'Left', 'General']
        #change to true when we are in events extraction
        events_extraction = False
        #change to true when we are in synch extraction
        sync_extraction = False
        cpt = 0
        #if no dictionnary is passed create a new dict
        if context[0] not in vicon_dict:
            for c in context:
                vicon_dict[c] = {}
                for e in events:
                    vicon_dict[c][e] = []
            #store frequency sampling for TDT data
            vicon_dict['fs'] = 0
            #store beginning of the TDT
            vicon_dict['synch'] = 0
        for row in data:
            #when there is a row with no data means we are at the end of data set for this kind of data
            if len(row) == 0:
                events_extraction = False
                sync_extraction = False
            elif len(row) == 1 and row[0] == 'EVENTS':
                events_extraction = True
                #ignore header so -1
                cpt = -1
            elif len(row) == 1 and row[0] == 'ANALOG':
                sync_extraction = True
                #ignore the 4th first line containing sampling frequency and header
                cpt = -3
            #add time events to the correct list
            elif events_extraction and cpt > 0:
                vicon_dict[row[1]][row[2]].append(float(row[3]))
            elif sync_extraction and cpt == -2:
                vicon_dict['fs'] = float(row[0])
            elif sync_extraction and cpt > 0 and float(row[11]) > 1:
                vicon_dict['synch'] = float(row[0])
                break
            cpt += 1
        return vicon_dict

    #correct the time of the vicon data to match with TDT time
    #because we usually add 2s before vicon start when extracting TDT signal
    #and vicon have a small delay with TDT (read the delay in ANALOG)
    def synch_vicon_with_TDT(self, vicon_dict, TDT_padding=0):
        #synchronise vicon data with the beginning of the TDT
        events = ['Foot Strike', 'Foot Off', 'Event']
        context = ['Right', 'Left', 'General']
        for c in context:
            for e in events:
                list_time=[]
                for time in vicon_dict[c][e]:
                    time -= vicon_dict['synch'] / vicon_dict['fs']
                    time += TDT_padding
                    list_time.append(time)

                vicon_dict[c][e] = sorted(list_time)
        return vicon_dict

    #for plotting vertical line at each beginning of a step phase
    def binarise_vicon_step(self, steps):
        step_time = [0]
        step_bin = [0]
        for val in steps:
            step_time.append(val)
            step_bin.append(0)
            step_time.append(val)
            step_bin.append(1)
            step_time.append(val)
            step_bin.append(0)

        return step_time, step_bin

    #cut signal in block and calculate the number of spike that appear during the block length
    #global_fire is the cumulative count of spike for each block
    def fire_rate(self, all_chan_clusters, length_signal, fs, block_duration):
        global_fire = []
        all_chan_firerates = []
        number_of_block = int((length_signal / fs) / block_duration) + 1
        #initialise list for storing global firing count of each block
        for time in range(number_of_block):
            global_fire.append(0)

        for chan in range(len(all_chan_clusters)):
            print('---- channel: ' + str(chan + 1) + ' ----')
            firerate_chan = []
            for clu in all_chan_clusters[chan]:
                #init list for storing firing count of each block of this cluster
                firerate_cluster = []
                for time in range(number_of_block):
                    firerate_cluster.append(0)
                for time in clu.spikes_time:
                    block_number = int((time / fs) / block_duration)
                    firerate_cluster[block_number] += 1
                    global_fire[block_number] += 1
                firerate_chan.append(firerate_cluster)
            all_chan_firerates.append(firerate_chan)
        return all_chan_firerates, global_fire

    #plot fire rate over time of the cluster and above the step phase
    def plot_global_firerate(self, global_fire, strike_time, strike_bin, off_time, off_bin, length_signal, fs,
                             block_duration, extra_txt=''):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(strike_time, strike_bin)
        plt.plot(off_time, off_bin)
        plt.xlim(0, int(length_signal / fs) + 1)
        plt.subplot(2, 1, 2)
        plt.xlim(0, int((length_signal / fs) / block_duration) + 1)
        plt.plot(np.array(global_fire)/block_duration)

        if self.save_img:
            plt.savefig('walk_firerate_correlation_global' + extra_txt + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    #for each channel plot fire rate of each cluster below step phase
    def plot_all_chan_firerates(self, all_chan_firerates, strike_time, strike_bin, off_time, off_bin, length_signal, fs,
                                block_duration, extra_txt=''):
        for chan in range(len(all_chan_firerates)):
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(strike_time, strike_bin)
            plt.plot(off_time, off_bin)
            plt.xlim(0, int(length_signal / fs) + 1)
            plt.subplot(2, 1, 2)
            plt.xlim(0, int((length_signal / fs) / block_duration) + 1)
            for list_times in all_chan_firerates[chan]:
                plt.plot(np.array(list_times)/block_duration)

            if self.save_img:
                plt.savefig('walk_firerate_correlation_chan' + str(chan + 1) + extra_txt + self.img_ext,
                            bbox_inches='tight')
            if not self.show:
                plt.close()

    #find end and start of a step
    def find_step_time(self, times_start, times_end):
        steps_time = []
        for i in range(len(times_start) - 1):
                end = filter(lambda x: times_start[i] < x < times_start[i + 1], times_end)
                if not len(end) == 0:
                    end = end[0]
                    start = times_start[i]
                    if end-start < 1:
                        steps_time.append([start, end])
        return steps_time

    #return list of complete step (with swing and stance phase)
    def find_full_step_time(self, stance_steps_time, swing_steps_time):
        full_step=[]
        for step_stance in stance_steps_time:
            for step_swing in swing_steps_time:
                if step_stance[1] == step_swing[0] and step_swing[0]-step_stance[0] < 1:
                    full_step.append([step_stance[0], step_stance[1], step_swing[1]])
                    break
        return full_step

    #compare mean (ttest) of firing rate between step phases
    def stat_mean_std_step(self, data_dict):
        stat_dict={}
        for chan in range(len(data_dict)):
            stat_dict[chan] = {}
            for cluster in data_dict[chan]:
                stat_dict[chan][cluster] = {'all_vs_stance': 0, 'all_vs_swing': 0, 'stance_vs_swing': 0}
                if not len(data_dict[chan][cluster]['all_delta']) == 0 and not len(data_dict[chan][cluster]['stance_delta_all_step']) == 0:
                    t, p = stats.ttest_ind(data_dict[chan][cluster]['all_delta'], data_dict[chan][cluster]['stance_delta_all_step'], equal_var=False)
                    stat_dict[chan][cluster]['all_vs_stance'] = p

                if not len(data_dict[chan][cluster]['all_delta']) == 0 and not len(data_dict[chan][cluster]['swing_delta_all_step']) == 0:
                    t, p = stats.ttest_ind(data_dict[chan][cluster]['all_delta'], data_dict[chan][cluster]['swing_delta_all_step'], equal_var=False)
                    stat_dict[chan][cluster]['all_vs_swing'] = p

                if not len(data_dict[chan][cluster]['swing_delta_all_step']) == 0 and not len(data_dict[chan][cluster]['stance_delta_all_step']) == 0:
                    t, p = stats.ttest_ind(data_dict[chan][cluster]['swing_delta_all_step'], data_dict[chan][cluster]['stance_delta_all_step'], equal_var=False)
                    stat_dict[chan][cluster]['stance_vs_swing'] = p

        return stat_dict

    #count the number of spike well classed or wrong classed
    def step_spike_error_stat(self, steps_list, spikes_time):
        correct = 0
        false = 0
        missing = 0
        for step in steps_list:
            done = False
            for time in spikes_time:
                if step[0] < time < step[1]:
                    correct += 1
                    done = True
                    break
            if not done:
                missing += 1
        for time in spikes_time:
            done = False
            for step in steps_list:
                if step[0] < time < step[1]:
                    done = True
                    break
            if not done:
                false += 1
        return correct, false, missing

    #
    def find_delta_time_step(self, steps_time, spikes_time, delta_time, fs, max_step_time):
            delta_all_step = []
            delta_each_step = []
            for step in steps_time:
                    # exclude too long move
                    if step[1] - step[0] < max_step_time:
                        delta_list = []
                        for j in range(len(spikes_time) - 1):
                            #if the two spikes append during step then add their delta time in the list
                            if step[0] < spikes_time[j] / fs < step[1] and step[0] < spikes_time[j + 1] / fs < step[1]:
                                delta_all_step.append(delta_time[j])
                                delta_list.append(delta_time[j])
                        delta_each_step.append(delta_list)
            #delta_all_step contain all delta time for this step phase
            #delta_each_step contain a list of delta time for each step
            return delta_each_step, delta_all_step

    #compare (ttest) spike mean delta time during each step phase
    def cluster_step_ttest(self, all_delta, stance_delta, swing_delta):
        all_vs_stance = 0
        all_vs_swing = 0
        stance_vs_swing = 0
        if not len(all_delta) == 0 and not len(stance_delta) == 0:
            t, p = stats.ttest_ind(all_delta, stance_delta, equal_var=False)
            all_vs_stance = p

        if not len(all_delta) == 0 and not len(swing_delta) == 0:
            t, p = stats.ttest_ind(all_delta, swing_delta, equal_var=False)
            all_vs_swing = p

        if not len(swing_delta) == 0 and not len(stance_delta) == 0:
            t, p = stats.ttest_ind(swing_delta, stance_delta, equal_var=False)
            stance_vs_swing = p
        return all_vs_stance, all_vs_swing, stance_vs_swing

    #for each spike determine in wich step phase it has more probably append
    def class_spike_in_step(self, spikes_time, delta_time, all_mean, stance_mean, swing_mean, window_size, fs):
        tmp_list = []
        stance_spikes_time = []
        swing_spikes_time = []
        other_spikes_time = []
        spike_is_step = []
        #init list
        #window size is the number of delta between spike to take for mean comparison to determine which step it is
        for x in range(window_size):
            spike_is_step.append(0)
        for i in range(len(spikes_time)-1):
            #we add the current delta_time to the list and delete the oldest delta time
            tmp_list.append(delta_time[i])
            if len(tmp_list) > window_size:
                del tmp_list[0]

            #when we have the correct number of delta_time in the list
            #we compute the distance between list mean delta and mean of th step phase
            #we took the smallest distance to define which step phase it is.
            if len(tmp_list) == window_size:
                err_all = (np.mean(tmp_list) - all_mean)**2
                err_stance = (np.mean(tmp_list) - stance_mean)**2
                err_swing = (np.mean(tmp_list) - swing_mean)**2
                min_err = min(err_all, err_stance, err_swing)
                if min_err == err_all:
                    spike_is_step.append(0)
                    other_spikes_time.append(spikes_time[i] / fs)
                elif min_err == err_stance:
                    spike_is_step.append(1)
                    stance_spikes_time.append(spikes_time[i] / fs)
                elif min_err == err_swing:
                    spike_is_step.append(-1)
                    swing_spikes_time.append(spikes_time[i] / fs)
                else:
                    print('error in class_spike_step spike' + spikes_time[i] + ' is not classed')
        return swing_spikes_time, stance_spikes_time, other_spikes_time, spike_is_step

    #plot the spike according to step guessed (-1=swing step phase, 1=stance step phase, 0=no step
    def plot_step_spike_classify(self, strike_times, strike_bin, off_times, off_bin, spike_is_step, spikes_time, length_signal, fs, extra_txt=''):
        plot_x = [0]
        plot_y = [0]
        for i in range(len(spike_is_step)):
            plot_x.append(spikes_time[i] / fs)
            plot_y.append(0)
            plot_x.append(spikes_time[i] / fs)
            plot_y.append(spike_is_step[i])
            plot_x.append(spikes_time[i] / fs)
            plot_y.append(0)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(strike_times, strike_bin)
        plt.plot(off_times, off_bin)
        plt.xlim(0, int(length_signal / fs) + 1)
        plt.subplot(2, 1, 2)
        plt.xlim(0, int(length_signal / fs) + 1)
        plt.plot(plot_x, plot_y)

        if self.save_img:
            plt.savefig('spike_step_classed' + extra_txt + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    #compute spike frequency relative to step duration
    #phase step duration is normalized and we count the number of spike that append during this period
    def phase_step_spike_fq(self, spikes_time, full_step, nb_block, fs):
        stance_spike_fq=[]
        swing_spike_fq=[]
        for step in full_step:
            stance_block_duration = (step[1]-step[0])/nb_block
            swing_block_duration = (step[2]-step[1])/nb_block
            step_stance_count = []
            step_swing_count = []
            for i in range(nb_block):
                step_stance_count.append(0)
                step_swing_count.append(0)

            for spike_time in spikes_time:
                #if stance phase
                if step[0] < spike_time/fs < step[1]:
                    list_block = np.arange(step[0], step[1], stance_block_duration)
                    list_block = np.hstack((list_block, step[1]))
                    for i in range(nb_block):
                        if list_block[i] < spike_time/fs < list_block[i+1]:
                            step_stance_count[i] += 1
                #if swing phase
                elif step[1] < spike_time/fs < step[2]:
                    list_block = np.arange(step[1], step[2], swing_block_duration)
                    list_block = np.hstack((list_block, step[2]))
                    for i in range(nb_block):
                        if list_block[i] < spike_time/fs < list_block[i+1]:
                            step_swing_count[i] += 1
                # elif spike_time/fs > step[2]:
                #     break
            stance_spike_fq.append(np.array(step_stance_count) / stance_block_duration)
            swing_spike_fq.append(np.array(step_swing_count) / swing_block_duration)

        return stance_spike_fq, swing_spike_fq

    #compute spike frequency during initiation period
    def step_initiation_fq(self, spikes_time, step, nb_block, block_duration, fs):
        list_block = []

        for n in range(nb_block+1):
            list_block.append(step-n*block_duration)
        list_block = sorted(list_block)
        step_spike_count=[]
        for i in range(nb_block):
            step_spike_count.append(0)
        for time in spikes_time:
            if list_block[0] < time/fs < list_block[nb_block]:
                for i in range(nb_block):
                    if list_block[i] < time/fs < list_block[i+1]:
                        step_spike_count[i] += 1
            elif time/fs > list_block[nb_block]:
                break

        return np.array(step_spike_count)/block_duration

#store spikes values and time that match a specific template
class Spikes_cluster:
    def __init__(self, template, number):
        self.template = template
        self.spikes_values = []
        self.spikes_time = []
        self.number = number
        self.delta_time = []

    def dist(self, val):
        dist = 0
        for i in range(len(self.template)):
            dist += (self.template[i] - val[i]) ** 2
        return math.sqrt(dist)

    def add_spike(self, values, time):
        self.spikes_values.append(values)
        self.spikes_time.append(time)

    def compute_delta_time(self):
        l1 = copy.copy(self.spikes_time)
        l2 = copy.copy(self.spikes_time)
        del l1[-1]
        del l2[0]
        self.delta_time = np.array(l2) - np.array(l1)
        return self.delta_time