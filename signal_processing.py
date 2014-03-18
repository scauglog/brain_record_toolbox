import scipy.io  #open matlab file
import matplotlib.pyplot as plt
import math  #abs value
import numpy as np  #array and arange
from scipy.signal import butter, filtfilt
from scipy import stats  #for zscore calc
import copy  #used for list copy same pb as ruby
import kohonen_neuron as nn
import random as rnd  #used for plotting spike
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

    #filter multichannel filter subtract col mean -> butterworth -> zscore
    def signal_mc_filtering(self, signal, lowcut, highcut, fs):
        print('\n### signal filtering ###')
        mean = signal.mean(0)  #mean of each column
        mean = np.tile(mean, (signal.shape[0], 1))  #duplicate matrix for each row
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
        list = []
        spikes_values = []
        spikes_time = []
        for i in signal:
            #list store x (x=size of the spike length) values of signal
            list.append(i)
            if len(list) > (a_spike + b_spike):
                del list[0]
                if list[b_spike] < tresh and (cpt - last_spike) > a_spike:  #and list[b_spike]<list[b_spike+1]:
                    spikes_values.append(copy.copy(list))
                    spikes_time.append(cpt - a_spike)
                    last_spike = cpt
            cpt += 1
        spikes_values = np.array(
            spikes_values)  #can't use directly np.array because it raise an error for first spike (empty array)
        return spikes_values, spikes_time

    def smooth_spikes(self, spikes_values, window_size):
        s = []
        window = np.ones(int(window_size)) / float(window_size)
        #window=[1/4.0,2/4.0,1/4.0]
        for val in spikes_values:
            s.append(np.convolve(val, window, 'same'))
        return np.array(s)

    def classify_spikes(self, spikes_values, spikes_time, clusters, threshold_template):
        spikes_classes = []
        classed_count = 0
        for i in range(len(spikes_time)):
            best_clus = self.find_best_cluster(spikes_values[i], clusters, threshold_template)
            if best_clus != None:
                best_clus.add_spike(spikes_values[i], spikes_time[i])
                classed_count += 1
                spikes_classes.append(best_clus.number)
            else:
                spikes_classes.append(-1)
        print ('spike classified: ' + str(classed_count))
        print ('spike unclassified: ' + str(len(spikes_values) - classed_count))
        return spikes_classes

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

    def show_plot(self):
        if self.show:
            plt.show()

    #find the pattern of spikes using mode
    def find_spike_template_mode(self, spikes_values):
        print('\n## find mode ##')
        self.spike_mode = []
        for col in spikes_values.transpose():
            values, med = np.histogram(col, bins=15)
            tmp = values.argmax()
            self.spike_mode.append(med[tmp])

        return self.spike_mode

    #find patterns of spikes using kohonen network
    def find_spike_template_kohonen(self, spikes_values, col, row, weight_count, max_weight, alpha, neighbor, min_win,
                                    dist_treshold):
        print('\n## kohonen ##')

        self.map = nn.Kohonen(col, row, weight_count, max_weight, alpha, neighbor, min_win, self.img_ext, self.save_img,
                              self.show)
        iteration_count = 0
        i = 0
        #while iteration_count<2000:
        #	print(i)
        iteration_count += spikes_values.shape[0]
        i += 1
        self.map.algo_kohonen(spikes_values)

        print('# find best neurons #')
        self.map.best_neurons(spikes_values)

        print('# group neurons #')
        self.map.group_neurons(spikes_values, dist_treshold)
        #self.map.find_cluster_center(spikes_values,20)
        return self.map

    #compute parameters for butterworth filter
    def butter_bandpass(self, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    #filter signal using butterworth filter and filtfilt
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order)
        y = filtfilt(b, a, data)
        return y

    def load_csv(self, filename):
        csvfile = open(filename, 'rb')
        return csv.reader(csvfile, delimiter=',', quotechar='"')

    def vicon_extract(self, data, dict={}):
        events = ['Foot Strike', 'Foot Off', 'Event']
        context = ['Right', 'Left', 'General']
        events_extraction = False  #flag for events extraction
        sync_extraction = False  #flag for sync extraction
        cpt = 0
        #if no dictionnary is passed create a new dict
        if context[0] not in dict:
            for c in context:
                dict[c] = {}
                for e in events:
                    dict[c][e] = []
            dict['fq'] = 0  #store frequency sampling
            dict['sync'] = 0  #store beginning of the TDT
        for row in data:
            if len(
                    row) == 0:  #when there is a row with no data means we are at the end of data set for this kind of data
                events_extraction = False
                sync_extraction = False
            elif len(row) == 1 and row[0] == 'EVENTS':
                events_extraction = True
                cpt = -1  #ignore header
            elif len(row) == 1 and row[0] == 'ANALOG':
                sync_extraction = True
                cpt = -3  #ignore the 4th first line containing sampling frequency and header
            elif events_extraction and cpt > 0:  #add time events to the correct list
                dict[row[1]][row[2]].append(float(row[3]))
            elif sync_extraction and cpt == -2:
                dict['fq'] = float(row[0])
            elif sync_extraction and cpt > 0 and float(row[11]) > 1:
                dict['sync'] = float(row[0])
                break
            cpt += 1
        return dict

    def synch_vicon_with_TDT(self, dict, TDT_padding=0):
        #synchronise vicon data with the beginning of the TDT
        events = ['Foot Strike', 'Foot Off', 'Event']
        context = ['Right', 'Left', 'General']
        for c in context:
            for e in events:
                for time in dict[c][e]:
                    time -= dict['sync'] / dict['fq']
                    time += TDT_padding

                dict[c][e] = sorted(dict[c][e])
        return dict

    def binarise_vicon_step(self, steps):
        time = [0]
        bin = [0]
        for val in steps:
            time.append(val)
            bin.append(0)
            time.append(val)
            bin.append(1)
            time.append(val)
            bin.append(0)

        return time, bin

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

    def plot_global_firerate(self, global_fire, strike_time, strike_bin, off_time, off_bin, length_signal, fs,
                             block_duration, extra_txt=''):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(strike_time, strike_bin)
        plt.plot(off_time, off_bin)
        plt.xlim(0, int(length_signal / fs) + 1)
        plt.subplot(2, 1, 2)
        plt.xlim(0, int((length_signal / fs) / block_duration) + 1)
        plt.plot(global_fire)

        if self.save_img:
            plt.savefig('walk_firerate_correlation_global' + extra_txt + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

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
                plt.plot(list_times)

        if self.save_img:
            plt.savefig('walk_firerate_correlation_trial_chan' + str(chan + 1) + extra_txt + self.img_ext,
                        bbox_inches='tight')
        if not self.show:
            plt.close()


#store spikes values and time that match a specific template
class Spikes_cluster:
    def __init__(self, template, number):
        self.template = template
        self.spikes_values = []
        self.spikes_time = []
        self.number = number

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