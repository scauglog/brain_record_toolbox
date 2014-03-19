import signal_processing as sig_proc
import pickle
import numpy as np

dir_name = '../data/r448/r448_131022_rH/'

img_ext = '.png'
save_img = True
show = False
save_obj = True

#signal filtering parameter
low_cut = 3e2
high_cut = 3e3

#spike finding parameters
b_spike = 6
a_spike = 20
spike_thresh = -4

#kohonen parameter
koho_col = 4
koho_row = 8
weight_count = a_spike + b_spike
max_weight = spike_thresh * 2
#learning coef
alpha = 0.1
#number of neighbor to modified
neighbor = 4
#number of time a neuron should win to be a good neuron
min_win = 2
#distance from which it's acceptable to create a new class
dist_thresh = 4

#cluster parameter
#minimum cluster size (absolute value)
min_clus_abs = 20
#minimum cluster size (relative value)
min_clus_rel = 0.01

#trial to analyse
trials = range(5 - 1, 7)

#where to store data
record_data = {}

print('### load signal ###')
sp = sig_proc.Signal_processing(save_img, show, img_ext)

#load signal and sampling frequency
signals = sp.load_m(dir_name + 'cell_trial.mat', 't')
fs = float(sp.load_m(dir_name + 'fech.mat', 'sampFreq'))


trial = trials[0]
print('---- trial: ' + str(trial + 1) + ' ----')
signal = np.transpose(signals[0][trial])
fsignal = sp.signal_mc_filtering(signal, low_cut, high_cut, fs)

#all_chan_koho = []
#first we find templates using kohonen network
all_chan_templates = []
for chan in range(fsignal.shape[0]):
    #chan = 13
    print('\n\n--- processing chan : ' + str(chan + 1) + ' ---')
    s = fsignal[chan]
    spikes_values, spikes_time = sp.find_spikes(s, a_spike, b_spike, spike_thresh)
    print('spikes found: ' + str(spikes_values.shape[0]))
    spikes_values = sp.smooth_spikes(spikes_values, 3)
    koho = sp.find_spike_template_kohonen(spikes_values, koho_col, koho_row, weight_count, max_weight, alpha,
                                          neighbor, min_win, dist_thresh)
    #keep best cluster aka groups
    min_clus = max(min_clus_abs, min_clus_rel * spikes_values.shape[0])

    #keep only groups that have more spike than min_clus
    koho.evaluate_group(spikes_values, 2 * dist_thresh, min_clus)

    #to call plot you should have call evaluate_group before
    txt='_trial_' + str(trial) + '_channel_' + str(chan + 1)
    koho.plot_groups_stat(txt)
    koho.plot_spikes_classified(spikes_values, 20, 2 * dist_thresh, txt)

    all_chan_templates.append(koho.groups)
if show:
    sp.show_plot()

#class spikes for each trial
for trial in trials:
    print('---- trial: ' + str(trial + 1) + ' ----')
    signal = np.transpose(signals[0][trial])
    fsignal = sp.signal_mc_filtering(signal, low_cut, high_cut, fs)

    all_chan_spikes_values = []
    all_chan_spikes_times = []
    all_chan_clusters = []
    for chan in range(fsignal.shape[0]):
        #chan = 13
        print('\n\n--- processing chan : ' + str(chan + 1) + ' ---')
        s = fsignal[chan]
        spikes_values, spikes_time = sp.find_spikes(s, a_spike, b_spike, spike_thresh)
        print('spikes found: ' + str(spikes_values.shape[0]))
        spikes_values = sp.smooth_spikes(spikes_values, 3)
        all_chan_spikes_values.append(spikes_values)

        #classify spike
        all_chan_clusters.append([])
        for gpe in all_chan_templates[chan]:
            all_chan_clusters[chan].append(sig_proc.Spikes_cluster(gpe.template, gpe.number + 1))
        #spikes_classes = sp.classify_spikes(spikes_values, spikes_time, all_chan_clusters[chan], 2 * dist_thresh)

        #all_chan_spikes_classes.append(spikes_classes)
        all_chan_spikes_times.append(spikes_time)
        all_chan_spikes_values.append(spikes_values)

    if show:
        sp.show_plot()

    record_data[trial] = {'spikes_values': all_chan_spikes_values, 'spikes_time': all_chan_spikes_times,
                          'clusters': all_chan_clusters, 'length_signal': signal.shape[1], 'fs': fs}

if save_obj:
    with open(dir_name + 'data_processed', 'wb') as my_file:
        my_pickler = pickle.Pickler(my_file)
        my_pickler.dump(record_data)

print('\n\n#################')
print('####   END   ####')