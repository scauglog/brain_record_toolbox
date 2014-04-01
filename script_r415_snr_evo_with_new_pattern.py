#process all record
#for each record find template for spike in each channel
#then compute signal to noise ratio (snr) for each cluster
import pickle
import signal_processing as sig_proc
import numpy as np
import copy
import matplotlib.pyplot as plt

dir_name = '../data/r415/'

img_ext = '.png'
save_img = True
show = False
save_obj = False

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
alpha = 0.1     #learning coef
neighbor = 4    #number of neighbor to modified
min_win = 2     #number of time a neuron should win to be a good neuron
dist_thresh = 5 #distance from which it's acceptable to create a new class

#cluster parameter
min_clus_abs = 20      #minimum cluster size (absolute value)
min_clus_rel = 0.01    #minimum cluster size (relative value)


threshold_template = 4 #distance from which it's acceptable to put spike in class
base_name = 'r415_'
record_name = ['130926', '131008', '131009', '131011', '131016', '131017', '131018', '131021', '131023', '131025',
               '131030', '131101', '131118', '131129']
record_data = {}
with open(dir_name + 'templates', 'rb') as my_file:
    all_chan_templates = pickle.load(my_file)

sp = sig_proc.Signal_processing(save_img, show, img_ext)

global_snr = []
for record in record_name:
# record = record_name[0]
    print('----- processing record: ' + record + ' -----')
    signal = sp.load_m(dir_name + base_name + record + '.mat', 'd')#load multichannel signal
    fs = float(sp.load_m(dir_name + 'fech.mat', 'sampFreq')) #load sample frequency

    fsignal = sp.signal_mc_filtering(signal, low_cut, high_cut, fs)

    signal_noise_ratio_r415 = []
    for chan in range(fsignal.shape[0]):
        print('\n\n--- processing chan : ' + str(chan + 1) + ' ---')
        s = fsignal[chan]
        #signal dispersion
        sig_mean = np.array(fsignal[chan]).mean()
        sig_std = np.array(fsignal[chan]).std()
        min_sig = sig_mean-2*sig_std
        max_sig = sig_mean+2*sig_std

        #find spike using threshold and smooth them
        spikes_values, spikes_time = sp.find_spikes(s, a_spike, b_spike, spike_thresh)
        print('spikes found: ' + str(spikes_values.shape[0]))
        spikes_values = sp.smooth_spikes(spikes_values, 3)

        #find template for spikes
        koho = sp.find_spike_template_kohonen(spikes_values, koho_col, koho_row, weight_count, max_weight, alpha, neighbor,
                                              min_win, dist_thresh)
        #keep best cluster aka groups
        min_clus = max(min_clus_abs, min_clus_rel * spikes_values.shape[0])
        koho.evaluate_group(spikes_values, 2 * dist_thresh, min_clus)#keep only groups that have more spike than min_clus

        for group in koho.groups:
            if np.array(group.spikes).shape[0]>0:
                max_spike = np.array(group.spikes).max(1).mean()
                min_spike = np.array(group.spikes).min(1).mean()
                signal_noise_ratio_r415.append((max_spike-min_spike)/(max_sig-min_sig))
            else:
                signal_noise_ratio_r415.append(0)
    global_snr.append(copy.copy(signal_noise_ratio_r415))

box_plot=[]


plt.figure()
plt.boxplot(global_snr)

#compute mean snr for each experiment and plot
snr_mean = []
for l in global_snr:
    snr_mean.append(np.array(l).mean())
plt.plot(snr_mean)

if save_img:
    plt.savefig('box_plot_snr_r415_new'+img_ext, bbox_inches='tight')
if show:
    plt.show()
else:
    plt.close()


print('\n\n#################')
print('####   END   ####')