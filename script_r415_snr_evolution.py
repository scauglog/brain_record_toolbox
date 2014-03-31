import pickle
import signal_processing as sig_proc
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import copy
#step phase analysis for each neuron and global

dir_name = '../data/r448/r448_131022_rH/'

img_ext = '.eps'
save_img = True
show = True

trials = [2, 5, 6, 7]
#signal filtering parameter
low_cut = 3e2
high_cut = 3e3

sp = sig_proc.Signal_processing(save_img, show, img_ext)
global_snr = []

dir_name = '../data/r415/'
base_name = 'r415_'
record_name = ['130926', '131008', '131009', '131011', '131016', '131017', '131018', '131021', '131023', '131025',
               '131030', '131101', '131118', '131129']

print('### spikes load ###')
with open(dir_name + 'data_processed', 'rb') as my_file:
    record_data = pickle.load(my_file)
# record_data[trial] = {'spikes_values': all_chan_spikes_values,
#                       'spikes_time': all_chan_spikes_times,
#                       'spikes_classes': all_chan_spikes_classes,
#                       'clusters': all_chan_clusters,
#                       'length_signal': signal.shape[1],
#                       'fs': fs }
for record in record_name:
    signal = sp.load_m(dir_name + 'r415_'+record+'.mat', 'd')
    fs = float(sp.load_m(dir_name + 'fech.mat', 'sampFreq'))

    signal_noise_ratio_r415 = []
    fsignal = sp.signal_mc_filtering(signal, low_cut, high_cut, fs)

    for chan in range(len(record_data[record]['clusters'])):
        sig_mean = np.array(fsignal[chan]).mean()
        sig_std = np.array(fsignal[chan]).std()
        min_sig = sig_mean-2*sig_std
        max_sig = sig_mean+2*sig_std

        for cluster in record_data[record]['clusters'][chan]:
            if np.array(cluster.spikes_values).shape[0]>0:
                max_spike = np.array(cluster.spikes_values).max(1).mean()
                min_spike = np.array(cluster.spikes_values).min(1).mean()
                signal_noise_ratio_r415.append((max_spike-min_spike)/(max_sig-min_sig))
            else:
                signal_noise_ratio_r415.append(0)
    global_snr.append(copy.copy(signal_noise_ratio_r415))



plt.figure()
plt.boxplot(global_snr)
plt.plot(np.array(global_snr).mean(1))
if save_img:
    plt.savefig('box_plot_snr_r415'+img_ext, bbox_inches='tight')
if show:
    plt.show()
else:
    plt.close()