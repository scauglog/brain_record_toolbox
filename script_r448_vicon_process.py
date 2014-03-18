import pickle
import signal_processing as sig_proc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

dir_name = '../data/r448/r448_131022_rH/'

img_ext = '.png'
save_img = False
show = True
TDT_padding = 2  #tdt padding in second
block_duration = 0.5
sp = sig_proc.Signal_processing(save_img, show, img_ext)

trials = range(5 - 1, 7)
# for trial in trials:


print('### spikes load ###')
fs = float(sp.load_m(dir_name + 'fech.mat', 'sampFreq'))  #load TDT sample frequency
signals = sp.load_m(dir_name + 'cell_trial.mat', 't')  #load multichannel signal
with open(dir_name + 'data_processed', 'rb') as my_file:
    record_data = pickle.load(my_file)
# record_data[trial] = {'spikes_values': all_chan_spikes_values,
#                       'spikes_time': all_chan_spikes_times,
#                       'spikes_classes': all_chan_spikes_classes,
#                       'clusters': all_chan_clusters}

#for trial in trials:
trial = trials[0]
print('#### Trial ' + str(trial + 1) + ' ####')
print('### vicon data process ###')
filename = 'p0_3RW0' + str(trial + 1)
file_events = sp.load_csv(dir_name + filename + '_EVENTS.csv')
file_analog = sp.load_csv(dir_name + filename + '_ANALOG.csv')
data = sp.vicon_extract(file_events)
data = sp.vicon_extract(file_analog, data)
data = sp.synch_vicon_with_TDT(data, TDT_padding)

strike_times, strike_bin = sp.binarise_vicon_step(data['Right']['Foot Strike'])
off_times, off_bin = sp.binarise_vicon_step(data['Right']['Foot Off'])

data_spike = record_data[trial]
signal = np.transpose(signals[0][trial])
length_signal = signal.shape[1]
print('### data process ###')
#cut signal in block and calculate the number of spike that appear durring the block length
# global_fire is the cumulative count of spike for each block
all_chan_firerates, global_fire = sp.fire_rate(data_spike['clusters'], length_signal, fs, block_duration)

# sp.plot_all_chan_firerates(all_chan_firerates, strike_times, strike_bin, off_times, off_bin, length_signal, fs,
#                            block_duration, '_trial' + str(trial + 1))
# sp.plot_global_firerate(global_fire, strike_times, strike_bin, off_times, off_bin, length_signal, fs, block_duration,
#                         '_trial' + str(trial + 1))

print('### find push and off dspike ###')

data_dict = {}
stat_dict = {}
for chan in range(len(data_spike['clusters'])):
    data_dict[chan] = {}
    stat_dict[chan] = {}
    for cluster in data_spike['clusters'][chan]:
        data_dict[chan][cluster.number] = {'off_delta_all_step': [], 'push_delta_all_step': [], 'off_delta_each_step': [], 'push_delta_each_step': [], 'all_delta': []}
        stat_dict[chan][cluster.number] = {'all_vs_push': 0, 'all_vs_off': 0, 'push_vs_off': 0}

for trial in trials:
    data_spike = record_data[trial]
    print('#### Trial ' + str(trial + 1) + ' ####')
    for chan in range(len(data_spike['clusters'])):
        for cluster in data_spike['clusters'][chan]:
            cluster.compute_delta_time()
            data_dict[chan][cluster.number]['all_delta'] += cluster.delta_time.tolist()
            # for each strike (from foot strike to foot off) add delta time between spike into clus_push_spike
            clus_push_dspike_each_step = []
            for i in range(len(strike_times) - 1):
                off = filter(lambda x: strike_times[i] < x < strike_times[i + 1], off_times)
                if not len(off) == 0:
                    off = off[0]
                    strike = strike_times[i]
                    # exclude too long move
                    if off - strike < 1:
                        delta_list = []
                        for j in range(len(cluster.spikes_time) - 1):
                            if strike < cluster.spikes_time[j] / fs < off and strike < cluster.spikes_time[j + 1] / fs < off:
                                data_dict[chan][cluster.number]['push_delta_all_step'].append(cluster.delta_time[j])
                                delta_list.append(cluster.delta_time[j])
                        data_dict[chan][cluster.number]['push_delta_each_step'].append(delta_list)

            for i in range(len(off_times) - 1):
                strike = filter(lambda x: off_times[i] < x < off_times[i + 1], strike_times)
                if not len(strike) == 0:
                    strike = strike[0]
                    off = off_times[i]
                    # exclude too long move
                    if strike - off < 1:
                        delta_list = []
                        for j in range(len(cluster.spikes_time) - 1):
                            if off < cluster.spikes_time[j] / fs < strike and off < cluster.spikes_time[
                                        j + 1] / fs < strike:
                                data_dict[chan][cluster.number]['off_delta_all_step'].append(cluster.delta_time[j])
                                delta_list.append(cluster.delta_time[j])
                        data_dict[chan][cluster.number]['off_delta_each_step'].append(delta_list)

print('### some stat ###')
for chan in range(len(data_spike['clusters'])):
    for cluster in data_spike['clusters'][chan]:
            if not len(data_dict[chan][cluster.number]['all_delta']) == 0 and not len(data_dict[chan][cluster.number]['push_delta_all_step']) == 0:
                t, p = stats.ttest_ind(data_dict[chan][cluster.number]['all_delta'], data_dict[chan][cluster.number]['push_delta_all_step'], equal_var=False)
                stat_dict[chan][cluster.number]['all_vs_push'] = p

            if not len(data_dict[chan][cluster.number]['all_delta']) == 0 and not len(data_dict[chan][cluster.number]['off_delta_all_step']) == 0:
                t, p = stats.ttest_ind(data_dict[chan][cluster.number]['all_delta'], data_dict[chan][cluster.number]['off_delta_all_step'], equal_var=False)
                stat_dict[chan][cluster.number]['all_vs_off'] = p

            if not len(data_dict[chan][cluster.number]['off_delta_all_step']) == 0 and not len(data_dict[chan][cluster.number]['push_delta_all_step']) == 0:
                t, p = stats.ttest_ind(data_dict[chan][cluster.number]['off_delta_all_step'], data_dict[chan][cluster.number]['push_delta_all_step'], equal_var=False)
                stat_dict[chan][cluster.number]['push_vs_off'] = p

import pprint
pp = pprint.PrettyPrinter(depth=4)
pp.pprint(stat_dict)

sp.show_plot()

print('\n\n#################')
print('####   END   ####')