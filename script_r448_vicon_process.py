import pickle
import signal_processing as sig_proc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import copy

dir_name = '../data/r448/r448_131022_rH/'

img_ext = '.eps'
save_img = True
show = False
# tdt padding in second
TDT_padding = 2
block_duration = 0.5
sp = sig_proc.Signal_processing(save_img, show, img_ext)
# maximum time for a step if the step is longer than max_step_time
max_step_time = 1
# number of delta between spike to take for mean comparison to determine wich step it is
window_size = 3
trials = [2, 5, 6, 7]


print('### spikes load ###')
# fs = float(sp.load_m(dir_name + 'fech.mat', 'sampFreq'))  #load TDT sample frequency
# signals = sp.load_m(dir_name + 'cell_trial.mat', 't')  #load multichannel signal
with open(dir_name + 'data_processed', 'rb') as my_file:
    record_data = pickle.load(my_file)
# record_data[trial] = {'spikes_values': all_chan_spikes_values,
#                       'spikes_time': all_chan_spikes_times,
#                       'spikes_classes': all_chan_spikes_classes,
#                       'clusters': all_chan_clusters,
#                       'length_signal': signal.shape[1],
#                       'fs': fs }

vicon_data = {}
for trial in trials:
    print('#### Trial ' + str(trial) + ' ####')
    print('### vicon data process ###')
    filename = 'p0_3RW0' + str(trial)
    file_events = sp.load_csv(dir_name + filename + '_EVENTS.csv')
    file_analog = sp.load_csv(dir_name + filename + '_ANALOG.csv')

    vicon_data[trial] = sp.vicon_extract(file_events)
    vicon_data[trial] = sp.vicon_extract(file_analog, copy.copy(vicon_data[trial]))
    vicon_data[trial] = sp.synch_vicon_with_TDT(vicon_data[trial], TDT_padding)

    vicon_data[trial]['strike_times'], vicon_data[trial]['strike_bin'] = sp.binarise_vicon_step(vicon_data[trial]['Right']['Foot Strike'])
    vicon_data[trial]['off_times'], vicon_data[trial]['off_bin'] = sp.binarise_vicon_step(vicon_data[trial]['Right']['Foot Off'])
    vicon_data[trial]['stance_steps_time'] = sp.find_step_time(vicon_data[trial]['strike_times'], vicon_data[trial]['off_times'])
    vicon_data[trial]['swing_steps_time'] = sp.find_step_time(vicon_data[trial]['off_times'], vicon_data[trial]['strike_times'])

print('### data process ###')
#init the data_dict cause he shouldn't be reseted between each trial
trial = trials[0]
all_chan_cluster = record_data[trial]['clusters']
data_dict = {}
for chan in range(len(all_chan_cluster)):
    data_dict[chan] = {}
    for cluster in all_chan_cluster[chan]:
        data_dict[chan][cluster.number] = {'swing_delta_all_step': [], 'stance_delta_all_step': [], 'swing_delta_each_step': [], 'stance_delta_each_step': [], 'all_delta': []}

for trial in trials:
    print('#### Trial ' + str(trial + 1) + ' ####')
    #init variable
    fs = record_data[trial]['fs']
    length_signal = record_data[trial]['length_signal']
    strike_times = vicon_data[trial]['strike_times']
    strike_bin = vicon_data[trial]['strike_bin']
    off_times = vicon_data[trial]['off_times']
    off_bin = vicon_data[trial]['off_bin']
    all_chan_cluster = record_data[trial]['clusters']
    stance_steps_time = vicon_data[trial]['stance_steps_time']
    swing_steps_time = vicon_data[trial]['swing_steps_time']

    print('### compute fire rate ###')
    #cut signal in block and calculate the number of spike that appear during the block length
    # global_fire is the cumulative count of spike for each block
    all_chan_firerates, global_fire = sp.fire_rate(all_chan_cluster, length_signal, fs, block_duration)

    #some plot
    sp.plot_all_chan_firerates(all_chan_firerates, strike_times, strike_bin, off_times, off_bin, length_signal, fs, block_duration, '_trial' + str(trial + 1))
    sp.plot_global_firerate(global_fire, strike_times, strike_bin, off_times, off_bin, length_signal, fs, block_duration, '_trial' + str(trial + 1))

    print('### find stance and swing delta time ###')
    #for each step in every trial put the delta time between two spike that append during the step in the data_dict

    for chan in range(len(all_chan_cluster)):
        for cluster in all_chan_cluster[chan]:

            cluster.compute_delta_time()
            stance_delta_each_step, stance_delta_all_step = sp.find_delta_time_step(stance_steps_time, cluster.spikes_time, cluster.delta_time, fs, max_step_time)
            swing_delta_each_step, swing_delta_all_step = sp.find_delta_time_step(swing_steps_time, cluster.spikes_time, cluster.delta_time, fs, max_step_time)

            data_dict[chan][cluster.number]['all_delta'] += cluster.delta_time.tolist()
            data_dict[chan][cluster.number]['stance_delta_each_step'] += stance_delta_each_step
            data_dict[chan][cluster.number]['stance_delta_all_step'] += stance_delta_all_step
            data_dict[chan][cluster.number]['swing_delta_each_step'] += swing_delta_each_step
            data_dict[chan][cluster.number]['swing_delta_all_step'] += swing_delta_all_step

print('### find interresting cluster to determine step ###')
good_chan_cluster = {}
for chan in range(len(data_dict)):
    for cluster in data_dict[chan]:
        #print('--- chan: ' + str(chan+1) + ' cluster: ' + str(cluster)+' ---')
        all_delta = data_dict[chan][cluster]['all_delta']
        stance_delta = data_dict[chan][cluster]['stance_delta_all_step']
        swing_delta = data_dict[chan][cluster]['swing_delta_all_step']
        all_vs_stance, all_vs_swing, stance_vs_swing = sp.cluster_step_ttest(all_delta, stance_delta, swing_delta)
        #store p value
        data_dict[chan][cluster]['all_vs_stance'] = all_vs_stance
        data_dict[chan][cluster]['all_vs_swing'] = all_vs_swing
        data_dict[chan][cluster]['stance_vs_swing'] = stance_vs_swing

        #selecting interesting channel
        if data_dict[chan][cluster]['all_vs_stance'] < 0.05:
            if chan not in good_chan_cluster:
                good_chan_cluster[chan] = {}
            good_chan_cluster[chan][cluster] = {}
            good_chan_cluster[chan][cluster]['stance'] = np.mean(data_dict[chan][cluster]['stance_delta_all_step'])
            good_chan_cluster[chan][cluster]['all'] = np.mean(data_dict[chan][cluster]['all_delta'])

        if data_dict[chan][cluster]['all_vs_swing'] < 0.05:
            if chan not in good_chan_cluster:
                good_chan_cluster[chan] = {}
            good_chan_cluster[chan][cluster] = {}
            good_chan_cluster[chan][cluster]['swing'] = np.mean(data_dict[chan][cluster]['swing_delta_all_step'])
            good_chan_cluster[chan][cluster]['all'] = np.mean(data_dict[chan][cluster]['all_delta'])

        if data_dict[chan][cluster]['stance_vs_swing'] < 0.05:
            if chan not in good_chan_cluster:
                good_chan_cluster[chan] = {}
            good_chan_cluster[chan][cluster] = {}
            good_chan_cluster[chan][cluster]['stance'] = np.mean(data_dict[chan][cluster]['stance_delta_all_step'])
            good_chan_cluster[chan][cluster]['swing'] = np.mean(data_dict[chan][cluster]['swing_delta_all_step'])

print good_chan_cluster

#for each interesting chan find which step is the spike
for trial in trials:
    print('#### Trial ' + str(trial + 1) + ' ####')
    stance_steps_time = vicon_data[trial]['stance_steps_time']
    swing_steps_time = vicon_data[trial]['swing_steps_time']
    fs = record_data[trial]['fs']
    for chan in good_chan_cluster:
        vicon_data[trial][chan] = {}
        for cluster in good_chan_cluster[chan]:
            #print('\n\n--- channel ' + str(chan) + ' cluster '+ str(cluster) + '---')
            #store dict variable in local variable for easy reading
            spikes_time = record_data[trial]['clusters'][chan][cluster - 1].spikes_time
            delta_time = record_data[trial]['clusters'][chan][cluster - 1].delta_time
            all_mean = good_chan_cluster[chan][cluster]['all']
            stance_mean = good_chan_cluster[chan][cluster]['stance']
            swing_mean = good_chan_cluster[chan][cluster]['swing']
            #classify spike
            swing_spikes_time, stance_spikes_time, other_spikes_time, spike_is_step = sp.class_spike_in_step(spikes_time, delta_time, all_mean, stance_mean, swing_mean, window_size, fs)

            #stance stat
            correct_stance, false_stance, missing_stance = sp.step_spike_error_stat(stance_steps_time, stance_spikes_time)
            #swing stat
            correct_swing, false_swing, missing_swing = sp.step_spike_error_stat(swing_steps_time, swing_spikes_time)

            #store result
            vicon_data[trial][chan][cluster] = {}
            vicon_data[trial][chan][cluster]['stance_spikes_time'] = stance_spikes_time
            vicon_data[trial][chan][cluster]['swing_spikes_time'] = swing_spikes_time
            vicon_data[trial][chan][cluster]['other_spikes_time'] = other_spikes_time
            vicon_data[trial][chan][cluster]['spike_is_step'] = spike_is_step
            vicon_data[trial][chan][cluster]['correct_stance'] = correct_stance
            vicon_data[trial][chan][cluster]['false_stance'] = false_stance
            vicon_data[trial][chan][cluster]['missing_stance'] = missing_stance
            vicon_data[trial][chan][cluster]['correct_swing'] = correct_swing
            vicon_data[trial][chan][cluster]['false_swing'] = false_swing
            vicon_data[trial][chan][cluster]['missing_swing'] = missing_swing
            vicon_data[trial][chan][cluster]['spikes_time'] = spikes_time

            print('\n\ncorrect stance: ' + str(correct_stance))
            print('false stance:   ' + str(false_stance))
            print('missing stance: ' + str(missing_stance))
            print('\ncorrect swing:  ' + str(correct_swing))
            print('false swing:    ' + str(false_swing))
            print('missing swing:  ' + str(missing_swing))

            strike_times = vicon_data[trial]['strike_times']
            strike_bin = vicon_data[trial]['strike_bin']
            off_times = vicon_data[trial]['off_times']
            off_bin = vicon_data[trial]['off_bin']
            fs = record_data[trial]['fs']
            length_signal = record_data[trial]['length_signal']
            spikes_time = record_data[trial]['clusters'][chan][cluster-1].spikes_time

            sp.plot_step_spike_classify(strike_times, strike_bin, off_times, off_bin, spike_is_step, spikes_time, length_signal, fs, '_chan'+str(chan)+'_cluster'+str(cluster)+'_trial'+str(trial))


#probe all interesting chan to estimate which step his this spike
for trial in trials:
    poll_block = []
    length_signal = record_data[trial]['length_signal']
    fs = record_data[trial]['fs']
    block_duration = 0.5
    length_signal_s = length_signal/fs
    block_list = np.arange(0, length_signal_s, block_duration)

    for i in range(len(block_list)-1):
        start_time = block_list[i]
        end_time = block_list[i+1]
        poll_stance = 0
        poll_swing = 0
        poll_none = 0
        for chan in good_chan_cluster:
            for cluster in good_chan_cluster[chan]:
                spike_is_step = vicon_data[trial][chan][cluster]['spike_is_step']
                spikes_time = vicon_data[trial][chan][cluster]['spikes_time']
                for j in range(len(spike_is_step)):
                    if start_time < spikes_time[j]/fs < end_time:
                        if spike_is_step[j] == 0:
                            poll_none += 1
                        elif spike_is_step[j] == 1:
                            poll_stance += 1
                        elif spike_is_step[j] == -1:
                            poll_swing += 1
        res = max(poll_none, poll_stance, poll_swing)
        if poll_none == res:
            poll_block.append(0)
        elif poll_stance == res:
            poll_block.append(1)
        elif poll_swing == res:
            poll_block.append(-1)

    strike_times = vicon_data[trial]['strike_times']
    strike_bin = vicon_data[trial]['strike_bin']
    off_times = vicon_data[trial]['off_times']
    off_bin = vicon_data[trial]['off_bin']

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(strike_times, strike_bin)
    plt.plot(off_times, off_bin)
    plt.xlim(0, int(length_signal_s) + 1)
    plt.subplot(2, 1, 2)
    plt.xlim(0, int((length_signal_s) / block_duration) + 1)
    plt.plot(range(len(poll_block)), poll_block)
    plt.show()
print('\n\n#################')
print('####   END   ####')