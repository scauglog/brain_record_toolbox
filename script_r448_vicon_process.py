import pickle
import signal_processing as sig_proc
import numpy as np

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
for trial in trials:
    print('#### Trial ' + str(trial + 1) + ' ####')
    print('### vicon data process ###')
    #trial = trials[0]
    filename = 'p0_3RW0' + str(trial + 1)
    file_events = sp.load_csv(dir_name + filename + '_EVENTS.csv')
    file_analog = sp.load_csv(dir_name + filename + '_ANALOG.csv')
    data = sp.vicon_extract(file_events)
    data = sp.vicon_extract(file_analog, data)
    data = sp.synch_vicon_with_TDT(data, TDT_padding)

    strike_time, strike_bin = sp.binarise_vicon_step(data['Right']['Foot Strike'])
    off_time, off_bin = sp.binarise_vicon_step(data['Right']['Foot Off'])

    data_spike = record_data[trial]
    signal = np.transpose(signals[0][trial])
    length_signal = signal.shape[1]
    print('### data process ###')
    all_chan_firerates, global_fire = sp.fire_rate(data_spike['clusters'], length_signal, fs, block_duration)
    sp.plot_all_chan_firerates(all_chan_firerates, strike_time, strike_bin, off_time, off_bin, length_signal, fs,
                               block_duration, '_trial' + str(trial + 1))
    sp.plot_global_firerate(global_fire, strike_time, strike_bin, off_time, off_bin, length_signal, fs, block_duration,
                            '_trial' + str(trial + 1))
    sp.show_plot()
    #walk
    # walk = sorted(data['Right']['Foot Strike'] + data['Right']['Foot Off'])
    # start_walk = data['General']['Foot Strike']
    # end_walk = []
    # for time in start_walk:
    #     l1 = filter(lambda x: x < time, data['Right']['Foot Strike'])
    #     l2 = filter(lambda x: x < time, data['Right']['Foot Off'])
    #     if len(l1) > 0 and len(l2) > 0:
    #         end_walk.append(max(l1 + l2))
    # end_walk.append(max(data['Right']['Foot Strike'], data['Right']['Foot Off']))


    # for time in range(int((length_signal / fs) / block_duration) + 1):
    #     global_fire.append(0)
    # for chan in range(len(data_spike['clusters'])):
    # print('---- channel: ' + str(chan + 1) + ' ----')
    # fq = []
    # for clu in data_spike['clusters'][chan]:
    #     l = []
    #     for time in range(int((length_signal / fs) / block_duration) + 1):
    #         l.append(0)
    #     for time in clu.spikes_time:
    #         l[int((time / fs) / block_duration)] += 1
    #         global_fire[int((time / fs) / block_duration)] += 1
    #     fq.append(l)
    # all_chan_firerates.append(fq)
    #
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(strike_time, strike_bin)
    # plt.plot(off_time, off_bin)
    # plt.xlim(0, int(length_signal / fs) + 1)
    # plt.subplot(2, 1, 2)
    # plt.xlim(0, int((length_signal / fs) / block_duration) + 1)
    # for list_times in fq:
    #     plt.plot(list_times)
    #
    # if save_img:
    #     plt.savefig('walk_firerate_correlation_trial'+str(trial+1)+'_chan' + str(chan + 1) + img_ext, bbox_inches='tight')
    # if show:
    #     plt.show()
    # else:
    #     plt.close()
    #
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(strike_time, strike_bin)
    # plt.plot(off_time, off_bin)
    # plt.xlim(0, int(length_signal / fs) + 1)
    # plt.subplot(2, 1, 2)
    # plt.xlim(0, int((length_signal / fs) / block_duration) + 1)
    # plt.plot(global_fire)
    #
    # if save_img:
    #     plt.savefig('walk_firerate_correlation_trial'+str(trial+1)+'_global'+ img_ext, bbox_inches='tight')
    # if show:
    #     plt.show()
    # else:
    #     plt.close()

print('\n\n#################')
print('####   END   ####')