import pickle
import signal_processing as sig_proc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import copy

#step initiation frequency analysis
dir_name = '../data/r448/r448_131022_rH/'

img_ext = '.png'
save_img = True
show = False
# tdt padding in second
TDT_padding = 2
sp = sig_proc.Signal_processing(save_img, show, img_ext)
# number of delta between spike to take for mean comparison to determine wich step it is
trials = [2, 5, 6, 7]
nb_block = 5
block_duration = 0.4
first_step_good = {2: [1, 2], 5: [1, 2]}

print('### spikes load ###')
with open(dir_name + 'data_processed', 'rb') as my_file:
    record_data = pickle.load(my_file)
# record_data[trial] = {'spikes_values': all_chan_spikes_values,
#                       'spikes_time': all_chan_spikes_times,
#                       'spikes_classes': all_chan_spikes_classes,
#                       'clusters': all_chan_clusters,
#                       'length_signal': signal.shape[1],
#                       'fs': fs }

vicon_data = {}
my_data={}
global_count=[]
for trial in trials:
    print('#### Trial ' + str(trial) + ' ####')
    print('### vicon data process ###')
    filename = 'p0_3RW0' + str(trial)
    file_events = sp.load_csv(dir_name + filename + '_EVENTS.csv')
    file_analog = sp.load_csv(dir_name + filename + '_ANALOG.csv')

    vicon_data[trial] = sp.vicon_extract(file_events)
    vicon_data[trial] = sp.vicon_extract(file_analog, copy.copy(vicon_data[trial]))
    vicon_data[trial] = sp.synch_vicon_with_TDT(vicon_data[trial], TDT_padding)

    fs = record_data[trial]['fs']
    first_step = vicon_data[trial]['General']['Foot Strike']
    first_step_block = []
    my_data[trial]={}
    for step in range(len(first_step)):
        if trial in first_step_good and step+1 in first_step_good[trial]:
            list_block = []
            for n in range(nb_block+1):
                list_block.append(first_step[step]-n*block_duration)
            first_step_block.append(sorted(list_block))

    for chan in range(len(record_data[trial]['clusters'])):
        my_data[trial][chan]={}
        for cluster in record_data[trial]['clusters'][chan]:
            my_data[trial][chan][cluster.number] = {}
            my_data[trial][chan][cluster.number]['spike_count_first_step'] = []

            for step in range(len(first_step)):
                if trial in first_step_good and step+1 in first_step_good[trial]:
                    global_count.append(sp.step_initiation_fq(cluster.spikes_time, first_step[step], nb_block, block_duration, fs))

plt.figure()
plt.plot(np.array(global_count).mean(0))
if save_img:
    plt.savefig('before_first_step_modulation_global_trial_' + img_ext, bbox_inches='tight')
if show:
    plt.show()
else:
    plt.close()