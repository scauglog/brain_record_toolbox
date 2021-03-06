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
show = False
# tdt padding in second
TDT_padding = 2
sp = sig_proc.Signal_processing(save_img, show, img_ext)
# number of delta between spike to take for mean comparison to determine wich step it is
trials = [2, 5, 6, 7]
nb_block = 4

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

trial=trials[0]
for chan in range(len(record_data[trial]['clusters'])):
        my_data[chan]={}
        for cluster in record_data[trial]['clusters'][chan]:
            my_data[chan][cluster.number] = {'stance_spike_fq': [], 'swing_spike_fq': [], 'step_spike_fq': []}

for trial in trials:
    print('#### Trial ' + str(trial) + ' ####')
    print('### vicon data process ###')
    filename = 'p0_3RW0' + str(trial)
    file_events = sp.load_csv(dir_name + filename + '_EVENTS.csv')
    file_analog = sp.load_csv(dir_name + filename + '_ANALOG.csv')

    vicon_data[trial] = sp.vicon_extract(file_events)
    vicon_data[trial] = sp.vicon_extract(file_analog, copy.copy(vicon_data[trial]))
    vicon_data[trial] = sp.synch_vicon_with_TDT(vicon_data[trial], TDT_padding)

    vicon_data[trial]['stance_steps_time'] = sp.find_step_time(vicon_data[trial]['Right']['Foot Strike'], vicon_data[trial]['Right']['Foot Off'])
    vicon_data[trial]['swing_steps_time'] = sp.find_step_time(vicon_data[trial]['Right']['Foot Off'], vicon_data[trial]['Right']['Foot Strike'])
    fs = record_data[trial]['fs']
    full_step = sp.find_full_step_time(vicon_data[trial]['stance_steps_time'], vicon_data[trial]['swing_steps_time'])
    print(len(full_step))
    for chan in range(len(record_data[trial]['clusters'])):
        # my_data[chan]={}
        for cluster in record_data[trial]['clusters'][chan]:
            stance_spike_fq, swing_spike_fq = sp.phase_step_spike_fq(cluster.spikes_time, full_step, nb_block, fs)
            step_spike_fq = np.hstack((stance_spike_fq, swing_spike_fq))
            for val in stance_spike_fq:
                my_data[chan][cluster.number]['stance_spike_fq'].append(copy.copy(val))
            for val in swing_spike_fq:
                my_data[chan][cluster.number]['swing_spike_fq'].append(copy.copy(val))
            for val in step_spike_fq:
                my_data[chan][cluster.number]['step_spike_fq'].append(copy.copy(val))


global_step=[]
tot_neuron=0
tot_neuron_modulate=0
for chan in my_data:
    for cluster in my_data[chan]:
        tot_neuron+=1
        step_spike_fq = np.array(my_data[chan][cluster]['step_spike_fq'])

        max_col = np.argmax(step_spike_fq.mean(0))
        min_col = np.argmin(step_spike_fq.mean(0))

        v_max = step_spike_fq[:, max_col]
        v_min = step_spike_fq[:, min_col]



        h_stat, pvalue = stats.wilcoxon(v_max,v_min)
        if pvalue < 0.05 and v_max.mean()-v_min.mean() > 10 and step_spike_fq.shape[0] > 10:
            tot_neuron_modulate += 1
        else:
            print('not chan '+str(chan+1)+' cluster '+str(cluster))
        plt.figure()
        plt.plot(step_spike_fq.mean(0))
        plt.plot((step_spike_fq.mean(0)-step_spike_fq.std(0)/math.sqrt(len(step_spike_fq))), 'b--')
        plt.plot((step_spike_fq.mean(0)+step_spike_fq.std(0)/math.sqrt(len(step_spike_fq))), 'b--')
        global_step.append(step_spike_fq.mean(0))
        if save_img:
                plt.savefig('step_modulation_chan' + str(chan + 1) + '_cluster_' + str(cluster)+ img_ext, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

print tot_neuron_modulate
plt.figure()
plt.plot(np.array(global_step).mean(0))
plt.plot(np.array(global_step).mean(0)-(np.array(global_step).std(0)/math.sqrt(len(global_step))), 'b--')
plt.plot(np.array(global_step).mean(0)+(np.array(global_step).std(0)/math.sqrt(len(global_step))), 'b--')
if save_img:
    plt.savefig('global_modulation_'+img_ext, bbox_inches='tight')
if show:
    plt.show()
else:
    plt.close()
