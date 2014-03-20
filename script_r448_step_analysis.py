import pickle
import signal_processing as sig_proc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import copy

dir_name = '../data/r448/r448_131022_rH/'

img_ext = '.png'
save_img = False
show = False
# tdt padding in second
TDT_padding = 2
sp = sig_proc.Signal_processing(save_img, show, img_ext)
# number of delta between spike to take for mean comparison to determine wich step it is
trials = range(5 - 1, 6)
nb_block = 8

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
global_step=[]
global_strike=[]
global_off=[]
for trial in trials:
    print('#### Trial ' + str(trial + 1) + ' ####')
    print('### vicon data process ###')
    filename = 'p0_3RW0' + str(trial + 1)
    file_events = sp.load_csv(dir_name + filename + '_EVENTS.csv')
    file_analog = sp.load_csv(dir_name + filename + '_ANALOG.csv')

    vicon_data[trial] = sp.vicon_extract(file_events)
    vicon_data[trial] = sp.vicon_extract(file_analog, copy.copy(vicon_data[trial]))
    vicon_data[trial] = sp.synch_vicon_with_TDT(vicon_data[trial], TDT_padding)

    vicon_data[trial]['strike_times'], vicon_data[trial]['strike_bin'] = sp.binarise_vicon_step(vicon_data[trial]['Right']['Foot Strike'])
    vicon_data[trial]['off_times'], vicon_data[trial]['off_bin'] = sp.binarise_vicon_step(vicon_data[trial]['Right']['Foot Off'])
    vicon_data[trial]['push_steps_time'] = sp.find_step_time(vicon_data[trial]['strike_times'], vicon_data[trial]['off_times'])
    vicon_data[trial]['off_steps_time'] = sp.find_step_time(vicon_data[trial]['off_times'], vicon_data[trial]['strike_times'])
    fs = record_data[trial]['fs']
    full_step = sp.find_full_step_time( vicon_data[trial]['push_steps_time'],vicon_data[trial]['off_steps_time'])

    plt.figure()
    for chan in range(len(record_data[trial]['clusters'])):
        my_data[chan]={}
        for cluster in record_data[trial]['clusters'][chan]:
            my_data[chan][cluster.number] = {'strike_spike_fq': [], 'off_spike_fq': [], 'spike_fq':[]}

            for step in full_step:
                strike_block_duration = (step[1]-step[0])/nb_block
                off_block_duration = (step[2]-step[1])/nb_block
                step_strike_count = []
                step_off_count = []
                for i in range(nb_block):
                    step_strike_count.append(0)
                    step_off_count.append(0)

                for spike_time in cluster.spikes_time:
                    if step[0] < spike_time/fs < step[1]:
                        list_block = np.arange(step[0], step[1], strike_block_duration)
                        list_block = np.hstack((list_block, step[1]))
                        for i in range(nb_block):
                            if list_block[i] < spike_time/fs < list_block[i+1]:
                                step_strike_count[i] += 1
                    elif step[1] < spike_time/fs < step[2]:
                        list_block = np.arange(step[1], step[2], off_block_duration)
                        list_block = np.hstack((list_block, step[2]))
                        for i in range(nb_block):
                            if list_block[i] < spike_time/fs < list_block[i+1]:
                                step_off_count[i] += 1
                    elif spike_time/fs > step[2]:
                        break
                fq_strike = np.array(step_strike_count) / strike_block_duration
                fq_off = np.array(step_off_count) / off_block_duration
                fq_spike = np.hstack((fq_strike, fq_off))
                my_data[chan][cluster.number]['strike_spike_fq'].append(fq_strike)
                my_data[chan][cluster.number]['off_spike_fq'].append(fq_off)
                my_data[chan][cluster.number]['spike_fq'].append(np.hstack((fq_strike, fq_off)))
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(np.array(my_data[chan][cluster.number]['spike_fq']).mean(0))
            global_step.append(np.array(my_data[chan][cluster.number]['spike_fq']).mean(0))
            plt.subplot(3, 1, 2)
            plt.plot(np.array(my_data[chan][cluster.number]['strike_spike_fq']).mean(0))
            global_strike.append(np.array(my_data[chan][cluster.number]['strike_spike_fq']).mean(0))
            plt.subplot(3, 1, 3)
            plt.plot(np.array(my_data[chan][cluster.number]['off_spike_fq']).mean(0))
            global_off.append(np.array(my_data[chan][cluster.number]['off_spike_fq']).mean(0))
            plt.savefig('step_modulation_chan' + str(chan + 1) + '_cluster_' + str(cluster.number) +'_trial'+str(trial+1)+ img_ext, bbox_inches='tight')
            #plt.show()
            plt.close()
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.array(global_step).mean(0))
    plt.subplot(3, 1, 2)
    plt.plot(np.array(global_strike).mean(0))
    plt.subplot(3, 1, 3)
    plt.plot(np.array(global_off).mean(0))
    plt.savefig('global_modulation_trial'+str(trial+1)+img_ext, bbox_inches='tight')
    plt.close()
