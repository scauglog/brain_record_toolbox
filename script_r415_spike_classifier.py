import signal_processing as sig_proc
import pickle
from matplotlib import pyplot as plt
import copy
import random as rnd
import math
import numpy as np

def plot_spikes_classified_mod(spikes_values, spike_count, threshold_template, template, extra_text=''):
    if spike_count > spikes_values.shape[0]:
        spike_count = spikes_values.shape[0]

    fig = plt.figure()
    plt.suptitle('spikes classified mod' + extra_text)
    sample = rnd.sample(xrange(spikes_values.shape[0]), spike_count)
    for r in sample:
        #select a spike randomly
        value = spikes_values[r]

        #select if spikes is in threshold or not
        dst = 0
        for i in range(len(template)):
            dst += (template[i]-value[i])**2
        dst = math.sqrt(dst)
        color_gpe = 'b'
        if dst > threshold_template:
            color_gpe = 'k'
        plt.plot(range(len(value)), value, color=color_gpe)

    if save_img:
        plt.savefig('spikes_classified_mod' + extra_text + img_ext, bbox_inches='tight')
    if not show:
        plt.close(fig)

dir_name = '../data/r415/'

img_ext = '.png'
save_img = False
show = True
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
alpha = 0.1 #learning coef
neighbor = 4 #number of neighbor to modified
min_win = 2 #number of time a neuron should win to be a good neuron
dist_thresh = 4 #distance from which it's acceptable to create a new class

#cluster parameter
min_clus_abs = 20      #minimum cluster size (absolute value)
min_clus_rel = 0.01    #minimum cluster size (relative value)

sp = sig_proc.Signal_processing(save_img, show, img_ext)

#load signal and sampling frequency
#0926
signal = sp.load_m(dir_name + 'r415_131009.mat', 'd')#load multichannel signal
fs = float(sp.load_m(dir_name + 'fech.mat', 'sampFreq')) #load sample frequency

#filter signal
fsignal = sp.signal_mc_filtering(signal, low_cut, high_cut, fs)

all_chan_spikes_values = []
all_chan_koho = []
all_chan_templates = []
for chan in range(fsignal.shape[0]):
#chan = 13
    print('\n\n--- processing chan : ' + str(chan + 1) + ' ---')
    #select the channel and find spikes
    s = fsignal[chan]
    spikes_values, spikes_time = sp.find_spikes(s, a_spike, b_spike, spike_thresh)
    print('spikes found: ' + str(spikes_values.shape[0]))
    spikes_values = sp.smooth_spikes(spikes_values, 3)

    #find mod of the spike and plot
    mod = sp.find_spike_template_mode(spikes_values)
    plot_spikes_classified_mod(spikes_values, 20, 2*dist_thresh, mod, '_channel_' + str(chan+1))
    plt.figure()
    gpe = []
    for value in spikes_values:
        dst = 0
        for v in range(value.shape[0]):
            dst += (value[v]-mod[v]) ** 2
        if math.sqrt(dst) > 2*dist_thresh:
            gpe.append(value)
    gpe = np.array(gpe)
    plt.plot(mod)
    plt.plot(mod, color='b')
    plt.plot(mod - gpe.std(0), '--', color='b')
    plt.plot(mod + gpe.std(0), '--', color='b')
    #plt.close()
    #classify spike using kohonen
    koho = sp.find_spike_template_kohonen(spikes_values, koho_col, koho_row, weight_count, max_weight, alpha, neighbor,
                                          min_win, dist_thresh)
    #store spikes values and corresponding kohonen map
    all_chan_spikes_values.append(spikes_values)
    all_chan_koho.append(koho)
    #koho.plot_network('_channel_'+str(chan+1))
    #keep best cluster aka groups
    min_clus = max(min_clus_abs, min_clus_rel * spikes_values.shape[0])
    koho.evaluate_group(spikes_values, 2 * dist_thresh, min_clus)#keep only groups that have more spike than min_clus

    koho.plot_groups_stat('_channel_' + str(chan + 1)) #to call this you should have call evaluate_group before
    #koho.plot_groups('_channel_'+str(chan+1))
    koho.plot_spikes_classified(spikes_values, 20, 2 * dist_thresh, '_channel_' + str(chan + 1))

    # gpe_list=[]
    # for gpe in koho.groups:
    # gpe_list.append(gpe.template)
    all_chan_templates.append(koho.groups)

if show:
    sp.show_plot()

# save templates in a file
if save_obj:
    with open(dir_name + 'templates', 'wb') as my_file:
        my_pickler = pickle.Pickler(my_file)
        my_pickler.dump(all_chan_templates)

print('\n\n#################')
print('####   END   ####')