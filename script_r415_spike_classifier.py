import signal_processing as sig_proc

img_ext='.png'
save=False
show=True

#signal filtering parameter
low_cut=3e2
high_cut=5e3

#spike finding parameters
b_spike=6
a_spike=20
spike_thresh=-4

#kohonen parameter
koho_col=1
koho_row=20
weight_count=a_spike+b_spike
max_weight=spike_thresh*2
alpha=0.1
neighbor=4
min_win=10 #number of time a neuron should win to be a good neuron
dist_thresh=3.5 #distance from which it's acceptable to create a new class


sp=sig_proc.Signal_processing(save,img_ext)

#load signal and sampling frequency
signal=sp.load_m('r415_130926.mat','d')#load multichannel signal
fs=float(sp.load_m('fech.mat','sampFreq')) #load sample frequency

fsignal=sp.signal_mc_filtering(signal,low_cut,high_cut,fs)

all_chan_spikes_values=[]
all_chan_koho=[]
#for chan in range(fsignal.shape[0]):
chan=1
print('\n\n--- processing chan : '+str(chan+1)+' ---')
s=fsignal[chan]
spikes_values=sp.find_spikes(s,a_spike,b_spike,spike_thresh)
spikes_values=sp.smooth_spikes(spikes_values,3)
koho=sp.find_spike_template_kohonen(spikes_values,koho_col,koho_row,weight_count,max_weight,alpha,neighbor,min_win,dist_thresh,img_ext,save)
all_chan_spikes_values.append(spikes_values)
all_chan_koho.append(koho)
#sp.plot_spikes(spikes_values,20,'_channel_'+str(chan+1))
#koho.plot_best_neurons('_channel_'+str(chan+1))
koho.plot_network('_channel_'+str(chan+1))
koho.plot_groups('_channel_'+str(chan+1))
koho.plot_spikes_classified(spikes_values,20,7,'_channel_'+str(chan+1))
koho.evaluate_group(spikes_values,50,7,10)
koho.plot_groups('_channel_'+str(chan+1))
koho.plot_spikes_classified(spikes_values,20,7,'_channel_'+str(chan+1))

if show:	
	sp.show_plot()