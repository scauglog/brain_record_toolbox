import signal_processing as sig_proc
import pickle

img_ext='.png'
save_img=False
show=True
save_obj=False

#signal filtering parameter
low_cut=3e2
high_cut=3e3

#spike finding parameters
b_spike=6
a_spike=20
spike_thresh=-4

#kohonen parameter
koho_col=4
koho_row=8
weight_count=a_spike+b_spike
max_weight=spike_thresh*2
alpha=0.1 #learning coef
neighbor=4 #number of neighbor to modified
min_win=2 #number of time a neuron should win to be a good neuron
dist_thresh=4 #distance from which it's acceptable to create a new class

#cluster parameter
min_clus_abs = 20;      #minimum cluster size (absolute value)
min_clus_rel = 0.01; 	#minimum cluster size (relative value)

sp=sig_proc.Signal_processing(save_img,show,img_ext)

#load signal and sampling frequency
signal=sp.load_m('r415_131009.mat','d')#load multichannel signal
fs=float(sp.load_m('fech.mat','sampFreq')) #load sample frequency

fsignal=sp.signal_mc_filtering(signal,low_cut,high_cut,fs)

all_chan_spikes_values=[]
all_chan_koho=[]
all_chan_templates=[]
for chan in range(fsignal.shape[0]):
#chan=13
	print('\n\n--- processing chan : '+str(chan+1)+' ---')
	s=fsignal[chan]
	spikes_values,spikes_time=sp.find_spikes(s,a_spike,b_spike,spike_thresh)
	print('spikes found: ' + str(spikes_values.shape[0]))
	spikes_values=sp.smooth_spikes(spikes_values,3)
	koho=sp.find_spike_template_kohonen(spikes_values,koho_col,koho_row,weight_count,max_weight,alpha,neighbor,min_win,dist_thresh,img_ext,save_img,show)
	all_chan_spikes_values.append(spikes_values)
	all_chan_koho.append(koho)
	#koho.plot_network('_channel_'+str(chan+1))
	#keep best cluster aka groups
	min_clus=max(min_clus_abs,min_clus_rel*spikes_values.shape[0])
	koho.evaluate_group(spikes_values,2*dist_thresh,min_clus)#keep only groups that have more spike than min_clus

	koho.plot_groups_stat('_channel_'+str(chan+1)) #to call this you should have call evaluate_group before
	#koho.plot_groups('_channel_'+str(chan+1))
	koho.plot_spikes_classified(spikes_values,20,2*dist_thresh,'_channel_'+str(chan+1))

	# gpe_list=[]
	# for gpe in koho.groups:
		# gpe_list.append(gpe.template)
	all_chan_templates.append(koho.groups)

if show:	
	sp.show_plot()
	
# save templates in a file
if save_obj:
	with open('templates','wb') as my_file:
		my_pickler= pickle.Pickler(my_file)
		my_pickler.dump(all_chan_templates)