#process all record
import pickle
import signal_processing as sig_proc

dir_name='../data/r415/'

img_ext='.png'
save_img=False
show=False
save_obj=False

#signal filtering parameter
low_cut=3e2
high_cut=3e3

#spike finding parameters
b_spike=6
a_spike=20
spike_thresh=-4

threshold_template=4 #distance from which it's acceptable to put spike in class
base_name='r415_'
record_name=['130926','131008','131009','131011','131016','131017','131018','131021','131023','131025','131030','131101','131118','131129']
record_data={}
with open(dir_name+'templates','rb') as my_file:
	all_chan_templates=pickle.load(my_file)

sp=sig_proc.Signal_processing(save_img,show,img_ext)
		
#for record in record_name:	
record=record_name[0]
print('----- processing record: '+record+' -----')
signal=sp.load_m(dir_name+base_name+record+'.mat','d')#load multichannel signal
fs=float(sp.load_m(dir_name+'fech.mat','sampFreq')) #load sample frequency

fsignal=sp.signal_mc_filtering(signal,low_cut,high_cut,fs)

all_chan_spikes_values=[]
all_chan_spikes_times=[]
all_chan_spikes_classes=[]
all_chan_clusters=[]
#create cluster for storing classed spikes inside
print('### create cluster storing spikes_values and spikes_time ###')
for chan in range(len(all_chan_templates)):
	all_chan_clusters.append([])
	for gpe in all_chan_templates[chan]:
		all_chan_clusters[chan].append(sig_proc.Spikes_cluster(gpe.template,gpe.number+1))
for chan in range(fsignal.shape[0]):
	print('\n\n--- processing chan: '+str(chan+1)+' ---')
	s=fsignal[chan]
	spikes_values,spikes_time=sp.find_spikes(s,a_spike,b_spike,spike_thresh)
	spikes_values=sp.smooth_spikes(spikes_values,3)
	print('spikes found: ' + str(spikes_values.shape[0]))
	spikes_classes=sp.classify_spikes(spikes_values,spikes_time,all_chan_clusters[chan],2*threshold_template)
	
	all_chan_spikes_classes.append(spikes_classes)
	all_chan_spikes_times.append(spikes_time)
	all_chan_spikes_values.append(spikes_values)
	sp.plot_signal_spikes_classified(spikes_time,spikes_classes,base_name+'_'+record+'_'+str(chan)+'_')
	
	# unclass_spikes=[]
	# for i in range(len(spikes_classes)):
		# if spikes_classes[i]==-1:
			# unclass_spikes.append(spikes_values[i])
	# sp.plot_spikes(np.array(unclass_spikes),20,'chan_'+str(chan))
	
	#plt.show()
record_data[record]={'spikes_values':all_chan_spikes_values,'spikes_time':all_chan_spikes_times,'spikes_classes':all_chan_spikes_classes,'clusters':all_chan_clusters}

if save_obj:
	with open(dir_name+'data_processed','wb') as my_file:
		my_pickler= pickle.Pickler(my_file)
		my_pickler.dump(record_data)
	