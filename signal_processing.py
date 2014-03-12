import scipy.io #open matlab file
import matplotlib.pyplot as plt
#import operator #sorting
import math #abs value
import numpy as np #array and arange
from scipy.signal import butter, filtfilt
from scipy import stats #for zscore calc
import copy #used for list copy same pb as ruby
import kohonen_neuron as nn
import random as rnd #used for plotting spike


class Signal_processing:
	def __init__(self,save=False,show=False,img_ext='.png'):
		self.save=save
		self.show=show
		self.img_ext=img_ext
	#filter a multichannel signal

	def load_m(self,filename,var_name):
		return scipy.io.loadmat(filename)[var_name]

	def signal_mc_filtering(self,signal,lowcut,highcut,fs):
		print('\n### signal filtering ###')
		mean=signal.mean(0) #mean of each column
		mean=np.tile(mean,(signal.shape[0],1)) #duplicate matrix for each row
		signal-=mean

		#apply butterworth filter and zscore
		for chan in range(signal.shape[0]): 
			signal[chan] = self.butter_bandpass_filter(signal[chan], lowcut, highcut, fs, 10)
			signal[chan] = stats.zscore(signal[chan])
		
		return signal
		
	def plot_signal(self,signal):
		plt.figure('signal after filtering')
		plt.suptitle()
		plt.plot(range(signal.shape[0]),signal)
		if self.save:
			plt.savefig('signal_after_filtering'+self.img_ext, bbox_inches='tight')
		if not self.show:
			plt.close()
	def find_spikes(self,signal,a_spike,b_spike,tresh):
		print('\n find spike')
		cpt=0
		last_spike=0
		list=[]
		spikes_values=[]
		spikes_time=[]
		for i in signal:
			#list store x (x=size of the spike length) values of signal
			list.append(i)
			if len(list)>(a_spike+b_spike):
				del list[0]
				if list[b_spike]<tresh and (cpt-last_spike)>a_spike :#and list[b_spike]<list[b_spike+1]:
					spikes_values.append(copy.copy(list))
					spikes_time.append(cpt-a_spike)
					last_spike=cpt
			cpt+=1
		spikes_values=np.array(spikes_values) #can't use directly np.array because it raise an error for first spike (empty array)
		return spikes_values,spikes_time
		
	def smooth_spikes(self,spikes_values,window_size):
		s=[]
		window = np.ones(int(window_size))/float(window_size)
		#window=[1/4.0,2/4.0,1/4.0]
		for val in spikes_values:
			s.append(np.convolve(val,window,'same'))
		return np.array(s)
		
	def classify_spikes(self,spikes_values,templates,threshold_template):
		spikes_classes=[]
		for val in spikes_values:
			gpe=-1
			
			#find best template
			best_dist=threshold_template
			for i in range(len(templates)):
				#compute dist
				dist=0
				for j in range(len(template[i])):
					dist+=(self.template[j]-val[j])**2
				dist=math.sqrt(dist)
				#check if dist is better than previous
				if dist<best_dist:
					color_gpe=i
					best_dist=dist
					
			spikes_classes.append(gpe)
		return spikes_classes
	def plot_spikes(self,spikes_values,spike_count,extra_text=''):
		s=copy.copy(spikes_values).tolist()
		if spike_count>len(s):
			spike_count=len(s)
		
		plt.figure()
		plt.suptitle('spikes find'+extra_text)
		for i in range(spike_count):
			r=rnd.randrange(len(s))
			value=s.pop(r)
			plt.plot(range(len(value)),value)
		
		if self.save:
			plt.savefig('spikes_find'+extra_text+self.img_ext, bbox_inches='tight')
		if not self.show:
			plt.close()
			
	def show_plot(self):
		plt.show()
		
	def find_spike_template_mode(self,spikes_values):
		print('\n## find mode ##')
		self.spike_mode=[]
		for col in spikes_values.transpose():
			values,med=np.histogram(col, bins=15)
			tmp=values.argmax()
			spike_mode.append(med[tmp])

		return self.spike_mode
	
	def find_spike_template_kohonen(self,spikes_values,col,row,weight_count,max_weight,alpha,neighbor,min_win,dist_treshold,ext_img,save,show):
		print('\n## kohonen ##')
		
		self.map=nn.Kohonen(col,row,weight_count,max_weight,alpha,neighbor,min_win,ext_img,save,show)
		iteration_count=0
		i=0
		#while iteration_count<2000:
		#	print(i)
		iteration_count+=spikes_values.shape[0]
		i+=1
		self.map.algo_kohonen(spikes_values)
	
		print('# find best neurons #')
		self.map.best_neurons(spikes_values)
		
		print('# group neurons #')
		self.map.group_neurons(spikes_values,dist_treshold)
		#self.map.find_cluster_center(spikes_values,20)
		return self.map
		
	def butter_bandpass(self,lowcut, highcut, fs, order):
		nyq = 0.5 * fs
		low = lowcut / nyq
		high = highcut / nyq
		b, a = butter(order, [low, high], btype='band')
		return b, a

	def butter_bandpass_filter(self,data, lowcut, highcut, fs, order):
		b, a = self.butter_bandpass(lowcut, highcut, fs, order)
		y = filtfilt(b, a, data)
		return y