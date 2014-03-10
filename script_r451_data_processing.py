import csv
import matplotlib.pyplot as plt
import collections #dictionary
import operator #sorting
import math #abs value
import numpy as np #array and arange

trial_last=14
trial_first=2
exclude_trial=[10,12]
TDT_padding=2 # number of second added before TDT record
pulse_width=0.0002
filter_delta=True
min_delta=0
max_delta=4
#set color drawing
walk_line='r-'
walk_point='rs'
move_line='b-'
move_point='bv'
stim_line='k-'
save=True #save plot as image
img_ext='.eps' #image extension type
show=False #show plot

mydata=collections.OrderedDict()
stim_count=0
max_charge=0
max_amp=100
print('\n### extract data ###')

for trial in range(trial_first,trial_last+1):
	if not(trial in exclude_trial):
		trial_name='Trial'+str(trial).zfill(2)
		filename='TDT/'+trial_name+'.csv' 
		csvfile = open(filename, 'rb')
		data = csv.reader(csvfile, delimiter=',', quotechar='"')
		
		print(trial_name)
		#process data from TDT
		start_stim=[]
		end_stim=[]
		amp_stim=[]
		charge_stim=[]
		charge2_stim=[]
		for row in data:
			#1 start time in tick
			#2 end time in tick
			#3 frequency of stimulation in tick
			#4 number of channel used
			#5 sampling frequency
			if not row[0]=='0':
				start_stim.append(float(row[0])/float(row[5])-TDT_padding) #subtract the padding added to the data
				end_stim.append(float(row[1])/float(row[5])-TDT_padding)
				amp_stim.append(math.fabs(float(row[3])))
				charge_calc=(float(row[1])-float(row[0]))/float(row[2])*pulse_width*math.fabs(float(row[3]))*float(row[4])
				charge_stim.append(charge_calc)
				charge2_stim.append(float(row[4])*pulse_width*math.fabs(float(row[3])))
				stim_count+=1
		
		#process data from vicon
		filename='vicon/'+trial_name+'_EVENTS.csv' 
		csvfile = open(filename, 'rb')
		data = csv.reader(csvfile, delimiter=',', quotechar='"')
		
		events_extraction=False #flag for events extraction
		sync_extraction=False #flag for sync extraction
		cpt=0
		start_move=[]
		start_walk=[]
		end_move=[]
		fq=0 #store frequency sampling
		sync=0 #store beginning of the TDT 
		for row in data:
			move=[trial_name]
			if len(row)==0: #when there is a row with no data means we are at the end of data set for this kind of data
				events_extraction=False
				sync_extraction=False
			elif len(row)==1 and row[0]=='EVENTS':
				events_extraction=True
				cpt=-1 #ignore header
			elif len(row)==1 and row[0]=='ANALOG':
				sync_extraction=True
				cpt=-3 #ignore the 4th first line containing sampling frequency and header
			elif events_extraction and cpt>0:#add time events to the correct list
				if row[2]=='Foot Strike':
					start_move.append(float(row[3]))
				elif row[2]=='Foot Off':
					start_walk.append(float(row[3]))
				elif row[2]=='Event':
					end_move.append(float(row[3]))
			elif sync_extraction and cpt==-2:
				fq=float(row[0])
			elif sync_extraction and cpt>0 and row[11]>1:
				sync=cpt
				break
			cpt+=1
		cpt=0
		
		#synchronise vicon data with the beginning of the TDT
		if not fq==0 and not sync==0:
			for time in start_move:
				start_move[cpt]-=sync/fq
				cpt+=1
			cpt=0	
			for time in start_walk:
				start_walk[cpt]-=sync/fq
				cpt+=1
			cpt=0	
			for time in end_move:
				end_move[cpt]-=sync/fq
				cpt+=1
		#sort time
		start_walk=sorted(start_walk)
		start_move=sorted(start_move)
		end_move=sorted(end_move)
		
		#add data to dictionary
		mydata[trial_name]={'start_stim':start_stim,'end_stim':end_stim,'amp_stim':amp_stim,'charge_stim':charge_stim,'charge2_stim':charge2_stim,'start_move':start_move,'start_walk':start_walk,'end_move':end_move}
print('\n###  process and plot data  ###')

#all data are stored in data dictionary. key is name of the trial
delta_walk=[]
delta_move=[]
charge_move=[]
charge_walk=[]
charge2_move=[]
charge2_walk=[]
charge2_stim=[]
for k,trial in mydata.iteritems():
	print(k)
	#for plotting signal goes up at the start and goes down at the end
	#add stim time value in a list to plot them
	trial_stim=[0]
	trial_stim_binary=[0]
	trial_stim_charge=[0]
	for i in range(len(trial['start_stim'])):
		#just before the beginning of the stim
		trial_stim.append(trial['start_stim'][i])
		trial_stim_binary.append(0)
		trial_stim_charge.append(0)
		#at the beginning of the stim
		trial_stim.append(trial['start_stim'][i])
		trial_stim_binary.append(trial['amp_stim'][i])
		trial_stim_charge.append(0)
		#at the en of the stim
		trial_stim.append(trial['end_stim'][i])
		trial_stim_binary.append(trial['amp_stim'][i])
		trial_stim_charge.append(trial['charge_stim'][i])
		#just after the end of the stim
		trial_stim.append(trial['end_stim'][i])
		trial_stim_binary.append(0)
		trial_stim_charge.append(0)
		
		#add stim charge into the list charge2_stim
		charge2_stim.append(trial['charge2_stim'][i])
	
	#add walk time value in a list to plot them
	trial_walk=[0]
	trial_walk_binary=[0]
	for i in trial['start_walk']:
		#before the beginning of walk
		trial_walk.append(i)
		trial_walk_binary.append(0)
		#at the beginning of walk
		trial_walk.append(i)
		trial_walk_binary.append(1)
		
		#find the first value of end_move superior to the value of start_walk
		for n in trial['end_move']:
			if n>i:
				#at the end of move
				trial_walk.append(n)
				trial_walk_binary.append(1)
				#just after the end of move
				trial_walk.append(n)
				trial_walk_binary.append(0)
				break
	
	#add move time value in a list to plot them	
	trial_move=[0]
	trial_move_binary=[0]
	for i in trial['start_move']:
		#just before the beginning of move
		trial_move.append(i)
		trial_move_binary.append(0)
		#at the beginning of move
		trial_move.append(i)
		trial_move_binary.append(1)
		#find the first value of end_move superior to the value of start_move
		for n in trial['end_move']:
			if n>i:
				#at the end of move
				trial_move.append(n)
				trial_move_binary.append(1)
				#just after the end of move
				trial_move.append(n)
				trial_move_binary.append(0)
				break
				
	#store last time value of each signal in tmp and find the longest signal
	tmp=[]
	tmp.append(trial_stim[-1])
	tmp.append(trial_walk[-1])
	tmp.append(trial_move[-1])
	max=sorted(tmp)[-1]+1
	
	#add a point at the end of the signal in order to have the same length of signal when plotting
	trial_stim.append(max)
	trial_walk.append(max)
	trial_move.append(max)
	
	trial_stim_binary.append(0)
	trial_stim_charge.append(0)
	trial_walk_binary.append(0)
	trial_move_binary.append(0)
	
	#plot the three signal in the same figure 
	plt.figure()
	plt.suptitle(k)
	plt.subplot(411)
	plt.plot(trial_stim, trial_stim_binary, stim_line)
	plt.ylim(-0.5,max_amp)
	plt.xlabel('Time (s)')
	plt.ylabel('Amplitude ($\mu$A)')
	plt.grid(True)

	plt.subplot(412)
	plt.plot(trial_stim, trial_stim_charge, stim_line)
	#plt.ylim(-0.5,max_charge)
	plt.xlabel('Time (s)')
	plt.ylabel('Charge ($\mu$C)')
	plt.grid(True)
	
	plt.subplot(413)
	plt.plot(trial_move, trial_move_binary, move_line)
	plt.ylim(-0.5,1.5)
	plt.xlabel('Time (s)')
	plt.ylabel('Move')
	plt.grid(True)
	
	plt.subplot(414)
	plt.plot(trial_walk, trial_walk_binary, walk_line)
	plt.ylim(-0.5,1.5)
	plt.xlabel('Time (s)')
	plt.ylabel('Walk')
	plt.grid(True)
	if show:
		plt.show()
	if save:
		plt.savefig(k+img_ext, bbox_inches='tight')
	
	#calculate delta time between the start of the movement and the beginning of stimulation
	for i in trial['start_move']:
		stim_before=0
		stim_after=0
		index_stim=0
		for n in trial['start_stim']:
			stim_after=n
			if n>i:
				break
			stim_before=n
			index_stim+=1
		#find the minimal difference  in case the movement begin before the stim (bad tracking case)
		if not(stim_before==0 and stim_after==0):
			if math.fabs(i-stim_before)<math.fabs(i-stim_after):#use squared difference
				delta_move.append(i-stim_before)
				delta_stim=trial['end_stim'][index_stim]-stim_before
				delta_charge=trial['charge_stim'][index_stim]
			else:
				delta_move.append(i-stim_after)
				delta_stim=1
				delta_charge=0
		#compute slope of charge time curve
		a=delta_charge/delta_stim
		#compute charge when the movement start
		charge_move.append(a*delta_move[-1])
				
	for i in trial['start_walk']:
		stim_before=0
		stim_after=0
		index_stim=0
		for n in trial['start_stim']:
			stim_after=n
			if n>i:
				break
			stim_before=n
			index_stim+=1
		#find the minimal difference  in case the movement begin before the stim (bad tracking case)
		if not(stim_before==0 and stim_after==0):
			if math.fabs(i-stim_before)<math.fabs(i-stim_after):
				delta_walk.append(i-stim_before)
				delta_stim=trial['end_stim'][index_stim]-stim_before
				delta_charge=trial['charge_stim'][index_stim]
			else:
				delta_walk.append(i-stim_after)
				delta_stim=1
				delta_charge=0
		#compute slope of charge time curve
		a=delta_charge/delta_stim
		#compute charge when the walk start
		charge_walk.append(a*delta_walk[-1])
	
	#determine if walk append during stim
	for i in range(len(trial['start_stim'])):
		tmp_walk=False
		tmp_move=False
		for d in trial['start_walk']:
			if d>trial['start_stim'][i] and d<trial['end_stim'][i]:
				tmp_walk=True
				break
		for d in trial['start_move']:
			if d>trial['start_stim'][i] and d<trial['end_stim'][i]:
				tmp_move=True
				break
		if tmp_walk:
			charge2_walk.append(True)
		else:
			charge2_walk.append(False)
		if tmp_move:
			charge2_move.append(True)
		else:
			charge2_move.append(False)
				
print('\n###  plot delta time  ###')
#sort value of delta_walk and move				
sorted_delta_walk = sorted(delta_walk)
sorted_delta_move = sorted(delta_move)
#filter delta
if filter_delta:
	sorted_delta_walk=filter(lambda x: x>=min_delta and x<=max_delta,sorted_delta_walk)
	sorted_delta_move=filter(lambda x: x>=min_delta and x<=max_delta,sorted_delta_move)
		
#find max delta to draw horizontal line
if sorted_delta_walk[-1] < sorted_delta_move[-1]:
	max_delta=sorted_delta_move[-1]
else:
	max_delta=sorted_delta_walk[-1]
#find min delta to draw horizontal line
if sorted_delta_walk[0] < sorted_delta_move[0]:
	min_delta=sorted_delta_walk[0]
else:
	min_delta=sorted_delta_move[0]
#find max value count to draw vertical line
if len(sorted_delta_walk)> len(sorted_delta_move):
	max_value_count=len(sorted_delta_walk)
else:
	max_value_count=len(sorted_delta_move)
#value for the y axis
n_walk = np.array(range(1, len(sorted_delta_walk)+1))
n_move = np.array(range(1, len(sorted_delta_move)+1))
p_walk = n_walk/float(stim_count) #%of walking rat maybe greater than 100% if the number of walk is superior to the number of stimulation
p_move = n_move/float(stim_count)


#plot point
plt.figure()
plt.plot(sorted_delta_walk, n_walk, walk_point, label="walk")
plt.plot(sorted_delta_move, n_move, move_point, label="move")
#draw horizontal line
plt.plot([min_delta,max_delta],[stim_count,stim_count],'k-')
#draw vertical line
plt.plot([0, 0], [0, max_value_count], 'k-')
plt.xlabel('Delta Time (s)')
plt.ylabel('Number of started move')
plt.grid(True)
plt.legend(loc='upper left')
if show:
	plt.show()
if save:
	plt.savefig('num_delta_time'+img_ext, bbox_inches='tight')
	
#plot with % value
plt.figure()
plt.plot(sorted_delta_walk, p_walk, walk_point, label="walk")
plt.plot(sorted_delta_move, p_move, move_point, label="move")
#draw vertical line
plt.plot([0, 0], [0, max_value_count/float(stim_count)], 'k-')
#draw horizontal line
plt.plot([min_delta,max_delta],[1,1],'k-')
plt.xlabel('Delta Time (s)')
plt.ylabel('% of started move')
plt.grid(True)
plt.legend(loc='upper left')
#save the plot image
if show:
	plt.show()
if save:
	plt.savefig('perc_delta_time'+img_ext, bbox_inches='tight')

print('### stat ###')
average_delta_move=sum(delta_move)/float(len(delta_move))
average_delta_walk=sum(delta_walk)/float(len(delta_walk))

stderror_move=0
stderror_walk=0
for value in delta_move:
	stderror_move+=(value-average_delta_move)**2
for value in delta_walk:
	stderror_walk+=(value-average_delta_walk)**2

stderror_move/=float(len(delta_move))
stderror_walk/=float(len(delta_walk))

stderror_move=math.sqrt(stderror_move)
stderror_walk=math.sqrt(stderror_walk)

print('average delta move: '+str(average_delta_move))
print('stderror delta move: '+str(stderror_move))
print('\naverage delta walk: '+str(average_delta_walk))
print('stderror delta walk: '+str(stderror_walk))

#boxplot charge
data_boxplot=[delta_move,delta_walk]

plt.figure()
plt.boxplot(data_boxplot)
plt.xticks([1,2],['move', 'walk'])
plt.ylabel('delta time (s)')
if show:
	plt.show()
if save:
	plt.savefig('boxplot_delta_time'+img_ext, bbox_inches='tight')
	
print('\n### plot charge ###')
data_boxplot=[charge_move,charge_walk]	
plt.figure()
plt.boxplot(data_boxplot)
plt.xticks([1,2],['move','walk'])
if show:
	plt.show()
if save:
	plt.savefig('boxplot_charge'+img_ext,bbox_inches='tight')

print('### stat ###')
average_charge_move=sum(charge_move)/float(len(charge_move))
average_charge_walk=sum(charge_walk)/float(len(charge_walk))

stderror_move=0
stderror_walk=0
for value in delta_move:
	stderror_move+=(value-average_charge_move)**2
for value in delta_walk:
	stderror_walk+=(value-average_charge_walk)**2

stderror_move/=float(len(charge_move))
stderror_walk/=float(len(charge_walk))

stderror_move=math.sqrt(stderror_move)
stderror_walk=math.sqrt(stderror_walk)

print('average charge for move: '+str(average_charge_move))
print('stderror charge for move: '+str(stderror_move))
print('\naverage charge for walk: '+str(average_charge_walk))
print('stderror charge for walk: '+str(stderror_walk))

print('### plot charge ###')
charge2_n=range(1,len(charge2_stim)+1)
plot_charge_move=[]
plot_charge_move_n=[]
plot_charge_walk=[]
plot_charge_walk_n=[]
plot_charge_noreaction=[]
plot_charge_noreaction_n=[]
#make three plot one for no reaction one for move one for walk
for k in range(len(charge2_move)):
	if charge2_walk[k]:
		plot_charge_walk.append(charge2_stim[k])
		plot_charge_walk_n.append(charge2_n[k])
	elif charge2_move[k]:
		plot_charge_move.append(charge2_stim[k])
		plot_charge_move_n.append(charge2_n[k])
	else:
		plot_charge_noreaction.append(charge2_stim[k])
		plot_charge_noreaction_n.append(charge2_n[k])

plt.figure()
#plt.plot(charge2_n,charge2_stim, 'kv', label="no reaction")
plt.plot(plot_charge_noreaction_n,plot_charge_noreaction,'kx',label="no reaction")
plt.plot(plot_charge_move_n,plot_charge_move, move_point, label="move")
plt.plot(plot_charge_walk_n,plot_charge_walk, walk_point, label="walk")
plt.xlabel('stim number')
plt.ylabel('charge ($\mu$C)')
plt.grid(True)
plt.legend(loc='upper left')
#save the plot image
if show:
	plt.show()
if save:
	plt.savefig('charge_stim'+img_ext, bbox_inches='tight')

print('\n\n#################')
print('####   END   ####')