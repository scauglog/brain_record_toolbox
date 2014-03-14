%original file here : http://www.vis.caltech.edu/~rodri/Wave_clus/Wave_clus_home.htm
clear all;
clc;

dir_name='../../data/r415/'
load(strcat(dir_name,'r415_130926.mat'))
load(strcat(dir_name,'fech.mat'))

signal=d;
clear d;
channel_count=size(signal,1);
%spike frequency = 800 to 2000Hz
f_low=3e2; %min frequency for bandpass filter
f_high=3e3; %max frequency for bandpass filter
b_spike=6;%number of value to save before threshold exceeding
a_spike=20;%number of value to save after threshold exceeding
TDT_padding=0;%number of second before TDT record set in extract script
tresh=-4;
img_ext='png';
spikes_values = cell(channel_count,1);
spikes_time = cell(channel_count,1);
all_templates=cell(channel_count,1);
all_templates_std=cell(channel_count,1);
%%SIGNAL FILTERING
disp('filter signal')
%noise cancellation (the noise is the same on each channel)
%compute the mean value for time t_i and subtract it to all the channel 
average_channels = mean(signal,1); %mean of each column
average_channels=repmat(average_channels,channel_count,1);%repeat matrix vertically
signal=signal-average_channels;


%compute the mean value fore each channel and subtract it to each value
%signal=signal-repmat(mean(signal,2),1,size(signal,2));

%apply Butterworth filter to each channel
%Butterworth filter design
n = 10; Wn = [f_low f_high]/(sampFreq/2);
ftype = 'bandpass';
[B,A] = butter(n,Wn,ftype);
for chan = 1:channel_count
	signal(chan,:) = filtfilt(B,A,double(signal(chan,:)));	
	signalz(chan,:)= zscore(signal(chan,:));
%end

%chan=14; %studied channel

	% X=linspace(0,size(signal,2)/sampFreq,size(signal,2));
	% figure,plot(X,signal(chan,:))
	%%FIND SPIKE
	disp('find spike')
	last_spike=0; %store the last time we found a spike to avoid selecting multiple time the same spike 

	spikes=find(signalz(chan,b_spike:end-a_spike)<tresh);
	last_spike=0;

	for i=1:length(spikes)
		if signalz(spikes(i)) < signalz(spikes(i)+1) && (spikes(i)-last_spike)>a_spike
			last_spike=spikes(i);
			%spikes_values{chan,1}=[spikes_values{chan,1};signalz(chan,(spikes(i)-b_spike):(spikes(i)+a_spike))];
			spikes_values{chan,1}=[spikes_values{chan,1};signal(chan,(spikes(i)-b_spike):(spikes(i)+a_spike))];
		end
	end

	%%QUIROGA
	disp('quiroga')
	
	handles.par.features = 'wav';               %choice of spike features
	handles.par.inputs = 10;                    %number of inputs to the clustering
	handles.par.scales = 4;                     %scales for wavelet decomposition
	if strcmp(handles.par.features,'pca');      %number of inputs to the clustering for pca
		handles.par.inputs=3; 
	end

	handles.par.mintemp = 0;                    %minimum temperature
	handles.par.maxtemp = 0.201;                %maximum temperature
	handles.par.tempstep = 0.01;                %temperature step
	handles.par.num_temp = floor(...
	(handles.par.maxtemp - ...
	handles.par.mintemp)/handles.par.tempstep); %total number of temperatures 
	handles.par.stab = 0.8;                     %stability condition for selecting the temperature
	handles.par.SWCycles = 100;                 %number of montecarlo iterations
	handles.par.KNearNeighb = 11;               %number of nearest neighbors
	handles.par.randomseed = 0;                 % if 0, random seed is taken as the clock value
	%handles.par.randomseed = 147;              % If not 0, random seed   
	handles.par.fname_in = 'tmp_data';          % temporary filename used as input for SPC

	handles.par.min_clus_abs = 20;              %minimum cluster size (absolute value)
	handles.par.min_clus_rel = 0.01;           %minimum cluster size (relative to the total nr. of spikes)
	%handles.par.temp_plot = 'lin';               %temperature plot in linear scale
	handles.par.temp_plot = 'log';              %temperature plot in log scale
	%handles.par.force_auto = 'y';               %automatically force membership if temp>3.
	handles.par.max_spikes = 20;              %maximum number of spikes to plot.

	handles.par.sr = sampFreq

	spikes=spikes_values{chan,1};
	nspk = size(spikes,1);
	handles.par.min_clus = max(handles.par.min_clus_abs,handles.par.min_clus_rel*nspk);

	% CALCULATES INPUTS TO THE CLUSTERING ALGORITHM.
	inspk = wave_features(spikes,handles);   %takes wavelet coefficients.

	% LOAD SPIKES
	handles.par.fname = ['data_' 'temp'];   %filename for interaction with SPC

	%INTERACTION WITH SPC
	save(handles.par.fname_in,'inspk','-ascii');
	[clu, tree] = run_cluster(handles);
	[temp] = find_temp(tree,handles);

	%DEFINE CLUSTERS
	class1=find(clu(temp,3:end)==0);
	class2=find(clu(temp,3:end)==1);
	class3=find(clu(temp,3:end)==2);
	class4=find(clu(temp,3:end)==3);
	class5=find(clu(temp,3:end)==4);
	class0=setdiff(1:size(spikes,1), sort([class1 class2 class3 class4 class5]));
	%classX contains spike index of spike belonging to this class
	img=figure()
	%plot(spikes(class0(1:handles.par.max_spikes),:)','k');
	subplot(1,2,1); 	
	hold on
	subplot(1,2,2); 
	hold on	
	template=[]
	if length(class1) > handles.par.min_clus;
		subplot(1,2,1); 
		plot(spikes(class1(1:handles.par.max_spikes),:)','b');
		template=mean(spikes(class1,:),1);
		template_std=std(spikes(class1,:),0,1);
		subplot(1,2,2); 
		plot(template,'b','linewidth',2);
		plot(template+template_std,['b' '--']);  %plot mean + sd
		plot(template-template_std,['b' '--']);  %plot mean - sd
	end
	if length(class2) > handles.par.min_clus;  
		subplot(1,2,1); 
		plot(spikes(class2(1:handles.par.max_spikes),:)','r');
		template=[template;mean(spikes(class2,:),1)];
		template_std=[template_std;std(spikes(class2,:),0,1)];
		subplot(1,2,2); 
		plot(template(2,:),'r','linewidth',2);
		plot(template(2,:)+template_std(2,:),['r' '--']);  %plot mean + sd
		plot(template(2,:)-template_std(2,:),['r' '--']);  %plot mean - sd
	end
	if length(class3) > handles.par.min_clus-1;  
		subplot(1,2,1); 
		plot(spikes(class3(1:handles.par.max_spikes),:)','g');  
		template=[template;mean(spikes(class3,:),1)];
		template_std=[template_std;std(spikes(class3,:),0,1)];
		subplot(1,2,2); 
		plot(template(3,:),'g','linewidth',2);
		plot(template(3,:)+template_std(3,:),['g' '--']);  %plot mean + sd
		plot(template(3,:)-template_std(3,:),['g' '--']);  %plot mean - sd
	end
	if length(class4) > handles.par.min_clus-1;  
		subplot(1,2,1); 
		plot(spikes(class4(1:handles.par.max_spikes),:)','m');  
		template=[template;mean(spikes(class4,:),1)];
		template_std=[template_std;std(spikes(class4,:),0,1)];
		subplot(1,2,2); 
		plot(template(4,:),'m','linewidth',2);
		plot(template(4,:)+template_std(4,:),['m' '--']);  %plot mean + sd
		plot(template(4,:)-template_std(4,:),['m' '--']);  %plot mean - sd
	end
	if length(class5) > handles.par.min_clus-1;  
		subplot(1,2,1); 
		plot(spikes(class5(1:handles.par.max_spikes),:)','c'); 
		template=[template;mean(spikes(class5,:),1)];
		template_std=[template_std;std(spikes(class5,:),0,1)];
		subplot(1,2,2); 
		plot(template(5,:),'c','linewidth',2);	
		plot(template(5,:)+template_std(5,:),['c' '--']);  %plot mean + sd
		plot(template(5,:)-template_std(5,:),['c' '--']);  %plot mean - sd
	end
	saveas(img,strcat(dir_name,'quiroga_cluster_chan_',num2str(chan)),img_ext);
	all_templates{chan,1}=template
	all_templates_std{chan,1}=template_std
end