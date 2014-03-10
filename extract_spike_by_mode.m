clear all;
clc;

load('r415_130926.mat')
load('fech.mat')

% k=1
% filename=strcat('Trial',num2str(k,'%02d'),'.csv')
% dlmwrite(filename, results(:,:,k), 'precision', '%i')

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

spikes_values = cell(channel_count,1);
spikes_time = cell(channel_count,1);

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
end

chan=16; %studied channel

X=linspace(0,size(signal,2)/sampFreq,size(signal,2));
figure,plot(X,signal(chan,:))

last_spike=0; %store the last time we found a spike to avoid selecting multiple time the same spike 


for i = b_spike:size(signalz,2)- a_spike
	if signalz(chan,i) < tresh && (i-last_spike)>a_spike %select spike when value is below threshold and when it's the first time
		spikes_values{chan,1}=[spikes_values{chan,1};signalz(chan,(i-b_spike):(i+a_spike))];%add the spike below the last spike found
		spikes_time{chan,1}=[spikes_time{chan,1};(i/sampFreq)-TDT_padding];% delay before signal (synch)
		last_spike=i;
	end
end
spike_mode=[];
%take the mod of each time point
%iterate over each column with for 
i=1;
for val = spikes_values{chan}
	[high,med_val] = hist(val,15);         
	spike_mode(chan,i) = med_val(find(max(high),1,'last'),1);
	i=i+1;
end

figure,plot(spike_mode(chan,:),'r')

clear MSE
good_spikes_values=cell(channel_count,1);
good_spikes_time=cell(channel_count,1);

for i = 1:size(spikes_values{chan},2)
	MSE(i) = norm(spikes_values{chan}(i,2:7) - spike_mode(chan,2:7),2);%squared error versus template
end
keep = MSE < 1*norm(spike_mode(chan,:),2); %keep it if the difference is less than the norm of the signal
good_spikes_values{chan} = spikes_values{chan}(keep,:);
good_spikes_time{chan} = spikes_time{chan}(keep,:);


%kt(chan) = size(good_spikes_values{chan},2); %save number of good spike
%kT = size(signal,1)/sampFreq; %save length of the signal

%PCA%

% keep_pca = cell(channel_count,1);

% figure;
% hold on
% keep_pca{chan} = good_spikes_values{chan} - mean(mean(good_spikes_values{chan}));
% [COEFF,SCORE] = princomp(keep_pca{chan});
% scatter(SCORE(:,1),SCORE(:,2));
% IDX = kmeans(SCORE(:,1:2),2);
% divide=IDX==1;
% hold on
% scatter(SCORE(divide,1),SCORE(divide,2),'r')
% cluster = divide(1:length(good_spikes_values{chan}));
%title(sprintf( '%s %d, %s %c', 'Channel', chan, 'Animal r451') );
 
%draw only good spike 
figure;
hold on
for i = 1:size(good_spikes_values{chan},2)
    graph(chan) = plot(good_spikes_values{chan}(i,:));
end

%draw all spike
figure;
hold on
for i = 1:size(spikes_values{chan},2)
    graph(chan) = plot(spikes_values{chan}(i,:));
end

disp('done')