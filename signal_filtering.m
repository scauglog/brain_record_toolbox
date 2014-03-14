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