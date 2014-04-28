import csv
import signal_processing
import copy

#we add ground truth to the cpp generated, ground truth are in vicon file
def convert_time(list_times, file_time, vicon_time):
    res=[]
    for n in range(len(list_times)):
        res.append((list_times[n]-vicon_time) * 10 + file_time)
    return res

files0422 = [5, 6, 7, 8]
files0423 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
trials_vicon0423 = [13, 14, 15, 16, 17, 18]

dir_name = '../RT_classifier/BMIOutputs/0423_r600/'
sp = signal_processing.Signal_processing()

trials_vicon = trials_vicon0423
files = files0423
date = '0423'
TDT_padding = 0
#first find start stim on row[4]
#convert with stimOn on mat
#convert with synch to vicon
#third trial multi stim

#TDT time for vicon record start
synch = sp.load_m(dir_name+'600_14'+date+'_synch_stim.mat', 'Synch')
#stim on for the TDT
stim_on = sp.load_m(dir_name+'600_14'+date+'_synch_stim.mat', 'stimON')
#stim on fot the cpp file
stim_on_file = []

#store stim on for each file
for f in range(len(files)):
    filename = date+'healthyOutput_'+str(files[f])+'.txt'
    csvfile = open(dir_name + filename, 'rb')
    file = csv.reader(csvfile, delimiter=' ', quotechar='"')
    for row in file:
        if len(row) > 7 and row[4] == '1':
            stim_on_file.append(float(row[1]))
            break

#set vicon time synch with TDT time
vicon_data={}
for trial in trials_vicon:
    filename = 'P0_4_RW' + str(trial)
    file_events = sp.load_csv(dir_name + filename + '_EVENTS.csv')
    file_analog = sp.load_csv(dir_name + filename + '_ANALOG.csv')
    vicon_data[trial] = sp.vicon_extract(file_events, {})
    vicon_data[trial] = sp.vicon_extract(file_analog, copy.copy(vicon_data[trial]))
    vicon_data[trial] = sp.synch_vicon_with_TDT(vicon_data[trial], TDT_padding)

#get time factor to synchronise data, store it in stim_on_vicon
stim_on_vicon = []
#cpp2vicon store file match between cpp and vicon (ex: vicon file 4 contain cpp file 7 and 8)
cpp2vicon = {}
for i in range(len(stim_on[0])/2):
    time = stim_on[1][i*2]
    for n in range(len(synch[0])/2):
        if synch[1][n*2] < time < synch[1][n*2+1]:
            time = stim_on[1][i*2]-synch[1][n*2]
            stim_on_vicon.append(time)
            cpp2vicon[i] = n
            break

print cpp2vicon
for f in range(len(files)):
    filename = date+'healthyOutput_'+str(files[f])+'.txt'
    file1 = open(dir_name + filename, 'rb')
    file2 = open(dir_name + 't_'+filename, 'wb')
    reader = csv.reader(file1, delimiter=' ', quotechar='"')
    writer = csv.writer(file2, delimiter=' ', quotechar='"')

    trial = trials_vicon[cpp2vicon[f]]

    #in vicon file
    #Right + foot off = we take the rat
    #Right + foot Strike = we release the rat on the floor
    #General + Foot Off = the rat begin to walk
    #General + Foot strike = the rat stop walking
    air_stop = convert_time(vicon_data[trial]['Right']['Foot Strike'], stim_on_file[f], stim_on_vicon[f])
    air_start = convert_time(vicon_data[trial]['Right']['Foot Off'], stim_on_file[f], stim_on_vicon[f])
    start_walk = convert_time(vicon_data[trial]['General']['Foot Off'], stim_on_file[f], stim_on_vicon[f])
    end_walk = convert_time(vicon_data[trial]['General']['Foot Strike'], stim_on_file[f], stim_on_vicon[f])

    #we store the row in tmp_file list
    tmp_file = []
    for row in reader:
        tmp_file.append(row)

    #if the rat is in the air (not in the ground) we set row[5] to -1 else row[5] stay to 0
    for g in range(len(air_start)):
        for r in range(len(tmp_file)):
            row = tmp_file[r]
            if len(row) > 7 and row[0] != '0' and air_start[g] < float(row[1]) < air_stop[g]:
                row[5] = '-1'
            tmp_file[r] = row
    file1.close()
    for w in range(len(start_walk)):
        for r in range(len(tmp_file)):
            row = tmp_file[r]
            #if the rat walk row[5] is set to 2
            if len(row) > 7 and row[0] != '0' and start_walk[w] < float(row[1]) < end_walk[w]:
                row[5] = '2'
            tmp_file[r] = row

    #write the new file
    for row in tmp_file:
        writer.writerow(row)
    file2.close()