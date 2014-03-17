import pickle
import signal_processing as sig_proc

dir_name = '../data/r448/r448_131022_rH/'

img_ext = '.png'
save_img = True
show = False
save_obj = True

sp = sig_proc.Signal_processing(save_img, show, img_ext)

filename='p0_3RW05'
file_events=sp.load_csv(dir_name+filename+'_EVENTS.csv')
file_analog=sp.load_csv(dir_name+filename+'_ANALOG.csv')
data=sp.vicon_extract(file_events)
data=sp.vicon_extract(file_analog,data)
data=sp.synch_vicon_with_TDT(data)


print('\n\n#################')
print('####   END   ####')