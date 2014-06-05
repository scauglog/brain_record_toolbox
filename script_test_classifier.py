import brain_state_calculate as bsc
import cpp_file_tools as cft
from matplotlib import pyplot as plt
import numpy as np
import Tkinter
import tkFileDialog

initdir="C:\\"

my_bsc = bsc.brain_state_calculate(32)
my_cft = cft.cpp_file_tools(32, 1, show=True)
my_bsc.init_networks_on_files(initdir, my_cft, train_mod_chan=False)

print "select the file to test"
root = Tkinter.Tk()
root.withdraw()
file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=initdir,  title="select cpp file to train the classifier", filetypes=[('all files', '.*'), ('text files', '.txt')])
print "test the file"
if not file_path == "":
    files = root.tk.splitlist(file_path)

    for f in files:
        print f
        l_res, l_obs = my_cft.read_cpp_files([initdir + f], is_healthy=False, cut_after_cue=True, init_in_walk=True)
        success, l_of_res = my_bsc.test(l_obs, l_res)
        my_cft.plot_result(l_of_res)
        plt.figure()
        plt.imshow(np.array(l_obs).T, interpolation='none')
        my_bsc.train_unsupervised_one_file(initdir+f, my_cft, is_healthy=False)

plt.show()

print '#############'
print '#### END ####'