import Tkinter
import tkFileDialog
import csv
import numpy as np

initdir = "C:\\"
root = Tkinter.Tk()
root.withdraw()
file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=initdir,  title="select cpp file to train the classifier", filetypes=[('all files', '.*'), ('text files', '.txt')])
paths = root.tk.splitlist(file_path)
mean = []
std = []
cycle = []
for filename in paths:
    print filename
    csv_file = open(filename, 'rb')
    file = csv.reader(csv_file, delimiter=' ', quotechar='"')
    time = []
    cycle_count = []
    for row in file:
        if len(row) > 7 and row[0] != '0':
            time.append(int(row[0]))
            cycle_count.append(int(row[1]))

    cycle_count_delta_tmp = (np.array(cycle_count[1:])-np.array(cycle_count[:-1]))-1
    cycle_count_delta = cycle_count_delta_tmp[cycle_count_delta_tmp >= 0]
    cycle_count_delta =np.hstack((cycle_count_delta, abs(cycle_count_delta_tmp[cycle_count_delta_tmp < 0]+10000)))

    cycle.append(cycle_count_delta.sum()/float(cycle_count_delta.shape[0]))
    time = np.array(time)
    mean.append(time.mean())
    std.append(time.std())

print "mean", np.array(mean).mean()
print "std", np.array(std).mean()
print "cycle count", np.array(cycle).mean()

fname = tkFileDialog.asksaveasfilename(initialdir=initdir, title="save as", initialfile="cppfile_stat", defaultextension="csv")
if not fname=="":
    file = open(fname, "wb")
    try:
        writer = csv.writer(file)
        writer.writerow(('cycle drop', 'mean', 'std'))
        for i in range(len(mean)):
            writer.writerow((cycle[i], mean[i], std[i]))
    finally:
        file.close()
