import brain_state_calculate as bsc
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#some graph about the neuron evolution over day
# how many neurons lost ?
# how many neurons gain ?
# how many neurons modulate ?
# how much the fire rate is modulated ?

base_dir = '../RT_classifier/BMIOutputs/BMISCIOutputs/'
files = {'r31':
             {'03': range(1, 25)+range(52, 58),
              '04': range(1, 45),
              '06': range(78, 113),
              '07': range(27, 51),
              '10': range(6, 31),
              '11': range(1, 16),
              '12': range(1, 27),
              '13': range(63, 89),
              '14': range(1, 23)},
         'r32':
             {'03': range(25, 52),
              '04': range(45, 83),
              '06': range(42, 78),
              '07': range(51, 82),
              '10': range(31, 69),
              '11': range(1, 36),
              '12': range(27, 54),
              '13': range(32, 63)},
         'r34':
             {'06': range(1, 42),
              '07': range(1, 27),
              '10': range(1, 6),
              '11': range(1, 31),
              '12': range(54, 87),
              '13': range(1, 32),
              '14': range(23, 48)}
        }

number_of_chan = 128
group_chan_by = 1
my_cft = bsc.cpp_file_tools(number_of_chan, group_chan_by)
#number of chan after grouping
number_of_chan /=group_chan_by
f = open('chan_evo_result.txt', 'w')

n = 0
all_chan_means = {}
all_chan_stds = {}
all_new_neuron = {}
all_lost_neuron = {}
all_mod_neuron = {}
all_chan_mod_count = {}
all_chan_mod = []

perc_modulation = []
for rat in files.keys():
    all_chan_means[rat] = []
    all_chan_stds[rat] = []
    all_new_neuron[rat] = []
    all_lost_neuron[rat] = []
    all_mod_neuron[rat] = []
    all_chan_mod_count[rat] = [0]*number_of_chan
    for date in files[rat].keys():
        file_date = '12'+date
        dir_name = base_dir+'Dec'+date+'/'+rat+'/'
        l_res, tmp_obs = my_cft.convert_cpp_file(dir_name, file_date, files[rat][date], False, 'SCIOutput_')
        l_obs = np.array(tmp_obs)
        all_chan_means[rat].append(l_obs.mean(0))
        all_chan_stds[rat].append(l_obs.std(0))

    all_chan_means[rat] = np.array(all_chan_means[rat])
    all_chan_stds[rat] = np.array(all_chan_stds[rat])
    for i in range(all_chan_means[rat].shape[1]):
        plt.plot(all_chan_means[rat][:, i], 'b-')
        plt.plot(all_chan_means[rat][:, i]+all_chan_stds[rat][:, i], 'b--')
        plt.plot(all_chan_means[rat][:, i]-all_chan_stds[rat][:, i], 'b--')
        plt.savefig('tmp_fig/chan_evol_'+str(n) +'.png', bbox_inches='tight')
        plt.close()
        n += 1
    for d in range(1, all_chan_means[rat].shape[0]):
        new_neuron = 0
        lost_neuron = 0
        mod_neuron = 0
        for c in range(all_chan_means[rat].shape[1]):
            if all_chan_means[rat][d-1, c] == 0 and all_chan_means[rat][d, c] != 0:
                new_neuron += 1
            if all_chan_means[rat][d-1, c] != 0 and all_chan_means[rat][d, c] == 0:
                lost_neuron += 1
            if all_chan_means[rat][d, c] != 0:
                all_chan_mod_count[rat][c] += 1
                mod_neuron += 1
        #we count only modulation for chan that are modulate at least one time during the day
        for c in range(all_chan_means[rat].shape[1]):
            all_chan_mod.append(all_chan_means[rat][d-1, c]-all_chan_means[rat][d, c])
        #divided by number_of_chan/100 cause we want %
        all_new_neuron[rat].append(new_neuron/(number_of_chan/100))
        all_lost_neuron[rat].append(lost_neuron/(number_of_chan/100))
        all_mod_neuron[rat].append(mod_neuron/(number_of_chan/100))

    print all_chan_means[rat].shape[0], Counter(all_chan_mod_count[rat])
    tmp = Counter(all_chan_mod_count[rat])
    for k in tmp.keys():
        perc_modulation += [k/float(all_chan_means[rat].shape[0]-1)*100]*tmp[k]
    plt.plot(all_new_neuron[rat], 'g-', label="new")
    plt.plot(all_lost_neuron[rat], 'r-', label="lost")
    plt.plot(all_mod_neuron[rat], 'b-', label="modulate")
    plt.legend(loc='upper right')
    plt.ylabel('% of neurons')
    plt.xlabel('day')
    plt.savefig('tmp_fig/neuron_evo_'+rat+'.png')
    plt.close()
boxplot_new_neuron = []
boxplot_lost_neuron = []
boxplot_mod_neuron = []
for rat in files.keys():
    boxplot_new_neuron += all_new_neuron[rat]
    boxplot_lost_neuron += all_lost_neuron[rat]
    boxplot_mod_neuron += all_mod_neuron[rat]
plt.boxplot([boxplot_new_neuron, boxplot_lost_neuron, boxplot_mod_neuron])
plt.xticks([1, 2, 3], ['new', 'lost', 'modulate'])
plt.ylabel('% of neuron')
plt.savefig('tmp_fig/neuron_evo_boxplot.png')
f.write('\n mean new: ')
f.write(str(np.array(boxplot_new_neuron).mean()))
f.write('\n std new: ')
f.write(str(np.array(boxplot_new_neuron).std()))
f.write('\n mean lost: ')
f.write(str(np.array(boxplot_lost_neuron).mean()))
f.write('\n std lost: ')
f.write(str(np.array(boxplot_lost_neuron).std()))
f.write('\n mean mod: ')
f.write(str(np.array(boxplot_mod_neuron).mean()))
f.write('\n std mod: ')
f.write(str(np.array(boxplot_mod_neuron).std()))
plt.close()

all_chan_mod=np.array(all_chan_mod)
plt.boxplot(all_chan_mod[all_chan_mod != 0])
plt.xticks([1], ['mean fire modulation between day'])
plt.savefig('tmp_fig/neuron_mod_over_d_boxplot.png')
plt.close()

f.write('\nmean var: ')
f.write(str(all_chan_mod[all_chan_mod != 0].mean()))
f.write('\nstds var: ')
f.write(str(all_chan_mod[all_chan_mod != 0].std()))

bins = np.linspace(0, 100, 6)
hist, bins = np.histogram(perc_modulation, bins)
hist /= 3.84
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
plt.ylabel('% of neuron')
plt.xlabel('% of day modulate')
plt.savefig('tmp_fig/modulation_hist.png')
plt.close()
#number of lost neurons between day
#number of new neurons
f.close()
print('###############')
print('####  END  ####')

