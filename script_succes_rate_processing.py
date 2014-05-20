import pickle
import matplotlib.pyplot as plt
import numpy as np
import brain_state_calculate as bsc

def import_success_rate(filename):
    with open(filename, 'rb') as my_file:
        return pickle.load(my_file)

#sr_dict[rat][date]={ "kohoRL": [], 'GMMonline': [], 'koho': [], 'GMMoffRL': [], 'GMMoff': []}

rat = 'r32'
sr_dict = import_success_rate('success_rate_SCI_r32_v3')
sr_dict_hist = import_success_rate('success_rate_SCI_r32_hist')
sr_dict_shuffle = import_success_rate('success_rate_SCI_r32_shuffle')

#group in one dictionnary
for d in sr_dict[rat]:
    sr_dict[rat][d]['koho_RL_hist'] = sr_dict_hist[rat][d]['koho_RL']
    sr_dict[rat][d]['koho_hist'] = sr_dict_hist[rat][d]['koho']
    sr_dict[rat][d]['koho_RL_shuffle'] = sr_dict_shuffle[rat][d]['koho_RL']
    sr_dict[rat][d]['koho_shuffle'] = sr_dict_shuffle[rat][d]['koho']

group_by=5
sr_all = {'koho_RL': [], 'GMM_online': [], 'koho': [], 'GMM_offline_RL': [], 'GMM_offline': [],  'koho_hist': [], 'koho_RL_hist': [], 'koho_shuffle': [], 'koho_RL_shuffle': []}
overday_mean = {'koho_RL': [], 'GMM_online': [], 'koho': [], 'GMM_offline_RL': [], 'GMM_offline': [],  'koho_hist': [], 'koho_RL_hist': [], 'koho_shuffle': [], 'koho_RL_shuffle': []}
overday_std= {'koho_RL': [], 'GMM_online': [], 'koho': [], 'GMM_offline_RL': [], 'GMM_offline': [],  'koho_hist': [], 'koho_RL_hist': [], 'koho_shuffle': [], 'koho_RL_shuffle': []}
color = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
net_order = ['GMM_online', 'koho', 'koho_RL', 'koho_hist', 'koho_RL_hist', 'koho_shuffle', 'koho_RL_shuffle', 'GMM_offline_RL', 'GMM_offline']
#net_order=['GMM_online', 'koho_RL']
end_day = []
cpt=0

for d in sr_dict[rat]:
    if group_by <= 1:
        cpt += len(sr_dict[rat][d]['GMM_online'])

    for res in sr_dict[rat][d]:
        if res != 'l_of_res':
            if group_by > 1:
                trial=1
                tmp=0
                for i in range(len(sr_dict[rat][d][res])):
                    tmp += sr_dict[rat][d][res][i]
                    if trial == 0:
                        sr_all[res].append(tmp/float(group_by))
                        tmp = 0
                        if res == 'GMM_online':
                            cpt += 1
                    trial += 1
                    trial %= group_by
                if res == 'GMM_online':
                    end_day.append(cpt)
            else:
                sr_all[res] += sr_dict[rat][d][res]
            overday_mean[res].append(np.array(sr_dict[rat][d][res]).mean())
            overday_std[res].append(np.array(sr_dict[rat][d][res]).std())

    if group_by <= 1:
        end_day.append(cpt)

plt.figure(figsize=(10, 16))
i = 0
for res in net_order:
    plt.subplot(len(net_order),1,i)
    sr_all[res] = np.array(sr_all[res])
    plt.plot(sr_all[res], color[i%len(color)]+'o-', label=res)
    i += 1
    plt.ylabel(res)
    plt.ylim(-0.1,1.1)
    for end in end_day:
        plt.vlines(end, -0.1, 1.1)
plt.tight_layout()
plt.figure()

i = 0
for res in net_order:
    overday_mean[res] = np.array(overday_mean[res])
    overday_std[res] = np.array(overday_std[res])
    #plt.subplot(len(sr_all),1,i)
    plt.plot(overday_mean[res], color[i%len(color)]+'o-', label=res)
    #plt.plot(overday_mean[res] - overday_std[res], color[i]+'--')
    #plt.plot(overday_mean[res] + overday_std[res], color[i]+'--')
    plt.ylim(-0.1, 1.1)
    plt.hlines(0, 0, overday_mean[res].shape[0])
    i += 1
plt.xlabel('day')
plt.ylabel('mean success rate')
plt.legend()

rat = 'r600'
sr_dict = import_success_rate('success_rate_simulated_r600_v2_hist')
prob = []
cpt = 0
end_day = []
for d in sr_dict[rat]:
    for l_of_res in sr_dict[rat][d]['l_of_res']:
        good = 0
        cpt += 1
        for i in range(l_of_res['gnd_truth'].shape[0]):
            if l_of_res['real_gnd_truth'][i] == l_of_res['koho'][i]:
                good += 1
        prob.append(good / float(l_of_res['gnd_truth'].shape[0]))
    end_day.append(cpt-1)
plt.figure()
plt.plot(prob)
for end in end_day:
    plt.vlines(end, 0, 1)

success_rate=[]
my_cft= bsc.cpp_file_tools(32, 1)
for d in sr_dict[rat]:
    for l_of_res in sr_dict[rat][d]['l_of_res']:
        success_rate.append(my_cft.success_rate(l_of_res['koho'], l_of_res['gnd_truth']))

plt.figure()
plt.plot(success_rate)
for end in end_day:
    plt.vlines(end, 0, 1)
plt.ylim(-0.2, 1.2)
plt.show()
print('###############')
print('####  END  ####')