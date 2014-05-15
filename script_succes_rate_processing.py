import pickle
import matplotlib.pyplot as plt
import numpy as np

def import_success_rate(filename):
    with open(filename, 'rb') as my_file:
        return pickle.load(my_file)

#sr_dict[rat][date]={ "kohoRL": [], 'GMMonline': [], 'koho': [], 'GMMoffRL': [], 'GMMoff': []}

rat='r32'
sr_dict = import_success_rate('success_rate_SCI_r32_v2')

sr_all_kohoRL=[]
sr_all_koho=[]
sr_all_GMMonline=[]
overday_mean_kohoRL=[]
overday_std_kohoRL=[]
overday_mean_GMMonline=[]
overday_std_GMMonline=[]

sr_all = {"koho_RL": [], 'GMM_online': [], 'koho': [], 'GMM_offline_RL': [], 'GMM_offline': []}
overday_mean = {"koho_RL": [], 'GMM_online': [], 'koho': [], 'GMM_offline_RL': [], 'GMM_offline': []}
overday_std= {"koho_RL": [], 'GMM_online': [], 'koho': [], 'GMM_offline_RL': [], 'GMM_offline': []}
color=['b','r','g','c','m','y','k']
net_order=['GMM_online', 'koho', 'koho_RL', 'GMM_offline_RL', 'GMM_offline']
#net_order=['GMM_online', 'koho_RL']
end_day = []
cpt=0
for d in sr_dict[rat]:
    cpt += len(sr_dict[rat][d]['GMM_online'])
    for res in sr_dict[rat][d]:
        if res != 'l_of_res':
            sr_all[res] += sr_dict[rat][d][res]
            overday_mean[res].append(np.array(sr_dict[rat][d][res]).mean())
            overday_std[res].append(np.array(sr_dict[rat][d][res]).std())
    end_day.append(cpt-1)

plt.figure()
i = 0
for res in net_order:
    plt.subplot(len(net_order),1,i)
    sr_all[res] = np.array(sr_all[res])
    plt.plot(sr_all[res], color[i]+'o-', label=res)
    i += 1
    plt.ylabel(res)
    plt.ylim(-0.1,1.1)
    for end in end_day:
        plt.vlines(end, -0.1, 1.1)
plt.figure()

i = 0
for res in net_order:
    overday_mean[res] = np.array(overday_mean[res])
    overday_std[res] = np.array(overday_std[res])
    #plt.subplot(len(sr_all),1,i)
    plt.plot(overday_mean[res], color[i]+'o-', label=res)
    #plt.plot(overday_mean[res] - overday_std[res], color[i]+'--')
    #plt.plot(overday_mean[res] + overday_std[res], color[i]+'--')
    plt.ylim(-0.2, 1.2)
    plt.hlines(0, 0, overday_mean[res].shape[0])
    i += 1
plt.xlabel('day')
plt.ylabel('mean success rate')
plt.legend()

rat = 'r600'
sr_dict = import_success_rate('success_rate_simulated_r600_v2')
prob = []
cpt = 0
end_day = []
for d in sr_dict[rat]:
    for l_of_res in sr_dict[rat][d]['l_of_res']:
        good = 0
        cpt += 1
        for i in range(l_of_res[0].shape[0]):
            if l_of_res[3][i] == l_of_res[2][i]:
                good += 1
        prob.append(good / float(l_of_res[0].shape[0]))
    end_day.append(cpt-1)
plt.figure()
plt.plot(prob)
for end in end_day:
    plt.vlines(end, 0, 1)


plt.show()
print('###############')
print('####  END  ####')