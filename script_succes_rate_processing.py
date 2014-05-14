import pickle
import matplotlib.pyplot as plt
import numpy as np

def import_success_rate():
    with open('success_rate', 'rb') as my_file:
        return pickle.load(my_file)

#sr_dict[rat][date]={ "kohoRL": [], 'GMMonline': [], 'koho': [], 'GMMoffRL': [], 'GMMoff': []}

rat='r32'
sr_dict = import_success_rate()

sr_all_kohoRL=[]
sr_all_koho=[]
sr_all_GMMonline=[]
overday_mean_kohoRL=[]
overday_std_kohoRL=[]
overday_mean_GMMonline=[]
overday_std_GMMonline=[]

# sr_all = {"koho_RL": [], 'GMM_online': [], 'koho': [], 'GMM_offline_RL': [], 'GMM_offline': []}
# overday_mean = {"koho_RL": [], 'GMM_online': [], 'koho': [], 'GMM_offline_RL': [], 'GMM_offline': []}
# overday_std= {"koho_RL": [], 'GMM_online': [], 'koho': [], 'GMM_offline_RL': [], 'GMM_offline': []}
sr_all = {"kohoRL": [], 'GMMonline': [], 'koho': [], 'GMMoffRL': [], 'GMMoff': []}
overday_mean = {"kohoRL": [], 'GMMonline': [], 'koho': [], 'GMMoffRL': [], 'GMMoff': []}
overday_std = {"kohoRL": [], 'GMMonline': [], 'koho': [], 'GMMoffRL': [], 'GMMoff': []}

for d in sr_dict[rat]:
    for res in sr_dict[rat][d]:
        sr_all[res] += sr_dict[rat][d][res]
        overday_mean[res].append(np.array(sr_dict[rat][d][res]).mean())
        overday_std[res].append(np.array(sr_dict[rat][d][res]).std())


plt.figure()
i = 0
for res in sr_all:
    sr_all[res] = np.array(sr_all[res])+ i * 1.2
    plt.plot(sr_all[res], 'o-', label=res)
    i += 1
plt.ylim(-0.2, i*1.2)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center bottom', ncol=4, mode="expand")
plt.figure()
i = 0
for res in sr_all:
    overday_mean[res] = np.array(overday_mean[res])
    overday_std[res] = np.array(overday_std[res])
    plt.plot(overday_mean[res] + i * 1.2, 'o-')
    plt.plot(overday_mean[res] - overday_std[res]+ i * 1.2, '--')
    plt.plot(overday_mean[res] + overday_std[res]+ i * 1.2, '--')
    i += 1
# plt.plot(overday_mean_GMMonline, 'bo-')
# plt.plot(overday_mean_GMMonline-overday_std_GMMonline, 'b--')
# plt.plot(overday_mean_GMMonline+overday_std_GMMonline, 'b--')
# plt.plot(overday_mean_kohoRL, 'ro-')
# plt.plot(overday_mean_kohoRL-overday_std_kohoRL, 'r--')
# plt.plot(overday_mean_kohoRL+overday_std_kohoRL, 'r--')

plt.show()

print('###############')
print('####  END  ####')