import mlp
import kohonen_neuron as kn
import csv
import pickle
import time
import copy

save_obj=True
l_obs = []
l_res = []
alpha = 0.1
koho_row = 20
koho_col = 5
neighbor = 3
min_win = 4
ext_img = '.png'
save = False
show = False
init_tail = 5
#11/26
files1126 = [32, 33, 34, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 53, 54, 55, 56, 57, 58, 59, 60]
#11/27
files1127 = [57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 86]
#12/03 SCI
files1203 = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

date = '1126'
files = files1126[0:9]
for f in files:
    filename = 'r32/'+date+'healthyOutput_'+str(f)+'.txt'
    csvfile = open(filename, 'rb')
    file = csv.reader(csvfile, delimiter=' ', quotechar='"')
    #grab expected result in file and convert, grab input data
    for row in file:
        if len(row) > 7:
            ratState = row[3]
            if ratState == '1':
                l_res.append([1, 0, 0])
                l_obs.append(map(float, row[7:128+7]))
            elif ratState == '-2':
                l_res.append([0, 1, 0])
                l_obs.append(map(float, row[7:128+7]))
            elif ratState == '2':
                l_res.append([0, 0, 1])
                l_obs.append(map(float, row[7:128+7]))

    for i in range(1, len(l_res)):
        if l_res[i] == [0, 0, 1] and l_res[i-1] == [1, 0, 0]:
            for j in range(i-init_tail, i):
                l_res[j] = [0, 1, 0]
perceptron = mlp.Network([129, 64, 32, 16, 3])
for i in range(5):
    print i
    start = time.time()
    perceptron.backprop(alpha, l_obs, l_res)
    end = time.time()
    print(end-start)

good = 0.0
for i in range(len(l_obs)):
    perceptron.run(l_obs[i])
    res = [0, 0, 0]
    rank = perceptron.output().index(max(perceptron.output()))
    res[rank] = 1
    print(res,l_res[i])
    if res == l_res[i]:
        good += 1

print good/float(len(l_obs))

# koho=kn.Kohonen(koho_row, koho_col, 128, 5, alpha, neighbor, min_win, ext_img, save, show)
#
# for i in range(10):
#     print i
#     start = time.time()
#     koho.algo_kohonen(l_obs)
#     end = time.time()
#     print(end-start)
# koho.best_neurons(l_obs)
# koho.group_neuron_into_x_class(10)
#
# gpe_label = {}
# for g in koho.groups:
#     gpe_label[g.number] = {1: 0, 2: 0, 3: 0}
#
# for i in range(len(l_obs)):
#     gpe_res = koho.find_group_min_dist(l_obs[i])
#     if l_res[i] == [1, 0, 0]:
#         gpe_label[gpe_res.number][1] += 1
#     elif l_res[i] == [0, 1, 0]:
#         gpe_label[gpe_res.number][2] += 1
#     elif l_res[i] == [0, 0, 1]:
#         gpe_label[gpe_res.number][3] += 1
#
# for k1 in gpe_label:
#     sum_obs = 0.0
#     sum_obs += gpe_label[k1][1]
#     sum_obs += gpe_label[k1][2]
#     sum_obs += gpe_label[k1][3]
#     gpe_label[k1][1] /= sum_obs
#     gpe_label[k1][2] /= sum_obs
#     gpe_label[k1][3] /= sum_obs
# print gpe_label
#
# mlp_obs = []
# mlp_res = []
# tmp_list = []
# for i in range(len(l_obs)):
#     gpe = koho.find_group_min_dist(l_obs[i])
#     for v in gpe_label[gpe.number].values():
#         tmp_list.append(v)
#     if len(tmp_list) > 9:
#         del tmp_list[0]
#         del tmp_list[0]
#         del tmp_list[0]
#         mlp_obs.append(copy.copy(tmp_list))
#         mlp_res.append(l_res[i])
#
#
# perceptron = mlp.Network([10, 10, 10, 3])
# for i in range(10):
#     perceptron.backprop(alpha, mlp_obs, mlp_res)
#
# good = 0.0
# for i in range(len(mlp_obs)):
#     perceptron.run(mlp_obs[i])
#     res = [0, 0, 0]
#     rank = perceptron.output().index(max(perceptron.output()))
#     res[rank] = 1
#     print(res,mlp_res[i])
#     if res == mlp_res[i]:
#         good += 1
#
# print good/float(len(mlp_obs))

print('###############')
print('####  END  ####')