import mlp
import csv

l_obs = []
l_res = []
training_count = 10
alpha = 0.1
init_tail = 5
#files = [3, 6, 7, 8, 9, 10, 11, 16, 17, 18, 21, 22, 23, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 40, 41, 42]
files = [3, 6, 7, 8]
date = '1119'
for f in files:
    filename = date+'healthyOutput_'+str(f)+'.txt'
    csvfile = open(filename, 'rb')
    file = csv.reader(csvfile, delimiter=' ', quotechar='"')
    #grab expected result in file and convert, grab input data
    for row in file:
        if len(row) > 7:
            ratState = row[3]
            if ratState == '1':
                l_res.append([1, 0, 0])
                l_obs.append(row[8:128+8])
            elif ratState == '-2':
                l_res.append([0, 1, 0])
                l_obs.append(row[8:128+8])
            elif ratState == '2':
                l_res.append([0, 0, 1])
                l_obs.append(row[8:128+8])

    for i in range(1,len(l_res)):
        if l_res[i] == [0, 0, 1] and l_res[i-1] == [1, 0, 0]:
            for j in range(i-init_tail, i):
                l_res[j] = [0, 1, 0]
perceptron = mlp.Network(129, 3, 4, 16)

print(len(l_obs))
for i in range(training_count):
    print i
    perceptron.backprop(alpha, l_obs, l_res)

good = 0
cpt = 0
for i in range(len(l_obs)):
    perceptron.run(l_obs[i])
    res = perceptron.output()
    res = res.index(max(res))+1
    if res == 1:
        res = [1, 0, 0]
    elif res == 2:
        res = [0, 1, 0]
    elif res == 3:
        res = [0, 0, 1]

    if res == l_res[i]:
        good += 1
    cpt += 1

print (good/cpt)*100