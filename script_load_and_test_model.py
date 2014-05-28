import brain_state_calculate as bsc
from matplotlib import pyplot as plt
import cpp_file_tools as cft

file=['0527healthyOutput_4.txt']
my_bsc = bsc.brain_state_calculate(32)
my_cft = cft.cpp_file_tools(32, 1)

my_bsc.load_networks('koho_networks_0527')
for f in file:
    l_res, l_obs = my_cft.convert_one_cpp_file(f)
    my_bsc.init_test()
    result = []
    for obs in l_obs:
        res=my_bsc.test_one_obs(obs)
        print res
        result.append(res)
    #success, l_of_res = my_bsc.test(l_obs, l_res)
    plt.plot(result)
    plt.show()

print 'END'