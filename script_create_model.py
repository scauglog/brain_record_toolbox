import brain_state_calculate as bsc
import cpp_file_tools as cft

file=['0527healthyOutput_4.txt']
my_bsc = bsc.brain_state_calculate(32)
my_cft = cft.cpp_file_tools(32, 1)
my_bsc.init_networks(file, my_cft)
my_bsc.save_networks('', '0527')

print 'END'