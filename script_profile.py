import brain_state_calculate_c as bsc
import cpp_file_tools_c as cft
import pstats, cProfile

file=["F:/data/r617/0620healthyOutput_1.txt","F:/data/r617/0620healthyOutput_2.txt","F:/data/r617/0620healthyOutput_3.txt"]
my_bsc = bsc.brain_state_calculate(32)
my_cft = cft.cpp_file_tools(32, 1)

# cProfile.runctx('my_bsc.init_networks(file, my_cft)', globals(), locals(), "Profile.prof")
# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("time").print_stats()

my_bsc.init_networks(file, my_cft)
print "END kohonen"
#my_bsc.train_one_file(file[1], my_cft, is_healthy=False, new_day=True, obs_to_add=0, with_RL=True, train_mod_chan=False, on_stim=False, autosave=False)
cProfile.runctx('my_bsc.train_one_file(file[1], my_cft, is_healthy=False, new_day=True, obs_to_add=0, with_RL=True, train_mod_chan=False, on_stim=False, autosave=False)', globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

print 'END'