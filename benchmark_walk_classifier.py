import brain_state_calculate as bsc
import numpy as np
import copy
import random as rnd
import pickle
from collections import OrderedDict
from cpp_file_tools import cpp_file_tools

class ChangeObs:
    def __init__(self, l_obs):
        rnd.seed(42)
        #wich col we should move
        self.move_chan = []
        #where we should move the col
        self.move_chan_to = []
        #how much we should modulate the given col
        self.value_modulate = []
        #param for mean modulation
        mu = 0
        sigma = 2

        l_obs = np.array(l_obs)
        #index of modulated channel
        self.mod_chan = l_obs.sum(0).nonzero()[0]
        #number of channel
        self.nbchan = len(l_obs[0])
        #params for number of chan to move
        #35% of modulated chan lost or gain par day with 28% of std
        mean_move = 0.35 * self.mod_chan.shape[0]
        std_move = 0.28 * self.mod_chan.shape[0]
        change_x_chan = 0
        while change_x_chan < 1:
            change_x_chan = self.f2i(rnd.gauss(mean_move, std_move))

        for i in range(change_x_chan):
            self.move_chan.append(self.f2i(rnd.uniform(0, self.mod_chan.shape[0]-1)))
            self.move_chan_to.append(self.f2i(rnd.uniform(0, self.nbchan-1)))

        for i in range(self.nbchan):
            self.value_modulate.append(self.f2i(rnd.gauss(mu, sigma)))
        print self.mod_chan
        print self.move_chan
        print self.move_chan_to
        print self.value_modulate

    def change(self, l_obs):
        l_obs = np.array(l_obs)
        save_obs=copy.copy(l_obs)
        for c in range(l_obs.shape[1]):
            if c in self.mod_chan:
                l_obs[:, c] = l_obs[:, c]+self.value_modulate[c]
            if c in self.move_chan:
                ind = self.move_chan.index(c)
                move_to = self.move_chan_to[ind]
                tmp = copy.copy(l_obs[:, move_to])
                l_obs[:, move_to] = l_obs[:, c]
                l_obs[:, c] = tmp
        #we allow burst count to be negative in order to avoid all value set to zero after X "day"
        #l_obs[l_obs < 0] = 0
        return l_obs

    @staticmethod
    def f2i(number):
        return int(round(number, 0))

    @staticmethod
    def expand_walk(l_res, extend_before, extend_after):
        start_after = []
        for i in range(len(l_res)-1):
            if l_res[i] != l_res[i+1]:
                if l_res[i] == [1, 0]:
                    for n in range(i-extend_before, i+1):
                        if 0 < n < len(l_res):
                            l_res[n] = [0, 1]
                else:
                    start_after.append(i)
        for i in start_after:
            for n in range(i, i+extend_after):
                if 0 < n < len(l_res):
                    l_res[n] = [0, 1]
        return l_res

class NotImplementedException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Benchmark(object):
    def __init__(self, nb_chan, group_by):
        #general option
        self.save_obj = False
        self.ext_img = '.png'
        self.save_img = True
        self.show = False
        self.img_save_path = 'benchmark_img/'

        self.my_cft = cpp_file_tools(nb_chan, group_by, self.ext_img, self.save_img, self.show)
        self.res_dict={}

        #simulated benchmark option
        self.simulated_dir_name = '../RT_classifier/BMIOutputs/0423_r600/'
        simulated_iteration = 5
        self.simulated_files = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.simulated_date = 't_0423'
        self.simulated_rat = 'r0'
        self.simulated_corename = 'healthyOutput_'
        self.simulated_change_every = len(self.simulated_files)
        self.simulated_first_train = 5
        for i in range(simulated_iteration):
            self.simulated_files += self.simulated_files

        #SCI benchmark option
        self.SCI_dir_name = '../RT_classifier/BMIOutputs/BMISCIOutputs/'
        self.SCI_corename = 'SCIOutput_'
        self.SCI_first_train = 5
        self.SCI_min_obs = 10
        self.SCI_files = {'r31': OrderedDict([
                     ('03', range(1, 25)+range(52, 58)),
                     ('04', range(1, 45)),
                     ('06', range(78, 113)),
                     ('07', range(27, 51)),
                     ('10', range(6, 31)),
                     ('11', range(1, 16)),
                     ('12', range(1, 27)),
                     ('13', range(63, 89)),
                     ('14', range(1, 23))]),
                 'r32': OrderedDict([
                     ('03', range(25, 52)),
                     ('04', range(45, 83)),
                     ('06', range(42, 78)),
                     ('07', range(51, 82)),
                     ('10', range(31, 69)),
                     ('11', range(1, 36)),
                     ('12', range(27, 54)),
                     ('13', range(32, 63))]),
                 'r34': OrderedDict([
                     ('06', range(1, 42)),
                     ('07', range(1, 27)),
                     ('11', range(1, 31)),
                     ('12', range(54, 87)),
                     ('13', range(1, 32)),
                     ('14', range(23, 48))])
                 }

    def benchmark_SCI_data(self, shuffle_obs=False):
        self.res_dict = {}
        for rat in self.SCI_files.keys():
            init_networks = True
            self.res_dict[rat] = {}
            for date in self.SCI_files[rat].keys():
                dir_name = self.SCI_dir_name + 'Dec' + date + '/' + rat + '/'
                fulldate = '12'+date
                self.res_dict[rat][fulldate] = {'l_of_res': []}
                print '---------- ' + rat + ' ' + date + ' ----------'
                files = self.my_cft.convert_to_filename_list(dir_name, fulldate, self.SCI_files[rat][date][0:self.SCI_first_train], self.SCI_corename)
                if init_networks:
                    init_networks = False
                    self.init_classifier()
                    self.init_test(files)

                new_date = True
                #for each file of the day (=date)
                for n in range(self.SCI_first_train, len(self.SCI_files[rat][date])-1):
                    print '### ### ### ### ### ### ### ### ###'
                    print rat+'_'+str(fulldate)+'_'+str(n)+str(self.SCI_files[rat][date][n:n+1])

                    #get obs
                    files = self.my_cft.convert_to_filename_list(dir_name, fulldate, self.SCI_files[rat][date][n:n+1], self.SCI_corename)
                    l_res, l_obs = self.my_cft.read_cpp_files(files, is_healthy=False, cut_after_cue=False)
                    #if the trial is too short or have no neuron modulated we don't train
                    if len(l_obs) > self.SCI_min_obs and np.array(l_obs).sum() > 0:
                        if shuffle_obs:
                            l_obs=self.shuffle_obs(l_obs)
                        l_of_res = self.test_network_with_obs(l_obs, l_res)
                        self.res_dict[rat][fulldate]['l_of_res'].append(l_of_res)
                        if self.save_img or self.show:
                            self.my_cft.plot_result(l_of_res, 'SCI_data_'+rat+'_'+str(fulldate)+'_'+str(n)+str(self.SCI_files[rat][date][n:n+1]), self.img_save_path)

                        #when new day first learn with mod_chan
                        try:
                            self.train_with_obs(l_obs, l_res, new_date)
                            if new_date:
                                new_date = False
                        except ValueError:
                            print 'goto the next trial'

        print('###############')
        print('####  END  ####')
        return self.res_dict

    def benchmark_simulated_data_from_healthy(self, shuffle_obs=False):
        #save the res
        chg_obs = []
        rnd.seed(42)
        rat=self.simulated_rat
        self.res_dict = {rat: {str(len(chg_obs)): {'l_of_res': []}}}
        date = 0
        self.init_classifier()

        #init net
        files = self.my_cft.convert_to_filename_list(self.simulated_dir_name, self.simulated_date, self.simulated_files[0:self.simulated_first_train], self.simulated_corename)
        self.init_test(files)

        for i in range(self.simulated_first_train, len(self.simulated_files)):
            files = self.my_cft.convert_to_filename_list(self.simulated_dir_name, self.simulated_date, self.simulated_files[i:i+1], self.simulated_corename)
            l_res, l_obs = self.my_cft.read_cpp_files(files, is_healthy=False, cut_after_cue=False)

            #change the value
            for chg in chg_obs:
                l_obs = chg.change(l_obs)

            #prepare to change the value
            if i % self.simulated_change_every == 0:
                chg_obs.append(ChangeObs(l_obs))
                l_obs = chg_obs[-1].change(l_obs)
                print 'change obs:'+str(len(chg_obs))
                date = str(len(chg_obs))
                self.res_dict[rat][date] = {'l_of_res': []}
            #to simulate the cue we add
            extend_before = ChangeObs.f2i(rnd.gauss(0.4/0.1, 0.5))
            extend_after = ChangeObs.f2i(rnd.uniform(10, 30))
            l_res = ChangeObs.expand_walk(l_res, extend_before, extend_after)
            if shuffle_obs:
                l_obs=self.shuffle_obs(l_obs)
            print '### ### ### ### ### ### ### ### ###'
            print rat+'_'+str(date)+'_'+str(i)+str(self.simulated_files[i:i+1])

            l_of_res = self.test_network_with_obs(l_obs, l_res)
            l_res_gnd_truth, l_obs_trash = self.my_cft.convert_cpp_file(self.simulated_dir_name, self.simulated_date, self.simulated_files[i:i+1], is_healthy=False, file_core_name=self.simulated_corename,cut_after_cue=False)
            l_of_res['real_gnd_truth'] = np.array(l_res_gnd_truth).argmax(1)

            self.res_dict[rat][str(len(chg_obs))]['l_of_res'].append(l_of_res)
            if self.save_img or self.show:
                self.my_cft.plot_result(l_of_res, 'simulated_data_'+rat+'_'+str(date)+'_'+str(i)+str(self.simulated_files[i:i+1]),self.img_save_path)

            try:
                if i % self.simulated_change_every == 0:
                        self.train_with_obs(l_obs, l_res, True)
                else:
                        self.train_with_obs(l_obs, l_res, False)
            except ValueError:
                print 'goto the next trial'

        print('###############')
        print('####  END  ####')

        return self.res_dict

    def save_result(self, path='', extra_txt=''):
        filename = path+'result_'+extra_txt+'.pyObj'
        with open(filename, 'wb') as my_file:
            my_pickler = pickle.Pickler(my_file)
            my_pickler.dump(self.res_dict)

    @staticmethod
    def shuffle_obs(l_obs):
        rnd.shuffle(l_obs)

    def change_chan_group_by(self, nb_chan, group_by):
        self.my_cft = cpp_file_tools(nb_chan, group_by, self.ext_img, self.save_img, self.show)

    def init_classifier(self):
        raise NotImplementedException("Subclasses are responsible for creating this method")

    def init_test(self, files):
        raise NotImplementedException("Subclasses are responsible for creating this method")

    def test_network_with_files(self, files):
        raise NotImplementedException("Subclasses are responsible for creating this method")

    def test_network_with_obs(self, l_obs, l_res):
        raise NotImplementedException("Subclasses are responsible for creating this method")

    def train_with_file(self, files, new_day):
        raise NotImplementedException("Subclasses are responsible for creating this method")

    def train_with_obs(self, l_obs, l_res, new_day):
        raise NotImplementedException("Subclasses are responsible for creating this method")

class Benchmark_Koho(Benchmark):
    def __init__(self, nb_chan, group_by, input_classifier):
        super(Benchmark_Koho, self).__init__(nb_chan, group_by)
        self.input_count_classifier = input_classifier

    def init_classifier(self):
        my_bsc = bsc.brain_state_calculate(self.input_count_classifier, 'koho', self.ext_img, self.save_img, self.show)
        self.classifier = my_bsc

    def init_test(self, files):
        self.classifier.init_networks(files, self.my_cft, train_mod_chan=True)

    def test_network_with_files(self, files):
        l_res, l_obs = self.my_cft.read_cpp_files(files, is_healthy=False, cut_after_cue=False)
        return self.test_network_with_obs(l_obs, l_res)

    def test_network_with_obs(self, l_obs, l_res):
        #test and plot
        success, l_of_res = self.classifier.test(l_obs, l_res)
        return l_of_res

    def train_with_file(self, files, new_day):
        l_res, l_obs = self.my_cft.read_cpp_files(files, is_healthy=False, cut_after_cue=False)
        self.train_with_obs(l_obs, l_res, new_day)

    def train_with_obs(self, l_obs, l_res, new_day):
        if new_day:
            self.classifier.train_nets_new_day(l_obs, l_res, self.my_cft)
        self.classifier.train_nets(l_obs, l_res, self.my_cft, with_RL=True, obs_to_add=-1, train_mod_chan=True)
