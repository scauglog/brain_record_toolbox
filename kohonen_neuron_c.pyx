#!python
#cython: boundscheck=False
#cython: wraparound=False

import random as rnd
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import csv
from itertools import combinations
import pickle
from scipy.stats.mstats import mquantiles


#standalone function
def rebuild_neuron(weights, weight_count, col, row, win_count):
    p=Neurone(weight_count, 0, col, row)
    p.weights= weights
    p.win_count=win_count
    return p

#standalone function
def rebuild_kohonen(row, col, neighbor, alpha, min_win, network, good_neurons, groups, img_ext, save, show):
    p = Kohonen(0, 0, 0, 0, alpha, neighbor, min_win, img_ext, save, show)
    p.row = row
    p.col = col
    p.network = network
    p.good_neurons = good_neurons
    p.groups = groups
    return p

def rebuild_group(neurons, template, color, number, spikes):
    p=Group_neuron(neurons[0], number)
    p.neurons=neurons
    p.template=template
    p.color=color
    p.spikes=spikes
    return p

cdef class Neurone:
    #cdef public int weight_count, col, row, win_count
    #cdef public np.ndarray weights
    def __init__(self,int weight_count, double max_rnd, int col=-1,int row=-1, seed=42):
        rnd.seed(seed)
        self.weights = np.array(map(float,range(weight_count)))
        self.weight_count = weight_count
        self.col = col
        self.row = row
        #number of time a neuron win, used for weighted mean when group neurons
        self.win_count = 0
        cdef int i
        for i in range(<int>self.weight_count):
            self.weights[i]=<double>rnd.uniform(-max_rnd, max_rnd)

    #inside the class definition
    def __reduce__(self):
        return (rebuild_neuron, (self.weights, self.weight_count, self.col, self.row, self.win_count))

    cpdef double calc_error(self, np.ndarray[DTYPE_t, ndim=1] obs):
        cdef double error_sum = 0.0
        cdef int i
        for i in range(<int> self.weight_count):
            error_sum += (<double>self.weights[i] - <double>obs[i]) ** 2
        return <double> math.sqrt(error_sum)

    def change_weights(self,double dist, np.ndarray[DTYPE_t, ndim=1] obs, double alpha):
        cdef int i
        for i in range(<int> self.weight_count):
            #if the neuron is not the best neuron dist > 1, dist is the neighborhood distance
            self.weights[i] -= alpha * ((<double> self.weights[i] - <double> obs[i]) * 1 / dist)

    cpdef double interneuron_dist(self,Neurone n2):
        return <double> self.weights_dist(n2.weights)

    cpdef double weights_dist(self, np.ndarray[DTYPE_t, ndim=1] w2):
        cdef double dist = 0
        cdef int i
        for i in range(self.weight_count):
            dist += (<double> self.weights[i] - <double> w2[i]) ** 2
        return <double> math.sqrt(dist)

cdef class Kohonen:
    #cdef public int row, col, neighbor, min_win
    #cdef public double alpha
    #cdef public object network, good_neurons, groups, img_ext, show, save
    def __init__(self, int row, int col, int weight_count, int max_weight, double alpha, int neighbor, int min_win, ext_img, save, show, seed=42):
        self.row = row
        self.col = col
        self.neighbor = neighbor
        self.alpha = alpha
        self.min_win = min_win
        self.network = []
        self.good_neurons = []
        self.groups = []
        self.img_ext = ext_img
        self.save = save
        self.show = show
        rnd.seed(seed)
        cdef int c, r
        for c in range(self.col):
            self.network.append([])
            for r in range(self.row):
                self.network[c].append(Neurone(weight_count, max_weight, c, r, rnd.random()))
    #inside the class definition
    def __reduce__(self):
        return (rebuild_kohonen, (self.row, self.col, self.neighbor, self.alpha, self.min_win, self.network, self.good_neurons, self.groups, self.img_ext, self.save, self.show))

    def algo_kohonen(self, obs_list, neighbor_decrease=True):
        for obs in obs_list:
            self.update_closest_neurons(obs, neighbor_decrease)

    def update_closest_neurons(self, np.ndarray[DTYPE_t, ndim=1] obs, neighbor_decrease=True, push_away=False):
        cdef Neurone best_n = self.find_best_neuron(obs)
        cdef int best_c = best_n.col
        cdef int best_r = best_n.row
        cdef int c,r
        cdef double dist
        cdef Neurone neur
        #update closest neurons weights and also weight of his neighbor
        for c in range(best_c - <int> self.neighbor, best_c + <int> self.neighbor):
            for r in range(best_r - <int> self.neighbor, best_r + <int> self.neighbor):
                if 0 <= c < <int> self.col and 0 <= r < <int> self.row:
                    if neighbor_decrease:
                        dist = 1.0 + <double> abs(best_c - c) + <double> abs(best_r - r)
                    else:
                        dist = 1.0
                    neur = <Neurone>self.network[c][r]
                    if push_away:
                        neur.change_weights(dist, obs, -self.alpha)
                    else:
                        neur.change_weights(dist, obs, self.alpha)

    #count the number of time each neurons win
    def compute_win_count(self, obs_list):
        win_count = []
        #init table
        cdef int c,r
        cdef Neurone neur, best_n
        for c in range(self.col):
            win_count.append([])
            for r in range(self.row):
                win_count[c].append(0)
                neur = <Neurone>self.network[c][r]
                neur.win_count = 0
        self.good_neurons = []
        #for each obs find the best neurons and update his win count
        for obs in obs_list:
            best_n = <Neurone> self.find_best_neuron(obs)
            win_count[best_n.col][best_n.row] += 1
            best_n.win_count += 1

        return win_count

    #return the best neurons for the obs
    cpdef Neurone find_best_neuron(self, np.ndarray[DTYPE_t, ndim=1] obs):
        cdef Neurone best_n = <Neurone>self.network[0][0]
        cdef double minerror = best_n.calc_error(obs)
        cdef int c,r
        cdef double error
        cdef Neurone n
        for c in range(self.col):
            for r in range(self.row):
                n = <Neurone>self.network[c][r]
                error = n.calc_error(obs)
                if error < minerror:
                    minerror = error
                    best_n = n
        return best_n

    cpdef double find_mean_best_dist(self, np.ndarray[DTYPE_t, ndim=1] obs, int elements_range):
        cdef np.ndarray[DTYPE_t, ndim=1] best_dist
        cdef int c, r, cpt
        cdef Neurone n
        best_dist=np.empty(self.col*self.row)
        cpt = 0
        for c in range(self.col):
            for r in range(self.row):
                n = <Neurone>self.network[c][r]
                best_dist[cpt]=<double>n.calc_error(obs)
                cpt+=1
        best_dist.sort()

        if <int> len(best_dist) > elements_range:
            best_dist = best_dist[0:elements_range]
        cdef double mean = <double> best_dist.mean()
        return mean

    def find_best_X_neurons(self, obs, int elements_range):
        best_n = []
        best_d = []
        cdef int c,r,arg_max_d
        cdef double d
        for c in range(self.col):
            for r in range(self.row):
                n = <Neurone>self.network[c][r]
                d = <double>n.calc_error(obs)
                best_n.append(n)
                best_d.append(d)
                if <int> len(best_d) > elements_range:
                    arg_max_d = <int>best_d.index(max(best_d))
                    del best_n[arg_max_d]
                    del best_d[arg_max_d]
        return best_n

    #for each neurons keep a predefined number of closest observation and compute average distance between neurons and observation
    def compute_density(self, obs_list, int elements_range):
        dens = []
        cdef int c,r
        for c in range(self.col):
            dens.append([])
            for r in range(self.row):

                list_dist = []
                for obs in obs_list:
                    list_dist.append(<double> self.network[c][r].calc_error(obs))
                list_dist.sort()
                if <int> len(list_dist) > elements_range:
                    elements_range = <int> len(list_dist)
                list_dist = list_dist[0:elements_range]
                #store mean of dist
                dens[c].append(reduce(lambda x, y: x + y, list_dist) / float(len(list_dist)))
        return dens

    #compute density to find center of the cluster. if we are in a local minimum of density then we are in the center of a cluster
    #TODO better way to find cluster center using density
    def find_cluster_center(self, obs_list, int elements_range):
        dens = self.compute_density(obs_list, elements_range)
        for c in range(self.col):
            for r in range(self.row):
                if 0 < c < (self.col - 1) and (self.row < 3):
                    if (dens[c][r] < dens[c - 1][r]) and (dens[c][r] < dens[c + 1][r]):
                        self.groups.append(Group_neuron(<Neurone>self.network[c][r], <int>len(self.groups)))

                if 0 < r < (self.row - 1) and (self.col < 3):
                    if (dens[c][r] < dens[c][r - 1]) and (dens[c][r] < dens[c][r + 1]):
                        self.groups.append(Group_neuron(<Neurone>self.network[c][r], <int>len(self.groups)))

                if 0 < r < (self.row - 1) and 0 < c < (self.col - 1) and self.col > 2 and self.row > 2:
                    if (dens[c][r] < dens[c - 1][r]) and (dens[c][r] < dens[c + 1][r]) and (dens[c][r] < dens[c][r - 1]) and (dens[c][r] < dens[c][r + 1]):
                        self.groups.append(Group_neuron(<Neurone>self.network[c][r], <int>len(self.groups)))

    #return neurons who win more than threshold (min_win)
    def evaluate_neurons(self, obs_list):
        self.compute_win_count(obs_list)
        cdef int c,r
        cdef Neurone neur
        for c in range(self.col):
            for r in range(self.row):
                neur = <Neurone>self.network[c][r]
                if <int> neur.win_count > <int> self.min_win:
                    self.good_neurons.append(neur)
        return self.good_neurons

    #silly way to group neurons, we find the closest neurons and if they are close enough we add them to the closest group
    def group_neurons(self, double dist_threshold):
        self.groups = []
        list_n = copy.copy(self.good_neurons)
        cdef double dist_n1_gpe, dist_n2_gpe
        while not <int> len(list_n) < 2:
            #search the most close neurons in the list
            n1, n2 = self.find_closest_neurons(list_n)

            #for the two closest neurons search the closest groups for each neurons
            dist_n1_gpe, best_gpe_n1 = self.find_closest_group(n1)
            self.classify_neuron(list_n, n1, dist_n1_gpe, best_gpe_n1, dist_threshold)

            dist_n2_gpe, best_gpe_n2 = self.find_closest_group(n2)
            self.classify_neuron(list_n, n2, dist_n2_gpe, best_gpe_n2, dist_threshold)

        print('groups found: ' + str(len(self.groups)))

    def group_neuron_into_x_class(self, int class_count):
        self.groups = []
        cdef int c, r
        cdef double dst, best_dst
        cdef Neurone n
        cdef Group_neuron best_g1
        cdef Group_neuron best_g2
        cdef Group_neuron g1
        cdef Group_neuron g2
        if <int> len(self.good_neurons) == 0:
            print('good_neurons is empty. all neurons are considered')
            for c in range(self.col):
                for r in range(self.row):
                    n = copy.copy(<Neurone>self.network[c][r])
                    self.groups.append(Group_neuron(<Neurone> n, <int> len(self.groups)))
        else:
            list_n = self.good_neurons
            for n in list_n:
                self.groups.append(Group_neuron(<Neurone> n, <int> len(self.groups)))

        while <int> len(self.groups) > class_count:
            best_g1 = self.groups[0]
            best_g2 = self.groups[1]
            best_dst = best_g1.dist(best_g2.template)
            for g1 in self.groups:
                for g2 in self.groups:
                    if g1 != g2:
                        dst = <double> g1.dist(g2.template)
                        if dst < best_dst:
                            best_dst = dst
                            best_g1 = g1
                            best_g2 = g2
            best_g1.merge_group(best_g2)
            self.groups.remove(best_g2)

    #find the closest neurons to another neurons (minimal distance between weight vector)
    def find_closest_neurons(self, list_n):
        first = True
        best_n1 = 0
        best_n2 = 0
        cdef double best_dist_n1_n2 = 42
        cdef double dist_n
        cdef Neurone n1, n2
        for n1 in list_n:
            for n2 in list_n:
                if not n1 == n2:
                    dist_n = <double> n1.interneuron_dist(n2)
                    if first:
                        best_dist_n1_n2 = dist_n
                        best_n1 = n1
                        best_n2 = n2
                        first = False
                    if dist_n < best_dist_n1_n2:
                        best_dist_n1_n2 = dist_n
                        best_n1 = n1
                        best_n2 = n2
        return best_n1, best_n2

    #find the closest group to a neuron
    def find_closest_group(self, neuron):
        first = True
        cdef double best_dist_neuron_gpe = 0
        cdef int best_gpe_neuron = 0
        cdef int gpe
        for gpe in range(<int>len(self.groups)):
            dist_neuron_gpe = neuron.weights_dist(self.groups[gpe].template)
            if first:
                best_dist_neuron_gpe = dist_neuron_gpe
                best_gpe_neuron = gpe
                first = False

            if dist_neuron_gpe < best_dist_neuron_gpe:
                best_dist_neuron_gpe = dist_neuron_gpe
                best_gpe_neuron = gpe

        return best_dist_neuron_gpe, best_gpe_neuron

    #put the neuron in the closest group if distance between neuron and group is above the threshold then create a new group
    def classify_neuron(self, list_n, Neurone neuron, double dist_neuron_gpe, int best_gpe_neuron, double dist_threshold):
        if <int>len(self.groups) == 0:
            #if no group create a new one
            self.groups.append(Group_neuron(neuron, <int>len(self.groups)))
        else:
            #if neuron is close to one group add it to this group else create a new group
            if dist_neuron_gpe < dist_threshold:
                self.groups[best_gpe_neuron].add_neuron(neuron)
                list_n.remove(neuron)
            else:
                self.groups.append(Group_neuron(neuron, <int>len(self.groups)))
                list_n.remove(neuron)

    #plot weight vector of the network in the same graph
    def plot_network(self, extra_text=''):
        plt.figure()
        plt.suptitle('all neurons weights' + extra_text)
        for c in range(self.col):
            for r in range(self.row):
                w = self.network[c][r].weights
                plt.plot(range(w.shape[0]), w)
        if self.save:
            plt.savefig('all_neurons_weights' + extra_text + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    def plot_network_array(self, extra_text=''):
        plt.figure()
        cpt=1
        for c in range(self.col):
            for r in range(self.row):
                w = self.network[c][r].weights
                plt.subplot(self.col, self.row, cpt)
                plt.plot(range(w.shape[0]), w)
                cpt += 1
        if self.save:
            plt.savefig('all_neurons_weights_a' + extra_text + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    def plot_network_dist(self, extra_text=''):
        plt.figure()
        net_dst = []
        for c in range(self.col):
            net_dst.append([])
            for r in range(self.row):
                dst = 0
                cpt = 0
                if 0 <= c-1 < self.col:
                    dst += self.network[c][r].interneuron_dist(self.network[c-1][r])
                    cpt += 1
                if 0 <= c+1 < self.col:
                    dst += self.network[c][r].interneuron_dist(self.network[c+1][r])
                    cpt += 1
                if 0 <= r-1 < self.row:
                    dst += self.network[c][r].interneuron_dist(self.network[c][r+1])
                    cpt += 1
                if 0 <= r+1 < self.row:
                    dst += self.network[c][r].interneuron_dist(self.network[c][r-1])
                    cpt += 1
                net_dst[c].append(dst/float(cpt))

        plt.imshow(net_dst, interpolation='none')
        plt.colorbar()
        if self.save:
            plt.savefig('network_dst_map' + extra_text + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    #same as plot_network (above) but only for the best neurons (win count > threshold)
    def plot_best_neurons(self, extra_text=''):
        plt.figure()
        plt.suptitle('best neurons weights' + extra_text)
        for n in self.good_neurons:
            w = n.weights
            plt.plot(range(w.shape[0]), w)
        if self.save:
            plt.savefig('best_neurons_weights' + extra_text + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    #plot templates of groups in the same graph (template=neuron weight vector average)
    def plot_groups(self, extra_text=''):
        if len(self.groups) != 0:
            plt.figure()
            plt.suptitle('group weights' + extra_text)
            for gpe in self.groups:
                w = gpe.template
                plt.plot(range(w.shape[0]), w, color=gpe.color)
            if self.save:
                plt.savefig('group_weights' + extra_text + self.img_ext, bbox_inches='tight')
            if not self.show:
                plt.close()

    #plot x(=spike_count) spike and color them according to the groups they belongs to
    def plot_spikes_classified(self, spikes_values, spike_count, threshold_template, extra_text=''):
        s = copy.copy(spikes_values).tolist()
        if spike_count > len(s):
            spike_count = len(s)

        plt.figure()
        plt.suptitle('spikes classified' + extra_text)
        for i in range(spike_count):
            #select a spike randomly
            r = rnd.randrange(len(s))
            value = s.pop(r)

            best_gpe = self.find_best_group(value, threshold_template)
            if best_gpe is None:
                color_gpe = (0, 0, 0)
            else:
                color_gpe = best_gpe.color

            plt.plot(range(len(value)), value, color=color_gpe)

        if self.save:
            plt.savefig('spikes_classified' + extra_text + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    #return the closest group of an observation
    def find_best_group(self, obs, double threshold_template=-1):
        # best_dist = threshold_template
        # best_gpe = None
        cdef double best_dist = self.groups[0].min_dist(obs)
        cdef double dist
        best_gpe = self.groups[0]
        for gpe in self.groups:
            dist = <double>gpe.dist(obs)
            if dist < best_dist:
                best_gpe = gpe
                best_dist = dist
        if 0 < threshold_template < best_dist:
            best_gpe = None
        return best_gpe

    #if a group of neuron don't win enough we delete the groups
    def evaluate_group(self, spikes_values, double threshold_template, int threshold_count):
        self.compute_groups_stat(spikes_values, threshold_template)
        tmp = []
        for gpe in self.groups:
            if <int>len(gpe.spikes) > threshold_count:
                tmp.append(gpe)
        self.groups = tmp

        print('groups found: ' + str(len(self.groups)))

    #put all observations in the group which they belongs to
    def compute_groups_stat(self, obs, dist_thresh):
        for spike in obs:
            gpe = self.find_best_group(spike, dist_thresh)
            if not (gpe is None):
                gpe.add_spike(spike)

    #plot group template and std (template=mean of weight vector of neurons)
    def plot_groups_stat(self, extra_text=''):
        if len(self.groups) != 0:
            plt.figure()

            for gpe in self.groups:
                x = range(gpe.template.shape[0])
                plt.plot(x, gpe.mean(), color=gpe.color)
                plt.plot(x, gpe.mean() - gpe.std(), '--', color=gpe.color)
                plt.plot(x, gpe.mean() + gpe.std(), '--', color=gpe.color)
            if self.save:
                plt.savefig('groups_mean_std' + extra_text + self.img_ext, bbox_inches='tight')
            if not self.show:
                plt.close()


cdef class Group_neuron:
    #cdef public int number
    #cdef public np.ndarray template
    #cdef public object plot_color, neurons, color, spikes
    def __init__(self, Neurone neuron, int gpe_count):
        plot_color = ['r', 'g', 'b', 'm', 'c', 'y']
        self.neurons = [neuron]
        self.template = neuron.weights
        self.color = plot_color[gpe_count % len(plot_color)]
        self.number = gpe_count
        self.spikes = []
        #standalone function

    #inside the class definition
    def __reduce__(self):
        return (rebuild_group, (self.neurons, self.template, self.color, self.number, self.spikes))

    def add_neuron(self, Neurone neuron):
        self.neurons.append(neuron)
        self.compute_template()

    def merge_group(self, Group_neuron group):
        cdef Neurone n
        for n in group.neurons:
            self.neurons.append(n)
        self.compute_template()

    def compute_template(self):
        cdef np.ndarray sum_template = <np.ndarray[DTYPE_t, ndim=1]> self.template * 0
        cdef int count = 0
        cdef Neurone n
        for n in self.neurons:
            sum_template += <np.ndarray[DTYPE_t, ndim=1]> n.weights * <int>n.win_count
            count += <int>n.win_count
        if count > 0:
            self.template = sum_template / count
        else:
            self.template *= 0

    cpdef double dist(self, np.ndarray[DTYPE_t, ndim=1] val):
        cdef double dist = 0
        cdef int i
        for i in range(self.template.shape[0]):
            dist += (<double> self.template[i] - <double> val[i]) ** 2
        return <double> math.sqrt(dist)

    cpdef double min_dist(self, np.ndarray[DTYPE_t, ndim=1] val):
        cdef double best_dist = <double>self.neurons[0].weights_dist(val)
        cdef double dist
        cdef Neurone n
        for n in self.neurons:
            dist = <double>n.weights_dist(val)
            if dist < best_dist:
                best_dist = dist
        return best_dist

    def add_spike(self, spike):
        self.spikes.append(spike)

    def std(self):
        return np.array(self.spikes).std(0)

    def mean(self):
        return np.array(self.spikes).mean(0)