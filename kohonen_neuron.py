import random as rnd
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


class Neurone:
    def __init__(self, weight_count, max_rnd):
        self.weights = []
        self.weight_count = weight_count
        #number of time a neuron win used for weighted mean when group neurons
        self.win_count = 0
        for i in range(self.weight_count):
            self.weights.append(rnd.uniform(-max_rnd, max_rnd))

    def calc_error(self, obs):
        error_sum = 0
        for i in range(self.weight_count):
            error_sum += (self.weights[i] - obs[i]) ** 2
        return math.sqrt(error_sum)

    def change_weights(self, dist, obs, alpha):
        for i in range(self.weight_count):
            #if the neuron is not the best neuron dist > 1, dist is the neighbor distance
            self.weights[i] -= alpha * ((self.weights[i] - obs[i]) * 1 / dist)

    def interneuron_dist(self, n2):
        return self.weights_dist(n2.weights)

    def weights_dist(self, w2):
        dist = 0
        for i in range(self.weight_count):
            dist += (self.weights[i] - w2[i]) ** 2
        return math.sqrt(dist)


class Kohonen:
    def __init__(self, row, col, weight_count, max_weight, alpha, neighbor, min_win, ext_img, save, show):
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
        for c in range(self.col):
            self.network.append([])
            for r in range(self.row):
                self.network[c].append(Neurone(weight_count, max_weight))

    def algo_kohonen(self, obs_list):
        for obs in obs_list:
            minerror = 1e6
            best_r = 0
            best_c = 0
            for c in range(self.col):
                for r in range(self.row):
                    error = self.network[c][r].calc_error(obs)
                    if error < minerror:
                        minerror = error
                        best_r = r
                        best_c = c
            for c in range(best_c - self.neighbor, best_c + self.neighbor):
                for r in range(best_r - self.neighbor, best_r + self.neighbor):
                    if 0 <= c < self.col and 0 <= r < self.row:
                        dist = 1.0 + abs(best_c - c) + abs(best_r - r)
                        self.network[c][r].change_weights(dist, obs, self.alpha)

    def compute_win_count(self, obs_list):
        win_count = []
        #init table
        for c in range(self.col):
            win_count.append([])
            for r in range(self.row):
                win_count[c].append(0)
                self.network[c][r].win_count = 0
        self.good_neurons = []
        #count the number of time a neuron win
        for obs in obs_list:
            minerror = 0
            best_r = 0
            best_c = 0
            first = True
            for c in range(self.col):
                for r in range(self.row):
                    error = self.network[c][r].calc_error(obs)
                    if first:
                        first = False
                        minerror = error

                    if error < minerror:
                        minerror = error
                        best_r = r
                        best_c = c
            win_count[best_c][best_r] += 1
            self.network[best_c][best_r].win_count += 1

        return win_count

    def compute_density(self, obs_list, elements_range):
        dens = []
        for c in range(self.col):
            dens.append([])
            for r in range(self.row):

                list_dist = []
                for obs in obs_list:
                    list_dist.append(self.network[c][r].calc_error(obs))
                list_dist.sort()
                if len(list_dist) > elements_range:
                    elements_range = len(list_dist)
                list_dist = list_dist[0:elements_range]
                #store mean of dist
                dens[c].append(reduce(lambda x, y: x + y, list_dist) / len(list_dist))
        return dens

    #compute density to find center of the cluster.
    #TODO better way to find cluster center using density
    def find_cluster_center(self, obs_list, elements_range):
        dens = self.compute_density(obs_list, elements_range)
        for c in range(self.col):
            for r in range(self.row):
                if 0 < c < (self.col - 1) and (self.row < 3):
                    if (dens[c][r] < dens[c - 1][r]) and (dens[c][r] < dens[c + 1][r]):
                        self.groups.append(Group_neuron(self.network[c][r], len(self.groups)))

                if 0 < r < (self.row - 1) and (self.col < 3):
                    if (dens[c][r] < dens[c][r - 1]) and (dens[c][r] < dens[c][r + 1]):
                        self.groups.append(Group_neuron(self.network[c][r], len(self.groups)))

                if 0 < r < (self.row - 1) and 0 < c < (self.col - 1) and self.col > 2 and self.row > 2:
                    if (dens[c][r] < dens[c - 1][r]) and (dens[c][r] < dens[c + 1][r]) and (dens[c][r] < dens[c][r - 1]) and (dens[c][r] < dens[c][r + 1]):
                        self.groups.append(Group_neuron(self.network[c][r], len(self.groups)))

    def best_neurons(self, obs_list):
        self.compute_win_count(obs_list)
        for c in range(self.col):
            for r in range(self.row):
                if self.network[c][r].win_count > self.min_win:
                    self.good_neurons.append(self.network[c][r])
        return self.good_neurons

    def group_neurons(self, dist_threshold):
        self.groups = []
        list_n = copy.copy(self.good_neurons)
        while not len(list_n) < 2:
            #search the most close neurons in the list
            n1, n2 = self.find_closest_neurons(list_n)

            #for the two closest neurons search the closest groups for each neurons
            dist_n1_gpe, best_gpe_n1 = self.find_closest_group(n1)
            self.classify_neuron(list_n, n1, dist_n1_gpe, best_gpe_n1, dist_threshold)

            dist_n2_gpe, best_gpe_n2 = self.find_closest_group(n2)
            self.classify_neuron(list_n, n2, dist_n2_gpe, best_gpe_n2, dist_threshold)

        print('groups found: ' + str(len(self.groups)))

    def find_closest_neurons(self, list_n):
        first = True
        best_n1 = 0
        best_n2 = 0
        best_dist_n1_n2 = 42
        for n1 in list_n:
            for n2 in list_n:
                if not n1 == n2:
                    dist_n = n1.interneuron_dist(n2)
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

    def find_closest_group(self, neuron):
        first = True
        best_dist_neuron_gpe = 0
        best_gpe_neuron = 0
        for gpe in range(len(self.groups)):
            dist_neuron_gpe = neuron.weights_dist(self.groups[gpe].template)
            if first:
                best_dist_neuron_gpe = dist_neuron_gpe
                best_gpe_neuron = gpe
                first = False

            if dist_neuron_gpe < best_dist_neuron_gpe:
                best_dist_neuron_gpe = dist_neuron_gpe
                best_gpe_neuron = gpe

        return best_dist_neuron_gpe, best_gpe_neuron

    def classify_neuron(self, list_n, neuron, dist_neuron_gpe, best_gpe_neuron, dist_threshold):
        if len(self.groups) == 0:
            #if no group create a new one
            self.groups.append(Group_neuron(neuron, len(self.groups)))
        else:
            #if neuron is close to one group add it to this group else create a new group
            if dist_neuron_gpe < dist_threshold:
                self.groups[best_gpe_neuron].add_neuron(neuron)
                list_n.remove(neuron)
            else:
                self.groups.append(Group_neuron(neuron, len(self.groups)))
                list_n.remove(neuron)

    def plot_network(self, extra_text=''):
        plt.figure()
        plt.suptitle('all neurons weights' + extra_text)
        for c in range(self.col):
            for r in range(self.row):
                w = self.network[c][r].weights
                plt.plot(range(len(w)), w)
        if self.save:
            plt.savefig('all_neurons_weights' + extra_text + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    def plot_best_neurons(self, extra_text=''):
        plt.figure()
        plt.suptitle('best neurons weights' + extra_text)
        for n in self.good_neurons:
            w = n.weights
            plt.plot(range(len(w)), w)
        if self.save:
            plt.savefig('best_neurons_weights' + extra_text + self.img_ext, bbox_inches='tight')
        if not self.show:
            plt.close()

    def plot_groups(self, extra_text=''):
        if len(self.groups) != 0:
            plt.figure()
            plt.suptitle('group weights' + extra_text)
            for gpe in self.groups:
                w = gpe.template
                plt.plot(range(len(w)), w, color=gpe.color)
            if self.save:
                plt.savefig('group_weights' + extra_text + self.img_ext, bbox_inches='tight')
            if not self.show:
                plt.close()

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

    def find_best_group(self, obs, threshold_template):
        best_dist = threshold_template
        best_gpe = None
        for gpe in self.groups:
            dist = gpe.dist(obs)
            if dist < best_dist:
                best_gpe = gpe
                best_dist = dist
        return best_gpe

    def evaluate_group(self, spikes_values, threshold_template, threshold_count):
        self.compute_groups_stat(spikes_values, threshold_template)
        tmp = []
        for gpe in self.groups:
            if len(gpe.spikes) > threshold_count:
                tmp.append(gpe)
        self.groups = tmp

        print('groups found: ' + str(len(self.groups)))

    def compute_groups_stat(self, obs, dist_thresh):
        for spike in obs:
            gpe = self.find_best_group(spike, dist_thresh)
            if not (gpe is None):
                gpe.add_spike(spike)

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


class Group_neuron:
    def __init__(self, neuron, gpe_count):
        plot_color = ['r', 'g', 'b', 'm', 'c', 'y']
        self.neurons = [neuron]
        self.template = np.array(neuron.weights)
        self.color = plot_color[gpe_count % len(plot_color)]
        self.number = gpe_count
        self.spikes = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        self.compute_template()

    def compute_template(self):
        sum_template = self.template * 0
        count = 0
        for n in self.neurons:
            sum_template += np.array(n.weights) * n.win_count
            count += n.win_count
        self.template = sum_template / count

    def dist(self, val):
        dist = 0
        for i in range(len(self.template)):
            dist += (self.template[i] - val[i]) ** 2
        return math.sqrt(dist)

    def add_spike(self, spike):
        self.spikes.append(spike)

    def std(self):
        return np.array(self.spikes).std(0)

    def mean(self):
        return np.array(self.spikes).mean(0)