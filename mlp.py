import math
import random as rnd
import sys

class Neuron:
    def __init__(self, num, layer):
        self.number = num
        self.inputs = []
        self.outputs = []
        self.layer_number = layer
        self.output = 0.0
        self.sum = 0.0
        self.delta = 0.0

    def sigmoid(self, s):
        try:
            s = round(s, 6)
            return 1 / (1 + math.exp(-s))
        except OverflowError:
            print 'overflow'
            return sys.float_info.min

    def sigmoidDerivate(self, s):
        try:
            s = round(s, 6)
            return math.exp(-s)/((1+math.exp(-s))**2)
        except OverflowError:
            print 'overflow'
            return sys.float_info.min

    def computeOutput(self, weight_tab):
        self.sum = 0
        for i in range(len(self.inputs)):
            n = self.inputs[i]
            w = weight_tab[n.number][self.number]
            self.sum += w*n.output
        self.output = self.sigmoid(self.sum)

    def addInput(self, n):
        self.inputs.append(n)
        #self.weights.append(rnd.uniform(-1, 1))

class Network:
    def __init__(self, layer_info):
        #input_count = number of input including constant so +1
        #layer_count = number of layer including the input layer
        #n_by_layer = number of neurons by hidden layer
        self.neurons = []
        self.weight_tab = []
        self.layer_tab = []
        cpt = 0
        for i in range(len(layer_info)):
            self.layer_tab.append([])
            for j in range(layer_info[i]):
                n = Neuron(cpt, i)

                cpt += 1
                self.neurons.append(n)
                self.layer_tab[i].append(n)
                if i > 0:
                    for k in range(len(self.layer_tab[i-1])):
                        self.layer_tab[i-1][k].outputs.append(n)
                        n.addInput(self.layer_tab[i-1][k])

        self.neurons_count = len(self.neurons)
        for i in range(self.neurons_count):
            self.weight_tab.append([])
            for j in range(self.neurons_count):
                self.weight_tab[i].append(rnd.uniform(-1, 1))

    def reInit(self):
        for n in self.neurons:
            n.resetOutput()

    def processLayerK(self, k):
        for n in self.layer_tab[k]:
            n.computeOutput(self.weight_tab)


    def sumDelta(self, neuron):
        sum_delta = 0
        for n in neuron.outputs:
            sum_delta += self.weight_tab[neuron.number][n.number]*n.delta

        return sum_delta

    def backprop(self, alpha, l_obs, l_res):
        layer_count = len(self.layer_tab)

        for n in range(len(l_obs)):
            #neuron for constant
            self.layer_tab[0][0].output = 1
            obs = l_obs[n]
            res = l_res[n]
            self.run(obs)
            resultNet = self.output().index(max(self.output()))
            resultObs = res.index(max(res))
            #we adjust network only for bad decision
            if resultNet != resultObs:
                #compute delta for output layer
                for j in range(len(self.layer_tab[-1])):
                    neur = self.layer_tab[-1][j]
                    neur.delta = neur.sigmoidDerivate(neur.sum*(res[j]-neur.output))

                #we start at output layer and we go to input
                for k in range(layer_count-2, -1, -1):
                    #for each neuron in the layer k
                    for j in range(len(self.layer_tab[k])):
                        neur = self.layer_tab[k][j]
                        neur.delta = neur.sigmoidDerivate(neur.sum*self.sumDelta(neur))
                        #for each neuron of the next layer
                        for i in range(len(self.layer_tab[k+1])):
                            neurNext = self.layer_tab[k+1][i]
                            self.weight_tab[neur.number][neurNext.number] = self.weight_tab[neur.number][neurNext.number] + alpha * neur.output * neurNext.delta

    def run(self, obs):
        layer_count = len(self.layer_tab)
        self.layer_tab[0][0].output = 1

        #input neurons for observation
        for j in range(len(obs)):
            self.layer_tab[0][j+1].output = obs[j]

        #run mlp
        for k in range(layer_count-1):
            self.processLayerK(k+1)

    def output(self):
        n_out=self.layer_tab[-1]
        out = []
        for n in n_out:
            out.append(n.output)
        return out