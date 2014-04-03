import math
import random as rnd
import decimal

class Neuron:
    def __init__(self, num, layer):
        self.number = num
        self.inputs = []
        self.outputs = []
        self.layer_number = layer
        self.output = decimal.Decimal(0)
        self.sum = decimal.Decimal(0)
        self.delta = decimal.Decimal(0)

    def sigmoid(self, s):
        return decimal.Decimal(1 / (1 + math.exp(-s)))

    def sigmoidDerivate(self, s):
        return decimal.Decimal(math.exp(-s)/(1+math.exp(-s)**2))

    def computeOutput(self, weight_tab):
        self.sum = decimal.Decimal(0)
        for i in range(len(self.inputs)):
            n = self.inputs[i]
            w = weight_tab[n.number][self.number]
            self.sum = self.sum + w*n.output
        self.output = self.sigmoid(self.sum)

class Network:
    def __init__(self, input_count, output_count, layer_count, n_by_layer):
        #input_count = number of input including constant so +1
        #layer_count = number of layer including the input layer
        #n_by_layer = number of neurons by hidden layer
        self.neurons = []
        self.weight_tab = []
        self.layer_tab = []

        #init list
        for i in range(layer_count):
            self.layer_tab.append([])
        cpt = 0

        #init input layer
        for i in range(input_count):
            n = Neuron(cpt, 0)
            self.neurons.append(n)
            self.layer_tab[0].append(n)
            cpt += 1

        #init hidden layer
        for i in range(1, layer_count-1):
            for j in range(n_by_layer):
                n = Neuron(cpt, i)

                cpt += 1
                self.neurons.append(n)
                self.layer_tab[i].append(n)
                for k in range(len(self.layer_tab[i-1])):
                    self.layer_tab[i-1][k].outputs.append(n)
                    n.inputs.append(self.layer_tab[i-1][k])

        #init output layer
        for i in range(output_count):
            n = Neuron(cpt, 0)
            cpt += 1
            self.neurons.append(n)
            self.layer_tab[-1].append(n)
            for k in range(len(self.layer_tab[-2])):
                self.layer_tab[-2][k].outputs.append(n)
                n.inputs.append(self.layer_tab[-2][k])

        self.neurons_count = len(self.neurons)
        for i in range(self.neurons_count):
            self.weight_tab.append([])
            for j in range(self.neurons_count):
                self.weight_tab[i].append(decimal.Decimal(rnd.uniform(-1, 1)))

    def reInit(self):
        for n in self.neurons:
            n.resetOutput()

    def processLayerK(self, k):
        for n in self.layer_tab[k]:
            n.computeOutput(self.weight_tab)


    def sumDelta(self, neuron):
        sum_delta = decimal.Decimal(0)
        for n in neuron.outputs:
            sum_delta = sum_delta + self.weight_tab[neuron.number][n.number]*n.delta

        return sum_delta

    def backprop(self, alpha, l_obs, l_res):
        layer_count = len(self.layer_tab)

        for n in range(len(l_obs)):
            #neuron for constant
            self.layer_tab[0][0].output = 1
            obs = l_obs[n]
            res = l_res[n]
            self.run(obs)

            #compute delta for output layer
            for j in range(len(self.layer_tab[-1])):
                neur = self.layer_tab[-1][j]
                neur.delta = neur.sigmoidDerivate(neur.sum*(res[j]-neur.output))

            #we start at output layer and we go to input
            for k in range(layer_count-2, -1, -1):
                #for each neuron in the layer k
                for j in range(len(self.layer_tab[k])):
                    neur = self.layer_tab[k][j]
                    neur.dalta = neur.sigmoidDerivate(neur.sum*self.sumDelta(neur))
                    #for each neuron of the next layer
                    for i in range(len(self.layer_tab[k+1])):
                        neurNext = self.layer_tab[k+1][i]
                        self.weight_tab[neur.number][neurNext.number] = self.weight_tab[neur.number][neurNext.number] + decimal.Decimal(alpha)*neur.output * neurNext.delta

    def run(self, obs):
        layer_count = len(self.layer_tab)
        self.layer_tab[0][0].output = 1

        #input neurons for observation
        for j in range(len(obs)):
            self.layer_tab[0][j+1].output = obs[j]

        #run mlp
        for k in range(layer_count):
            self.processLayerK(k)

    def output(self):
        n_out=self.layer_tab[-1]
        out = []
        for n in n_out:
            out.append(n.output)
        return out