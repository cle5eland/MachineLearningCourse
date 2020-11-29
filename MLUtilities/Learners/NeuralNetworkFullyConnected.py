import random
import math
import time
from numpy import dot

import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryProbabilityEstimate as EvaluateBinaryProbabilityEstimate


class Neuron(object):
    def __init__(self, numInputs: int):
        self.output = 0.0
        self.error = 0.0
        self.inputs = [0.0 for _ in range(numInputs)]
        self.initWeights(numInputs)

    def initWeights(self, numInputs: int):
        stdv = 1.0/(math.sqrt(numInputs + 1))
        self.bias: float = random.uniform(-stdv, stdv)
        self.weights: [float] = [random.uniform(-stdv, stdv)
                                 for inputID in range(numInputs)]

    def feed(self, x: [float]):
        self.inputs = x
        self.output = self.__calculateOutput(x)
        return self.output

    def backPropagate(self, index, nextLayer):
        total = 0.0
        for nextNeuron in nextLayer.neurons:
            total += nextNeuron.error * nextNeuron.weights[index]
        self.error = self.calculateError(total)
        return self.error

    def updateWeights(self, stepSize):
        self.bias += stepSize * self.error
        for i in range(len(self.weights)):
            self.weights[i] += stepSize * self.error * self.inputs[i]

    def __calculateOutput(self, x: [float]):
        product = dot(x, self.weights)
        product += self.bias
        return self.sigmoid(product)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def getWeights(self):
        result = [self.bias]
        result.extend(self.weights)
        return result

    def calculateError(self, downstreamSum: float):
        total = downstreamSum
        total *= self.output * (1 - self.output)
        return total


class OutputNeuron(Neuron):
    def __init__(self, numInputs: int):
        super().__init__(numInputs)

    def backPropagate(self, y):
        self.error = self.calculateError(y)
        return self.error

    def calculateError(self, y: int):
        return self.output * (1 - self.output) * (float(y) - self.output)


class NeuronLayer(object):
    def __init__(self, numNeurons: int, numInputs: int):
        self.neurons: [Neuron] = [Neuron(numInputs)
                                  for _ in range(numNeurons)]

    def feed(self, x):
        return [neuron.feed(x) for neuron in self.neurons]

    def backPropagate(self, nextLayer, y: int):
        for i in range(len(self.neurons)):
            self.neurons[i].backPropagate(i, nextLayer)

    def updateWeights(self, stepSize: float):
        for neuron in self.neurons:
            neuron.updateWeights(stepSize)

    def outputs(self):
        return [neuron.output for neuron in self.neurons]

    def width(self):
        return len(self.neurons)

    def getWeights(self):
        return [neuron.getWeights() for neuron in self.neurons]


class InputLayer(NeuronLayer):
    def __init__(self, width: int):
        self.o: [float] = [0.0
                           for _ in range(width)]

    def feed(self, x):
        self.o = x
        return self.o

    def outputs(self):
        return self.o

    def width(self):
        return len(self.o)

    def weights(self):
        print('BAD')

    def updateWeights(self, stepSize: float):
        pass

    def backPropagate(self, nextLayer: NeuronLayer, y: int):
        return


class OutputLayer(NeuronLayer):
    def __init__(self, numInputs: int):
        self.neurons: [Neuron] = [OutputNeuron(numInputs)]

    def backPropagate(self, nextLayer: NeuronLayer, y: int):
        self.neurons[0].backPropagate(y)


class NeuralNetworkFullyConnected(object):
    """Framework for fully connected neural network"""

    def __init__(self, numInputFeatures: int, hiddenLayersNodeCounts=[2], seed=1000):
        random.seed(seed)

        self.totalEpochs = 0
        self.lastLoss = None
        self.converged = False

        self.layers: [NeuronLayer] = [InputLayer(numInputFeatures)]
        for i in range(len(hiddenLayersNodeCounts)):
            numInputs = self.layers[-1].width()
            self.layers.append(NeuronLayer(
                numNeurons=hiddenLayersNodeCounts[i], numInputs=numInputs))

        self.layers.append(OutputLayer(numInputs=self.layers[-1].width()))

    def feedForward(self, x):
        currentOutput = x
        for layer in self.layers:
            currentOutput = layer.feed(currentOutput)

        return currentOutput[0]

    def backpropagate(self, y):
        nextLayer = None
        for layer in reversed(self.layers):
            layer.backPropagate(nextLayer, y)
            nextLayer = layer

    def updateweights(self, step):
        for layer in self.layers:
            layer.updateWeights(step)

    def loss(self, x, y):
        return EvaluateBinaryProbabilityEstimate.MeanSquaredErrorLoss(y, self.predictProbabilities(x))

    def predictOneProbability(self, x):
        result = self.feedForward(x)
        return result

    def predictProbabilities(self, x):
        return [self.predictOneProbability(sample) for sample in x]

    def predict(self, x, threshold=0.5):
        return [1 if probability > threshold else 0 for probability in self.predictProbabilities(x)]

    def __CheckForConvergence(self, x, y, convergence):
        loss = self.loss(x, y)

        if self.lastLoss != None:
            deltaLoss = abs(self.lastLoss - loss)
            self.converged = deltaLoss < convergence

        self.lastLoss = loss

    # Allows you to partially fit, then pause to gather statistics / output intermediate information, then continue fitting
    def incrementalFit(self, x, y, epochs=1, stepSize=0.01, convergence=None):
        for i in range(epochs):
            if self.converged:
                self.totalEpochs = i
                return

            # do a full epoch of stocastic gradient descent
            for i in range(len(x)):
                self.feedForward(x[i])
                self.backpropagate(y[i])
                # Mitchell p94 -- for stochastic, update weights after every training sample is processed
                self.updateweights(stepSize)

            if convergence != None:
                self.__CheckForConvergence(x, y, convergence)

    def fit(self, x, y, maxEpochs=50000, stepSize=0.01, convergence=0.00001, verbose=True):
        startTime = time.time()

        self.incrementalFit(x, y, epochs=maxEpochs,
                            stepSize=stepSize, convergence=convergence)

        endTime = time.time()
        runtime = endTime - startTime

        if not self.converged:
            print("WARNING -- NeuralNetwork did not converge. Details: %d epochs (%.2f seconds) -- %d features. Hyperparameters: stepSize=%f and convergence=%f." %
                  (maxEpochs, runtime, len(x[0]), stepSize, convergence))
        elif verbose:
            print("NeuralNetwork converged in %d epochs (%.2f seconds) -- %d features. Hyperparameters: stepSize=%f and convergence=%f." %
                  (self.totalEpochs, runtime, len(x[0]), stepSize, convergence))
