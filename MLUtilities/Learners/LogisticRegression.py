import time
import math
import numpy
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryProbabilityEstimate as EvaluateBinaryProbabilityEstimate


class LogisticRegression(object):
    """Stub class for a Logistic Regression Model"""

    def __init__(self, featureCount=None):
        self.isInitialized = False

        if featureCount != None:
            self.__initialize(featureCount)

    def __testInput(self, x, y):
        if len(x) == 0:
            raise UserWarning(
                "Trying to fit but can't fit on 0 training samples.")

        if len(x) != len(y):
            raise UserWarning("Trying to fit but length of x != length of y.")

    def __initialize(self, featureCount):
        self.weights = [0.0 for i in range(featureCount)]
        self.weight0 = 0.0

        self.converged = False
        self.totalGradientDescentSteps = 0

        self.isInitialized = True

    def loss(self, x, y):
        return EvaluateBinaryProbabilityEstimate.LogLoss(y, self.predictProbabilities(x))

    def predictProbabilities(self, x):
        # For each sample do the dot product between features and weights (remember the bias weight, weight0)
        #  pass the results through the sigmoid function to convert to probabilities.
        return list(map(self.predictProbability, x))

    def predictProbability(self, x):
        sum = numpy.dot(self.weights, x) + self.weight0
        return self.sigmoid(sum)

    def predict(self, x, classificationThreshold=0.5):
        probabilities = self.predictProbabilities(x)
        return list(map(lambda prob: self.__classify(prob, classificationThreshold), probabilities))

    def __classify(self, value, classificationThreshold):
        if (value > classificationThreshold):
            return 1
        else:
            return 0

    def __gradientDescentStep(self, x, y, stepSize):
        self.totalGradientDescentSteps = self.totalGradientDescentSteps + 1
        yPredicted = self.predictProbabilities(x)

        partialDerivatives = self.__partialDerivatives(x, y, yPredicted)
        biasDerivative = self.__biasPartial(y, yPredicted)

        for i in range(len(partialDerivatives)):
            self.weights[i] -= stepSize * partialDerivatives[i]
        self.weight0 -= stepSize * biasDerivative

    def __partialDerivatives(self, x, y, yPredicted):
        partials = [0.0 for i in range(len(x[0]))]
        for i in range(len(x[0])):
            partials[i] = self.__partialDerivative(x, y, yPredicted, i)
        return partials

    def __partialDerivative(self, x, y, yPredicted, i):
        sum = 0
        for j in range(len(y)):
            Xj = x[j]
            Xji = Xj[i]
            yHatJ = yPredicted[j]
            Yj = y[j]

            inc = (yHatJ - Yj) * Xji
            sum += inc

        return sum/len(y)

    def __biasPartial(self, y, yPredicted):
        sum = 0
        Xji = 1
        for j in range(len(y)):
            yHatJ = yPredicted[j]
            Yj = y[j]

            inc = (yHatJ - Yj) * Xji
            sum += inc
        return sum/len(y)

    # Allows you to partially fit, then pause to gather statistics / output intermediate information, then continue fitting
    def incrementalFit(self, x, y, maxSteps=1, stepSize=1.0, convergence=0.005):
        self.__testInput(x, y)
        if self.isInitialized == False:
            self.__initialize(len(x[0]))

        loss = self.loss(x, y)
        for _ in range(maxSteps):
            self.__gradientDescentStep(x, y, stepSize)
            newLoss = self.loss(x, y)
            lossDiff = loss - newLoss
            if lossDiff < convergence:
                self.converged = True
                return
            loss = newLoss

    def fit(self, x, y, maxSteps=50000, stepSize=1.0, convergence=0.005, verbose=True):

        startTime = time.time()

        self.incrementalFit(x, y, maxSteps=maxSteps,
                            stepSize=stepSize, convergence=convergence)

        endTime = time.time()
        runtime = endTime - startTime

        if not self.converged:
            print(
                "Warning: did not converge after taking the maximum allowed number of steps.")
        elif verbose:
            print("LogisticRegression converged in %d steps (%.2f seconds) -- %d features. Hyperparameters: stepSize=%f and convergence=%f." %
                  (self.totalGradientDescentSteps, runtime, len(self.weights), stepSize, convergence))

    def visualize(self):
        print("w0: %f " % (self.weight0), end='')

        for i in range(len(self.weights)):
            print("w%d: %f " % (i+1, self.weights[i]), end='')

        print("\n")

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
