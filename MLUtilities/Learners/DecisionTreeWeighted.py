from MachineLearningCourse.MLUtilities.Learners.DecisionTree import DecisionTree, TreeNode
import collections
from collections import Counter
import math
import time
from math import log2
from tabulate import tabulate
from numpy import dot


def Entropy(y: [int], w: [float]):
    totalWeight = sum(w)
    if totalWeight < 0.0000001:
        return 0.0
    counter = Counter(y)
    labelWeights = {}
    for (value, _) in counter.items():
        labelWeights[value] = 0.0
    for i in range(len(y)):
        labelWeights[y[i]] += w[i]
    result = 0.0
    for (_, weight) in labelWeights.items():
        if weight < 0.0000001:
            continue
        p = weight/totalWeight
        result += p * log2(p)

    return result * -1.0


def SplitEntropy(x, y, w, index):
    result = 0
    fHX = x[:index]
    sHX = x[index:]
    firstHalf = y[:index]
    secondHalf = y[index:]
    fHW = w[:index]
    sHW = w[index:]

    ws = [fHW, sHW]
    ys = [firstHalf, secondHalf]
    for i in range(len(ys)):
        # TODO: something not quite right here
        result += Entropy(ys[i], ws[i]) * sum(ws[i])
    entropy = result/sum(w)
    splitX = [fHX, sHX]
    splitY = [firstHalf, secondHalf]
    splitW = [fHW, sHW]
    return (entropy, splitX, splitY, splitW)


def InformationGain(entropyY, x, y, w, index):
    (entropy, splitX, splitY, splitW) = SplitEntropy(x, y, w, index)
    IG = entropyY - entropy
    return (IG, splitX, splitY, splitW)


def FindBestSplitOnFeature(x, y, w, featureIndex):
    if len(y) < 2:
        # there aren't enough samples so there is no split to make
        return None

    # Do more tests to make sure you haven't hit a terminal case...
    indexesInSortedOrder = sorted(
        range(len(x)), key=lambda i: x[i][featureIndex])

    sortedX = list(map(lambda i: x[i], indexesInSortedOrder))
    sortedY = list(map(lambda i: y[i], indexesInSortedOrder))
    sortedW = list(map(lambda i: w[i], indexesInSortedOrder))

    totalEntropy = Entropy(y, w)

    bestThreshold = 0
    splitData = []
    gain = 0
    lastValidIndex = None

    for i in range(len(x) - 1):
        (currY, nextY) = (sortedY[i], sortedY[i+1])
        (currX, nextX) = [sortedX[i][featureIndex], sortedX[i+1][featureIndex]]

        if (currX != nextX):
            # we could split here if we wanted to
            lastValidIndex = i
        if (currX != nextX and currY == nextY):
            # this is not optimal. We can continue searching (though we could calculate)
            continue

        if lastValidIndex == None:
            # is this bad?
            continue

        (IG, splitX, splitY, splitW) = InformationGain(
            totalEntropy, sortedX, sortedY, sortedW, lastValidIndex+1)
        if (IG > gain):
            bestThreshold = (float(currX + nextX))/2.0
            splitData = [splitX, splitY, splitW]
            gain = IG

    if lastValidIndex != None:
        (currX, nextX) = [sortedX[lastValidIndex][featureIndex],
                          sortedX[lastValidIndex+1][featureIndex]]
        (IG, splitX, splitY, splitW) = InformationGain(
            totalEntropy, sortedX, sortedY, sortedW, lastValidIndex+1)
        if (IG > gain):
            bestThreshold = (float(currX + nextX))/2.0
            splitData = [splitX, splitY, splitW]
            gain = IG

    return (bestThreshold, splitData, gain)


class WeightedTreeNode(TreeNode):
    def __init__(self, depth=0):
        super().__init__(depth)
        self.weights = []
        self.prior = 1

    def findBestSplitOnFeature(self, i):
        return FindBestSplitOnFeature(
            self.x, self.y, self.weights, i)

    def __leafProbability(self, x):
        weight = dot(self.y, self.weights)
        print('dot weight', weight)
        return (weight + 1.0 * self.prior)/(sum(self.weights) + 2.0 * self.prior)

    def createNodes(self, splitData):
        leftNode = WeightedTreeNode(self.depth+1)
        rightNode = WeightedTreeNode(self.depth+1)

        [splitX, splitY, splitWeights] = splitData

        leftNode.addData(splitX[0], splitY[0])
        rightNode.addData(splitX[1], splitY[1])
        leftNode.addWeights(splitWeights[0], self.prior)
        rightNode.addWeights(splitWeights[1], self.prior)
        return (leftNode, rightNode)

    def addWeights(self, weights: [float], prior: float):
        self.prior = prior
        self.weights = weights


class DecisionTreeWeighted(DecisionTree):
    """Weighted Decision Tree."""

    def __init__(self):
        super().__init__()

    def fit(self, x: [[float]], y: [int], maxDepth=10000, weights: [float] = None, verbose=True):
        if weights == None:
            # Just do normal decision tree
            return super().fit(x, y, maxDepth=maxDepth, verbose=verbose)

        self.maxDepth = maxDepth

        startTime = time.time()

        self.treeNode = WeightedTreeNode(depth=0)

        self.treeNode.addData(x, y)
        self.treeNode.addWeights(weights, 1.0/float(len(y)))
        self.treeNode.growTree(maxDepth)

        endTime = time.time()
        runtime = endTime - startTime

        if verbose:
            print("Decision Tree completed with %d nodes (%.2f seconds) -- %d features. Hyperparameters: maxDepth=%d." %
                  (self.countNodes(), runtime, len(x[0]), maxDepth))
