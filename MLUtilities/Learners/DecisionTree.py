import collections
from collections import Counter
import math
import time
from math import log2
from tabulate import tabulate


def printFeatureData(x, y, featureIndex):
    indexesInSortedOrder = sorted(
        range(len(x)), key=lambda i: x[i][featureIndex])

    sortedX = list(map(lambda i: x[i], indexesInSortedOrder))
    sortedY = list(map(lambda i: y[i], indexesInSortedOrder))

    table = []
    for i in range(len(sortedX)):
        row = []
        row.extend(sortedX[i])
        row.append(sortedY[i])
        table.append(row)

    headers = ["x%s" % (i) for i in range(len(x[0]))]
    headers.append('y')

    print(tabulate(table, headers, tablefmt="github"))


def Entropy(y: []):
    counter = Counter(y)
    totalCount = float(len(y))
    result = 0
    for (_, count) in counter.items():
        count = float(count)
        p = count/totalCount
        result += p * log2(p)

    return result * -1.0


def SplitEntropy(x, y, index):
    result = 0
    fHX = x[:index]
    sHX = x[index:]
    firstHalf = y[:index]
    secondHalf = y[index:]

    ys = [firstHalf, secondHalf]
    for ySection in ys:
        result += Entropy(ySection) * float(len(ySection))
    entropy = result/float(len(y))
    splitX = [fHX, sHX]
    splitY = [firstHalf, secondHalf]
    return (entropy, splitX, splitY)


def InformationGain(entropyY, x, y, index):
    (entropy, splitX, splitY) = SplitEntropy(x, y, index)
    IG = entropyY - entropy
    return (IG, splitX, splitY)


def FindBestSplitOnFeature(x, y, featureIndex):
    if len(y) < 2:
        # there aren't enough samples so there is no split to make
        return None

    # Do more tests to make sure you haven't hit a terminal case...
    indexesInSortedOrder = sorted(
        range(len(x)), key=lambda i: x[i][featureIndex])

    sortedX = list(map(lambda i: x[i], indexesInSortedOrder))
    sortedY = list(map(lambda i: y[i], indexesInSortedOrder))

    totalEntropy = Entropy(y)

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

        (IG, splitX, splitY) = InformationGain(
            totalEntropy, sortedX, sortedY, lastValidIndex+1)
        if (IG > gain):
            bestThreshold = (float(currX + nextX))/2.0
            splitData = [splitX, splitY]
            gain = IG

    if lastValidIndex != None:
        (IG, splitX, splitY) = InformationGain(
            totalEntropy, sortedX, sortedY, lastValidIndex+1)
        if (IG > gain):
            bestThreshold = (float(currX + nextX))/2.0
            splitData = [splitX, splitY]
            gain = IG

    return (bestThreshold, splitData, gain)


class TreeNode(object):
    def __init__(self, depth=0):
        self.depth = depth
        self.labelDistribution = collections.Counter()
        self.splitIndex = None
        self.threshold = None
        self.children = []
        self.x = []
        self.y = []

    def isLeaf(self):
        return self.splitIndex == None

    def addData(self, x, y):
        self.x += x
        self.y += y

        for label in y:
            self.labelDistribution[label] += 1

    def growTree(self, maxDepth):
        if self.depth == maxDepth:
            # max recursion depth
            return self
        print('depth ', self.depth)
        if len(self.labelDistribution.items()) <= 1:
            # leaf. TODO: maybe explicitly set, probably just leave as is
            return self

        (maxFeature, bestThreshold, splitData, IG) = self.bestSplitAttribute()
        if IG == 0:
            # terminate
            return self

        self.splitIndex = maxFeature
        self.threshold = bestThreshold

        leftNode = TreeNode(self.depth+1)
        rightNode = TreeNode(self.depth+1)

        [splitX, splitY] = splitData

        leftNode.addData(splitX[0], splitY[0])
        rightNode.addData(splitX[1], splitY[1])
        leftNode.growTree(maxDepth)
        rightNode.growTree(maxDepth)
        self.children = [leftNode, rightNode]

        return self

    def bestSplitAttribute(self):
        informationGains = {}
        results = {}

        for i in range(len(self.x[0])):
            print('checking feature: ', i)
            (bestThreshold, splitData, IG) = FindBestSplitOnFeature(
                self.x, self.y, i)
            informationGains[i] = IG
            results[i] = (bestThreshold, splitData, IG)

        all0 = all(value == 0 for value in informationGains.values())
        if all0:
            # Not 100p on this one
            return (None, None, None, 0)

        maxFeature = max(informationGains, key=informationGains.get)
        (bestThreshold, splitData, IG) = results[maxFeature]
        return (maxFeature, bestThreshold, splitData, IG)

    def predictProbability(self, x):
        # Remember to find the correct leaf then use an m-estimate to smooth the probability:
        #  (#_with_label_1 + 1) / (#_at_leaf + 2)
        if (self.isLeaf()):
            return (float(self.labelDistribution[1] + 1))/float(len(self.y) + 2)
        if x[self.splitIndex] < self.threshold:
            return self.children[0].predictProbability(x)
        else:
            return self.children[1].predictProbability(x)

    def visualize(self, depth=1):
        # Here is a helper function to visualize the tree (if you choose to use the framework class)
        if self.isLeaf():
            print(self.labelDistribution)

        else:
            print("Split on: %d" % (self.splitIndex))

            # less than
            for _ in range(depth):
                print(' ', end='', flush=True)
            print("< %f -- " % self.threshold, end='', flush=True)
            self.children[0].visualize(depth+1)

            # greater than or equal
            for _ in range(depth):
                print(' ', end='', flush=True)
            print(">= %f -- " % self.threshold, end='', flush=True)
            self.children[1].visualize(depth+1)

    def countNodes(self):
        if self.isLeaf():
            return 1

        else:
            return 1 + self.children[0].countNodes() + self.children[1].countNodes()


class DecisionTree(object):
    """Wrapper class for decision tree learning."""

    def __init__(self):
        pass

    def fit(self, x, y, maxDepth=10000, verbose=True):
        self.maxDepth = maxDepth

        startTime = time.time()

        self.treeNode = TreeNode(depth=0)

        self.treeNode.addData(x, y)
        self.treeNode.growTree(maxDepth)

        endTime = time.time()
        runtime = endTime - startTime

        if verbose:
            print("Decision Tree completed with %d nodes (%.2f seconds) -- %d features. Hyperparameters: maxDepth=%d." %
                  (self.countNodes(), runtime, len(x[0]), maxDepth))

    def predictProbabilities(self, x):
        y = []

        for example in x:
            y.append(self.treeNode.predictProbability(example))

        return y

    def predict(self, x, classificationThreshold=0.5):
        return [1 if probability >= classificationThreshold else 0 for probability in self.predictProbabilities(x)]

    def visualize(self):
        self.treeNode.visualize()

    def countNodes(self):
        return self.treeNode.countNodes()
