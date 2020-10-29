from collections import Counter
from itertools import chain
import math


class SMSSpamFeaturize(object):
    """A class to coordinate turning SMS spam strings into feature vectors."""

    def __init__(self, useHandCraftedFeatures=False):
        # use hand crafted features specified in _FeaturizeXForHandCrafted()
        self.useHandCraftedFeatures = useHandCraftedFeatures

        self.ResetVocabulary()

    def ResetVocabulary(self):
        self.vocabularyCreated = False
        self.vocabulary = []

    def Tokenize(self, xRaw=""):
        return str.split(xRaw)

    def TokenizeAll(self, items=[]):
        return list(map(self.Tokenize, items))

    def FindMostFrequentWords(self, x: [], n: int):
        allWords = self.flatten(self.TokenizeAll(x))
        return list(map(lambda tup: tup[0], Counter(allWords).most_common(n)))

    def FindTopWordsByMutualInformation(self, xRaw: [], y: [], n: int):
        if (n == 0):
            return []
        x: [[str]] = self.TokenizeAll(xRaw)
        # du-dup-ed array of all words:
        count = Counter()
        allWords = self.FindMostFrequentWords(xRaw, None)
        # loop over words (potential features) and calculate MI
        for i in range(len(allWords)):
            word = allWords[i]
            # put a 1 if the word is in the data sample, 0 otherwise
            xBinary = list(map(lambda val: self._categorize(word, val), x))
            mI = self._calculateMI(xBinary, y)
            count[word] = mI

        return list(map(lambda tup: tup[0], count.most_common(n)))

    def _categorize(self, val, x):
        if (val in x):
            return 1
        else:
            return 0

    def _calculateMI(self, x, y):
        if (len(x) != len(y)):
            raise UserWarning(
                "Array length mismatch.")
        cTable = self._buildContingencyTable(x, y)
        mi = 0
        denom = len(x) + 2
        for i in range(2):
            for j in range(2):
                pXY = (cTable[j][i] + 1)/denom
                pX = (cTable[0][i] + cTable[1][i] + 1)/denom
                pY = (cTable[j][0] + cTable[j][1] + 1)/denom
                val = pXY * math.log(pXY/(pX*pY))
                mi += val
        return mi

    def _buildContingencyTable(self, x, y):
        # Build Contingency Table

        cTable = [[0, 0],  # y = 0
                  [0, 0]]  # y = 1

        for i in range(len(x)):
            xVal = x[i]
            yVal = y[i]
            cTable[yVal][xVal] += 1

        return cTable

    def flatten(self, l: [[]]):
        return list(chain.from_iterable(l))

    def CreateVocabulary(self, xTrainRaw, yTrainRaw, numFrequentWords=0, numMutualInformationWords=0, supplementalVocabularyWords=[]):
        if self.vocabularyCreated:
            raise UserWarning(
                "Calling CreateVocabulary after the vocabulary was already created. Call ResetVocabulary to reinitialize.")

        # This function will eventually scan the strings in xTrain and choose which words to include in the vocabulary.
        #   But don't implement that until you reach the assignment that requires it...

        mostFrequentWords = self.FindMostFrequentWords(
            xTrainRaw, numFrequentWords)

        topMIWords = self.FindTopWordsByMutualInformation(
            xTrainRaw, yTrainRaw, numMutualInformationWords)
        # For now, only use words that are passed in
        self.vocabulary = self.vocabulary + \
            supplementalVocabularyWords + mostFrequentWords + topMIWords
        self.vocabularyCreated = True

    def _FeaturizeXForVocabulary(self, xRaw):
        features = []

        # for each word in the vocabulary output a 1 if it appears in the SMS string, or a 0 if it does not
        tokens = self.Tokenize(xRaw)
        for word in self.vocabulary:
            if word in tokens:
                features.append(1)
            else:
                features.append(0)

        return features

    def _FeaturizeXForHandCraftedFeatures(self, xRaw):
        features = []

        # This function can produce an array of hand-crafted features to add on top of the vocabulary related features
        if self.useHandCraftedFeatures:
            # Have a feature for longer texts
            if(len(xRaw) > 40):
                features.append(1)
            else:
                features.append(0)

            # Have a feature for texts with numbers in them
            if(any(i.isdigit() for i in xRaw)):
                features.append(1)
            else:
                features.append(0)

        return features

    def _FeatureizeX(self, xRaw):
        return self._FeaturizeXForVocabulary(xRaw) + self._FeaturizeXForHandCraftedFeatures(xRaw)

    def Featurize(self, xSetRaw):
        return [self._FeatureizeX(x) for x in xSetRaw]

    def GetFeatureInfo(self, index):
        if index < len(self.vocabulary):
            return self.vocabulary[index]
        else:
            # return the zero based index of the heuristic feature
            return "Heuristic_%d" % (index - len(self.vocabulary))
