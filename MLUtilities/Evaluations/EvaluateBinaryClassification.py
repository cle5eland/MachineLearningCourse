# This file contains stubs for evaluating binary classifications. You must complete these functions as part of your assignment.
#     Each function takes in:
#           'y':           the arrary of 0/1 true class labels;
#           'yPredicted':  the prediction your model made for the cooresponding example.


from tabulate import tabulate


def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning(
            "Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning(
            "Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be 0 or 1.")


def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)
    if len(correct) == 0:
        return None
    return sum(correct)/len(correct)


def Precision(y, yPredicted):
    [[trueNegatives, falsePositives], [falseNegatives,
                                       truePositives]] = ConfusionMatrix(y, yPredicted)
    denom = truePositives + falsePositives
    if denom == 0:
        return None
    return truePositives/(denom)


def Recall(y, yPredicted):
    [[trueNegatives, falsePositives], [falseNegatives,
                                       truePositives]] = ConfusionMatrix(y, yPredicted)
    denom = truePositives + falseNegatives
    if denom == 0:
        return None
    return truePositives/(denom)


def FalseNegativeRate(y, yPredicted):
    [[trueNegatives, falsePositives], [falseNegatives,
                                       truePositives]] = ConfusionMatrix(y, yPredicted)
    denom = falseNegatives + truePositives
    if denom == 0:
        return None
    return falseNegatives/(denom)


def FalsePositiveRate(y, yPredicted):
    [[trueNegatives, falsePositives], [falseNegatives,
                                       truePositives]] = ConfusionMatrix(y, yPredicted)
    denom = falsePositives + trueNegatives
    if denom == 0:
        return None

    return falsePositives/(denom)


def ConfusionMatrix(y, yPredicted):
    # This function should return: [[<# True Negatives>, <# False Positives>], [<# False Negatives>, <# True Positives>]]

    trueNegatives = falsePositives = falseNegatives = truePositives = 0

    for i in range(len(y)):
        if(y[i]):
            # Should be true
            if(yPredicted[i]):  # y = 1, yHat = 1
                truePositives += 1
            else:  # y = 1, yHat = 0
                falseNegatives += 1
        else:
            if(yPredicted[i]):  # y = 0, yHat = 1
                falsePositives += 1
            else:
                trueNegatives += 1  # y = 0, yHat = 0

    return [[trueNegatives, falsePositives], [falseNegatives, truePositives]]


def ExecuteAll(y, yPredicted):
    printConfusionMatrix(ConfusionMatrix(y, yPredicted))
    print("Accuracy:", Accuracy(y, yPredicted))
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))


def printConfusionMatrix(matrix):
    [[trueNegatives, falsePositives], [falseNegatives, truePositives]] = matrix
    [top, bottom] = matrix
    print("Confusion Matrix")
    print("[trueNegatives, falsePositives]")
    print("[falseNegatives, truePositives]")
    print(top)
    print(bottom)
