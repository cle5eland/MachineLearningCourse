import math


def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning(
            "Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value < 0 or value > 1:
            valueError = True
    for value in yPredicted:
        if value < 0 or value > 1:
            valueError = True

    if valueError:
        raise UserWarning(
            "Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be between 0 and 1.")


def MeanSquaredErrorLoss(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    total = 0.0
    for i in range(len(y)):
        diff = yPredicted[i] - y[i]
        total += diff * diff

    return total/2.0


def LogLoss(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    sum = 0
    for i in range(len(y)):
        sum += IndividualLogLoss(y[i], yPredicted[i])

    return sum/len(y)


smallValue = 0  # 1e-15


def IndividualLogLoss(y, yPredicted):
    return -y * math.log(yPredicted + smallValue) - ((1-y)*math.log(1-yPredicted + smallValue))
