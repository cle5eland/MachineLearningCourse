import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset
from typing import Callable


def __countCorrect(y, yPredicted):
    correct = 0
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct += 1
    return correct


def CrossValidation(x, y, numberOfFolds, foldIDToSelect):
    if(len(x) != len(y)):
        raise UserWarning(
            "Attempting to split into training and testing set.\n\tx and y arrays do not have the same size. Check your work and try again.")

    if(numberOfFolds <= 1 or numberOfFolds > len(y)):
        raise UserWarning(
            "Attempting to split into %d numberOfFolds, must be between 2 and the number of samples.\n." % numberOfFolds)

    if(foldIDToSelect < 0 or foldIDToSelect >= numberOfFolds):
        raise UserWarning("Attempting to select fold %d, must be 0 - %d." %
                          (foldIDToSelect, numberOfFolds - 1))

    numberPerFold = round(len(y) / numberOfFolds)

    xTrain = []
    yTrain = []
    xEvaluate = []
    yEvaluate = []

    for i in range(numberOfFolds):
        if i == foldIDToSelect:
            xThis = xEvaluate
            yThis = yEvaluate
        else:
            xThis = xTrain
            yThis = yTrain

        firstIndex = numberPerFold * i
        lastIndex = numberPerFold * (i+1)

        if i == (numberOfFolds - 1):
            # last, pick up any stragglers
            lastIndex = len(y)

        xThis += x[firstIndex:lastIndex]
        yThis += y[firstIndex:lastIndex]

    return(xTrain, yTrain, xEvaluate, yEvaluate)


def NewExecute(numberOfFolds: int, xTrain: [], yTrain: [], modelType, modelParams: dict, modelInitParams: dict, featurizerParams: dict, featurizerType, featureCreateMethod: str, xValidationRaw=None, yValidation=None):
    totalCorrect = 0
    totalTrainCorrect = 0
    for i in range(numberOfFolds):
        # Get data for fold
        (xTrainFold, yTrainFold, xEvaluate, yEvaluate) = CrossValidation(
            xTrain, yTrain, numberOfFolds, i) if numberOfFolds > 1 else (xTrain, yTrain, xValidationRaw, yValidation)
        # Feature Engineering
        featurizer = featurizerType()
        createFeatures = getattr(featurizer, featureCreateMethod)
        createFeatures(xTrain, yTrain, **featurizerParams)

        xTrainFold = featurizer.Featurize(xTrainFold)
        xEvaluate = featurizer.Featurize(xEvaluate)
        # Fit models
        model = modelType(**modelInitParams)
        model.fit(xTrainFold, yTrainFold, **modelParams)

        # Count accurate predictions
        totalCorrect += __countCorrect(yEvaluate,
                                       model.predict(xEvaluate))
        totalTrainCorrect += __countCorrect(yTrainFold,
                                            model.predict(xTrainFold))

    # Calculate total accuracy
    denom = len(xTrain) if numberOfFolds > 1 else len(yValidation)
    denomTrain = len(xTrain)

    logAccuracy = totalCorrect/denom
    trainAccuracy = totalTrainCorrect/denomTrain
    return {"accuracy": logAccuracy, "trainAccuracy": trainAccuracy}


def Execute(numberOfFolds: int, xTrain: [], yTrain: [], numMutualInformationWords: int, numFrequentWords: int, convergence: float, stepSize: float):
    modelParams = {}
    featureParams = {}
    featureParams['numMutualInformationWords'] = numMutualInformationWords
    modelParams['stepSize'] = stepSize
    modelParams['convergence'] = convergence
    featureParams['numFrequentWords'] = numFrequentWords
    return NewExecute(numberOfFolds, xTrain, yTrain, LogisticRegression.LogisticRegression, modelParams, featureParams, SMSSpamFeaturize.SMSSpamFeaturize, 'CreateVocabulary')
