import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset


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


def Execute(numberOfFolds: int, xTrainAggregate: [], yTrainAggregate: [], numMutualInformationWords: int, numFrequentWords: int, convergence: float, stepSize: float):
    totalCorrect = 0
    for i in range(numberOfFolds):
        # Get data for fold
        (xTrainFold, yTrainFold, xEvaluate, yEvaluate) = CrossValidation(
            xTrainAggregate, yTrainAggregate, numberOfFolds, i)
        # Feature Engineering
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
        featurizer.CreateVocabulary(
            xTrainFold, yTrainFold, numMutualInformationWords=numMutualInformationWords, numFrequentWords=numFrequentWords)
        xTrainFold = featurizer.Featurize(xTrainFold)
        xEvaluate = featurizer.Featurize(xEvaluate)
        logisticModel = LogisticRegression.LogisticRegression()
        # Fit models
        logisticModel.fit(xTrainFold, yTrainFold, convergence=convergence,
                          stepSize=stepSize, verbose=True)

        # Count accurate predictions
        totalCorrect += __countCorrect(yEvaluate,
                                       logisticModel.predict(xEvaluate))

    # Calculate total accuracy
    logAccuracy = totalCorrect/len(xTrainAggregate)
    return logAccuracy
