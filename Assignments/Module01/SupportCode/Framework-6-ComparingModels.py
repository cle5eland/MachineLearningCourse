import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()


(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw,
 yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)


def countCorrect(y, yPredicted):
    correct = 0
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct += 1
    return correct


""" 

doModelEvaluation = False
if doModelEvaluation:
    ######
    # Build a model and evaluate on validation data
    # stepSize = 0.1
    # convergence = 0.0001
    stepSize = 1.0
    convergence = 0.001
    numMutualInformationWords = 25
    numFrequentWords = 0

    featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
    featurizer.CreateVocabulary(
        xTrainRaw, yTrain, numMutualInformationWords=25)

    xTrain = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest = featurizer.Featurize(xTestRaw)

    frequentModel = LogisticRegression.LogisticRegression()
    frequentModel.fit(xTrain, yTrain, convergence=convergence,
                      stepSize=stepSize, verbose=True)
    commonModel = MostCommonClassModel.MostCommonClassModel()
    commonModel.fit(xTrain, yTrain)
    ######
    # Use equation 5.1 from Mitchell to bound the validation set error and the true error
    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

    print("Logistic regression with 25 features by mutual information:")
    validationSetAccuracyLogistic = EvaluateBinaryClassification.Accuracy(
        yValidate, frequentModel.predict(xValidate))
    print("Common Model:")
    validationSetAccuracyCommon = EvaluateBinaryClassification.Accuracy(
        yValidate, commonModel.predict(xValidate))
    print("Validation set accuracy Logistic: %.4f." %
          (validationSetAccuracyLogistic))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(
            validationSetAccuracyLogistic, len(xValidate), confidence)
        print(" %.2f%% accuracy bound: %.4f - %.4f" %
              (confidence, lowerBound, upperBound))
    print("Validation set accuracy Common: %.4f." %
          (validationSetAccuracyCommon))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(
            validationSetAccuracyCommon, len(xValidate), confidence)
        print(" %.2f%% accuracy bound: %.4f - %.4f" %
              (confidence, lowerBound, upperBound))

    # Compare to most common class model here...

# Set this to true when you've completed the previous steps and are ready to move on...
doCrossValidation = True

"""
doCrossValidation = True

if doCrossValidation:
    import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
    numberOfFolds = 5

    stepSize = 1.0
    convergence = 0.001
    numMutualInformationWords = 25
    numFrequentWords = 0

    xTrainRaw.extend(xValidateRaw)
    xTrainAggregate = xTrainRaw
    yTrain.extend(yValidate)
    yTrainAggregate = yTrain

    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

    totalCorrectLogistic = 0
    totalCorrectCommon = 0
    for i in range(numberOfFolds):
        # Get data for fold
        (xTrainFold, yTrainFold, xEvaluate, yEvaluate) = CrossValidation.CrossValidation(
            xTrainAggregate, yTrainAggregate, numberOfFolds, i)
        # Feature Engineering
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
        featurizer.CreateVocabulary(
            xTrainFold, yTrainFold, numMutualInformationWords=25)
        xTrainFold = featurizer.Featurize(xTrainFold)
        xEvaluate = featurizer.Featurize(xEvaluate)
        logisticModel = LogisticRegression.LogisticRegression()
        commonModel = MostCommonClassModel.MostCommonClassModel()
        # Fit models
        logisticModel.fit(xTrainFold, yTrainFold, convergence=convergence,
                          stepSize=stepSize, verbose=True)
        commonModel.fit(xTrainFold, yTrainFold)

        # Count accurate predictions
        totalCorrectLogistic += countCorrect(yEvaluate,
                                             logisticModel.predict(xEvaluate))
        totalCorrectCommon += countCorrect(yEvaluate,
                                           commonModel.predict(xEvaluate))

    # Calculate total accuracy
    logAccuracy = totalCorrectLogistic/len(xTrainAggregate)
    commonAccuracy = totalCorrectCommon/len(xTrainAggregate)

    # Output accuracy and bounds for two-sided intervals
    print("Validation set accuracy Logistic: %.4f." %
          (logAccuracy))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(
            logAccuracy, len(xTrainAggregate), confidence)
        print(" %.2f%% accuracy bound: %.4f - %.4f" %
              (confidence, lowerBound, upperBound))
    print("Validation set accuracy Common: %.4f." %
          (commonAccuracy))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(
            commonAccuracy, len(xTrainAggregate), confidence)
        print(" %.2f%% accuracy bound: %.4f - %.4f" %
              (confidence, lowerBound, upperBound))
