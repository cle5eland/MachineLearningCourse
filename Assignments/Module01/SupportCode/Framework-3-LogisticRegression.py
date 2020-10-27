
kOutputDirectory = ""

runUnitTest = False
if runUnitTest:
    # Little synthetic dataset to help with implementation. 2 features, 8 samples.
    xTrain = [[.1, .1], [.2, .2], [.2, .1], [.1, .2],
              [.95, .95], [.9, .8], [.8, .9], [.7, .6]]
    yTrain = [0, 0, 0, 0, 1, 1, 1, 1]

    # create a linear model with the right number of weights initialized
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    model = LogisticRegression.LogisticRegression(featureCount=len(xTrain[0]))

    # To use this visualizer you need to install the PIL imaging library. Instructions are in the lecture notes.
    import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

    while not model.converged:
        # do 10 iterations of training
        model.incrementalFit(xTrain, yTrain, maxSteps=10,
                             stepSize=1.0, convergence=0.005)

        # then look at the models weights
        model.visualize()

        # then look at how training set loss is converging
        print(" fit for %d iterations, train set loss is %.4f" %
              (model.totalGradientDescentSteps, model.loss(xTrain, yTrain)))

        # and visualize the model's decision boundary
        visualization = Visualize2D.Visualize2D(
            kOutputDirectory, "{0:04}.test".format(model.totalGradientDescentSteps))
        visualization.Plot2DDataAndBinaryConcept(xTrain, yTrain, model)
        visualization.Save()


def simpleModel(xRaw):
    return list(map(evalSampleSimple, xRaw))


def evalSampleSimple(x):
    if ("Call" in x and "FREE" in x):
        return 1
    if ("mobile" in x and "claim" in x):
        return 1
    if ("&" in x and "Call" in x):
        return 1
    return 0


# Once your LogisticRegression learner seems to be working, set this flag to True and try it on the spam data
runSMSSpam = False
if runSMSSpam:
    import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

    ############################
    # Set up the data

    (xRaw, yRaw) = SMSSpamDataset.LoadRawData()

    import MachineLearningCourse.MLUtilities.Data.Sample as Sample
    (xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw,
     yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

    import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(
        useHandCraftedFeatures=False)
    featurizer.CreateVocabulary(
        xTrainRaw, yTrain, numMutualInformationWords=10)

    xTrain = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest = featurizer.Featurize(xTestRaw)

    #############################
    # Learn the logistic regression model

    print("Learning the logistic regression model:")
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    logisticRegressionModel = LogisticRegression.LogisticRegression()

    logisticRegressionModel.fit(
        xTrain, yTrain, stepSize=1.0, convergence=0.001)

    #############################
    # Evaluate the model

    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

    from tabulate import tabulate

    table = []
    for i in range(len(logisticRegressionModel.weights)):
        table.append([featurizer.vocabulary[i],
                      logisticRegressionModel.weights[i]])

    headers = ["words", "weights"]

    print(tabulate(table, headers, tablefmt="github"))

    print("\nLogistic regression model:")
    logisticRegressionModel.visualize()
    EvaluateBinaryClassification.ExecuteAll(
        yValidate, logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))

    print("Simple model:")
    EvaluateBinaryClassification.ExecuteAll(
        yValidate, simpleModel(xValidateRaw))

runVisual = True
if runVisual:
    import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

    ############################
    # Set up the data

    (xRaw, yRaw) = SMSSpamDataset.LoadRawData()

    import MachineLearningCourse.MLUtilities.Data.Sample as Sample
    (xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw,
     yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)
    import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    trainLosses = []
    validationLosses = []
    numFeaturesSweep = [1, 10, 20, 30, 40, 50]
    for numFeatures in numFeaturesSweep:
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize(
            useHandCraftedFeatures=False)
        featurizer.CreateVocabulary(
            xTrainRaw, yTrain, numMutualInformationWords=numFeatures)

        xTrain = featurizer.Featurize(xTrainRaw)
        xValidate = featurizer.Featurize(xValidateRaw)
        xTest = featurizer.Featurize(xTestRaw)
        #################
        # You may find the following module helpful for making charts. You'll have to install matplotlib (see the lecture notes).
        #

        #
        # # trainLosses, validationLosses, and lossXLabels are parallel arrays with the losses you want to plot at the specified x coordinates
        #

        print("Learning the logistic regression model:")
        model = LogisticRegression.LogisticRegression()

        model.fit(
            xTrain, yTrain, stepSize=1.0, convergence=0.001)
        # then look at the models weights
        model.visualize()
        trainLosses.append(
            model.loss(xTrain, yTrain))
        validationLosses.append(
            model.loss(xValidate, yValidate))

    Charting.PlotSeries([trainLosses, validationLosses], ['Train', 'Validate'], numFeaturesSweep, chartTitle="Logistic Regression -- Number of MI Features",
                        xAxisTitle="Number of Features", yAxisTitle="Avg. Loss", outputDirectory=kOutputDirectory, fileName="MIModelLossAssignment4")
