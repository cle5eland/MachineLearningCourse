from joblib import Parallel, delayed
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import time
from tabulate import tabulate
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset
kOutputDirectory = "./temp/assignment7"


(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw,
 yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)


# A helper function for calculating FN rate and FP rate across a range of thresholds

def TabulateModelPerformanceForROC(model, xValidate, yValidate):
    pointsToEvaluate = 100
    thresholds = [x / float(pointsToEvaluate)
                  for x in range(pointsToEvaluate + 1)]
    FPRs = []
    FNRs = []

    try:
        for threshold in thresholds:
            FPRs.append(EvaluateBinaryClassification.FalsePositiveRate(
                yValidate, model.predict(xValidate, classificationThreshold=threshold)))
            FNRs.append(EvaluateBinaryClassification.FalseNegativeRate(
                yValidate, model.predict(xValidate, classificationThreshold=threshold)))
    except NotImplementedError:
        raise UserWarning(
            "The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

    return (FPRs, FNRs, thresholds)


# This function will help you plot with error bars. Use it just like PlotSeries, but with parallel arrays of error bar sizes in the second variable
#     note that the error bar size is drawn above and below the series value. So if the series value is .8 and the confidence interval is .78 - .82, then the value to use for the error bar is .02

# Charting.PlotSeriesWithErrorBars([series1, series2], [errorBarsForSeries1, errorBarsForSeries2], ["Series1", "Series2"], xValues, chartTitle="<>", xAxisTitle="<>", yAxisTitle="<>", yBotLimit=0.8, outputDirectory=kOutputDirectory, fileName="<name>")


# This helper function should execute a single run and save the results on 'runSpecification' (which could be a dictionary for convienience)
#    for later tabulation and charting...
def ExecuteEvaluationRun(runSpecification, xTrainRaw, yTrain, numberOfFolds=5):
    startTime = time.time()

    # HERE upgrade this to use crossvalidation
    numFrequentWords = runSpecification['numFrequentWords']
    numMutualInformationWords = runSpecification['numMutualInformationWords']
    convergence = runSpecification['convergence']
    stepSize = runSpecification['stepSize']
    accuracy = CrossValidation.Execute(2, xTrainRaw, yTrain, numMutualInformationWords=numMutualInformationWords,
                                       numFrequentWords=numFrequentWords, convergence=convergence, stepSize=stepSize)

    runSpecification['accuracy'] = accuracy

    (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(
        accuracy, len(xTrainRaw), 0.5)
    runSpecification['lowerBound'] = lowerBound
    runSpecification['upperBound'] = upperBound

    endTime = time.time()
    runSpecification['runtime'] = endTime - startTime

    return runSpecification


# optimizing, numMutual, stepSize, convergence, numFrequentWords
best = [
    ["convergence", 20, 1, 0.0001, 0],
    ["stepSize", 20, 5, 0.0001, 0],
    ["numFrequentWords", 20, 5, 0.0001, 100],
    ["numMutualInformationWords", 75, 5, 0.0001, 100],
    ["convergence", 75, 5, 0.00001, 100],
    ["stepSize", 75, 8, 0.00001, 100]
    ["numFrequentWords", 75, 8, 0.00001, 100]
]


def makeRunSpec(input):
    runSpecification = {}
    runSpecification['optimizing'] = input[0]
    runSpecification['numMutualInformationWords'] = input[1]
    runSpecification['stepSize'] = input[2]
    runSpecification['convergence'] = input[3]
    runSpecification['numFrequentWords'] = input[4]


evaluationRunSpecifications = list(map(makeRunSpec, best))

"""
evaluationRunSpecifications = []
paramValues = [125, 150, 200, 250]
for numMutualInformationWords in paramValues:

    runSpecification = {}
    runSpecification['optimizing'] = 'numMutualInformationWords'
    runSpecification['numMutualInformationWords'] = numMutualInformationWords
    runSpecification['stepSize'] = 8.0
    runSpecification['convergence'] = 0.00001
    runSpecification['numFrequentWords'] = 100

    evaluationRunSpecifications.append(runSpecification)
evaluationRunSpecifications = []
paramValues = [125, 150, 200, 250]
for numFrequentWords in paramValues:

    runSpecification = {}
    runSpecification['optimizing'] = 'numFrequentWords'
    runSpecification['numMutualInformationWords'] = 75
    runSpecification['stepSize'] = 8.0
    runSpecification['convergence'] = 0.00001
    runSpecification['numFrequentWords'] = numFrequentWords

    evaluationRunSpecifications.append(runSpecification)


# Convergence = 0.0001
evaluationRunSpecifications = []
paramValues = [0.00005, 0.00001, 0.000001]  # 0.0000001]
for convergence in paramValues:

    runSpecification = {}
    runSpecification['optimizing'] = 'convergence'
    runSpecification['numMutualInformationWords'] = 75
    runSpecification['stepSize'] = 5.0
    runSpecification['convergence'] = convergence
    runSpecification['numFrequentWords'] = 100

    evaluationRunSpecifications.append(runSpecification)


evaluationRunSpecifications = []
paramValues = [1.0, 5.0, 8.0, 10.0, 15.0]
# paramValues = [1]
# Step Size = 5
for stepSize in paramValues:

    runSpecification = {}
    runSpecification['optimizing'] = 'stepSize'
    runSpecification['numMutualInformationWords'] = 75
    runSpecification['stepSize'] = stepSize
    runSpecification['convergence'] = 0.00001
    runSpecification['numFrequentWords'] = 100

    evaluationRunSpecifications.append(runSpecification)
"""

# if you want to run in parallel you need to install joblib as described in the lecture notes and adjust the comments on the next three lines...
evaluations = Parallel(n_jobs=12)(delayed(ExecuteEvaluationRun)(
    runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications)

optimizing = evaluations[0]["optimizing"]
print(evaluations[0])
headers = [optimizing, "accuracy", "lower bound", "upper bound", "runtime"]
table = list(map(lambda evaluation: [evaluation[optimizing], evaluation["accuracy"],
                                     evaluation["lowerBound"], evaluation["upperBound"], evaluation["runtime"]], evaluations))
print(tabulate(table, headers, tablefmt="github"))

""" def plotAccuracy(evaluation):
    series = [0 in range(len(paramValues))]
    index = series.index(evaluation[optimizing])
    series[index]"""
# evaluations = [ExecuteEvaluationRun(
#     runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications]
series = list(map(lambda evaluation: evaluation["accuracy"], evaluations))
errorBounds = list(map(lambda evaluation:
                       evaluation["upperBound"] - evaluation['accuracy'], evaluations))
Charting.PlotSeriesWithErrorBars([series], [errorBounds], ["Accuracies with 50% Double Error Bounds"], paramValues,
                                 chartTitle="Optimizing %s - Round 2" % optimizing, xAxisTitle=optimizing, yAxisTitle="Accuracy", outputDirectory=kOutputDirectory, yBotLimit=0.9, yTopLimit=1.0, fileName="%s-param-optimization-2" % optimizing)

for evaluation in evaluations:
    print(evaluation)

# Good luck!
