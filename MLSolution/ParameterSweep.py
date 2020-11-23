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


kOutputDirectory = "./temp/mod2/assignment6"


def ExecuteEvaluationRun(runSpecification, xTrainRaw, yTrain, numberOfFolds=5):
    startTime = time.time()

    modelSpecification = runSpecification['modelSpecification']
    paramSpecification = runSpecification['parameterSpecification']
    accuracy = CrossValidation.NewExecute(
        numberOfFolds, xTrainRaw, yTrain, **modelSpecification, **paramSpecification)

    runSpecification['accuracy'] = accuracy

    (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(
        accuracy, len(xTrainRaw), 0.5)
    runSpecification['lowerBound'] = lowerBound
    runSpecification['upperBound'] = upperBound

    endTime = time.time()
    runSpecification['runtime'] = endTime - startTime

    return runSpecification


def outputTable(optimizing: str, evaluations: []):
    headers = [optimizing, "accuracy", "lower bound", "upper bound", "runtime"]
    table = list(map(lambda evaluation: [evaluation[optimizing], evaluation["accuracy"],
                                         evaluation["lowerBound"], evaluation["upperBound"], evaluation["runtime"]], evaluations))
    print(tabulate(table, headers, tablefmt="github"))


def outputPlot(optimizing: str, paramValues: [], evaluations: [], outputName=None):
    series = list(map(lambda evaluation: evaluation["accuracy"], evaluations))
    errorBounds = list(map(lambda evaluation:
                           evaluation["upperBound"] - evaluation['accuracy'], evaluations))
    yBotLimit = min(series) - 0.01
    yBotLimit = yBotLimit if yBotLimit > 0 else 0
    yTopLimit = max(series) + 0.01
    yTopLimit = yTopLimit if yTopLimit < 1 else 1
    Charting.PlotSeriesWithErrorBars([series], [errorBounds], ["Accuracies with 50% Double Error Bounds"], paramValues,
                                     chartTitle="Optimizing %s" % optimizing, xAxisTitle=optimizing, yAxisTitle="Accuracy", yBotLimit=yBotLimit, yTopLimit=yTopLimit, outputDirectory=kOutputDirectory, fileName=outputName if outputName != None else "%s-param-optimization" % optimizing)


def outputResult(optimizing: str, paramValues: [], evaluations: [], outputName=None):
    outputTable(optimizing, evaluations)
    outputPlot(optimizing, paramValues, evaluations, outputName)


def hyperparameterSweep(parameterName: str, xTrainRaw: list, yTrain: list, modelType, featurizerType, featureCreateMethod: str, paramValues: list, modelDefaults: dict, featurizerDefaults: dict, modelParam: bool = True, xValidateRaw=None, yValidate=None, outputName=None):
    evaluationRunSpecifications = []
    # paramValues = [1]
    # Step Size = 5
    modelSpecification = {}
    modelSpecification['modelType'] = modelType
    modelSpecification['featurizerType'] = featurizerType
    modelSpecification['featureCreateMethod'] = featureCreateMethod
    for param in paramValues:
        runSpecification = {}
        modelParams = modelDefaults.copy()
        featurizerParams = featurizerDefaults.copy()
        if modelParam:
            modelParams[parameterName] = param
        else:
            featurizerParams[parameterName] = param

        parameterSpecification = {
            'modelParams': modelParams,
            'featurizerParams': featurizerParams
        }

        runSpecification['modelSpecification'] = modelSpecification
        runSpecification['parameterSpecification'] = parameterSpecification
        runSpecification['optimizing'] = parameterName
        runSpecification[parameterName] = param

        evaluationRunSpecifications.append(runSpecification)

    evaluations = Parallel(n_jobs=12)(delayed(ExecuteEvaluationRun)(
        runSpec, xTrainRaw, yTrain, 2) for runSpec in evaluationRunSpecifications)

    outputResult(parameterName, paramValues,
                 evaluations, outputName=outputName)
