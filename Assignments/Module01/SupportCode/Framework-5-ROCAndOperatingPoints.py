from tabulate import tabulate
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset
kOutputDirectory = "./temp/f75"


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


def findFPRAndFNRForThreshold(targetThreshold, FPRs, FNRs, thresholds):
    index = thresholds.index(targetThreshold)
    FNR = FNRs[index]
    FPR = FPRs[index]
    threshold = thresholds[index]
    return (FPR, FNR, threshold)


def findThresholdAndFNRForFPR(targetFPR, FPRs, FNRs, thresholds):
    for i in range(len(FNRs)):
        FNR = FNRs[i]
        FPR = FPRs[i]
        threshold = thresholds[i]
        if (FPR <= targetFPR):
            return (FPR, FNR, threshold)


def findThresholdAndFPRForFNR(targetFNR, FPRs, FNRs, thresholds):
    for i in range(len(FNRs)):
        FNR = FNRs[i]
        FPR = FPRs[i]
        threshold = thresholds[i]
        if (FNR >= targetFNR):
            return (FPR, FNR, threshold)


# Hyperparameters to use for the run
stepSize = 0.1
convergence = 0.0001

# Set up to hold information for creating ROC curves
seriesFPRs = []
seriesFNRs = []
seriesLabels = []

# Learn a model with 25 frequent features
featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords=25)

xTrain = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest = featurizer.Featurize(xTestRaw)

model = LogisticRegression.LogisticRegression()
model.fit(xTrain, yTrain, convergence=convergence, stepSize=stepSize)

(modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(
    model, xValidate, yValidate)

print("Frequent FRPs", modelFPRs)
print('Frequent 0.5', findThresholdAndFNRForFPR(
    0.5, modelFPRs, modelFNRs, thresholds))

seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('25 Frequent')

# Learn a model with 25 features by mutual information
featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords=25)

xTrain = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest = featurizer.Featurize(xTestRaw)

model = LogisticRegression.LogisticRegression()
model.fit(xTrain, yTrain, convergence=convergence, stepSize=stepSize)

(modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(
    model, xValidate, yValidate)
seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('25 Mutual Information')

print('MI 0.1', findThresholdAndFNRForFPR(
    0.1, modelFPRs, modelFNRs, thresholds))

totalFeatures = 75
numMIs = [x * 5 + 10
          for x in range(int(totalFeatures/5)-4)]

# New Model
stepSize = 0.1
convergence = 0.0001

perf10 = []
perf50 = []

for numMI in numMIs:
    numFrequent = totalFeatures - numMI
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(
        useHandCraftedFeatures=False)
    featurizer.CreateVocabulary(
        xTrainRaw, yTrain, numMutualInformationWords=numMI, numFrequentWords=totalFeatures-numMI)

    xTrain = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest = featurizer.Featurize(xTestRaw)

    model = LogisticRegression.LogisticRegression()
    model.fit(xTrain, yTrain, convergence=convergence, stepSize=stepSize)

    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(
        model, xValidate, yValidate)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('Custom Model %s-%s' % (numMI, numFrequent))
    currentFNRs = [seriesFNRs[0], seriesFNRs[1]]
    currentFPRs = [seriesFPRs[0], seriesFPRs[1]]
    currentLabels = [seriesLabels[0], seriesLabels[1]]
    currentFPRs.append(modelFPRs)
    currentFNRs.append(modelFNRs)
    currentLabels.append('Custom Model %s-%s' % (numMI, numFrequent))

    p10 = [numMI]
    p10.extend(findThresholdAndFNRForFPR(
        0.1, modelFPRs, modelFNRs, thresholds))
    perf10.append(p10)

    p50 = [numMI]
    p50.extend(findThresholdAndFNRForFPR(
        0.5, modelFPRs, modelFNRs, thresholds))
    perf50.append(p50)

    print("Target threshold 0.11", findFPRAndFNRForThreshold(
        0.11, modelFPRs, modelFNRs, thresholds))
    print("Target threshold 0.17", findFPRAndFNRForThreshold(
        0.17, modelFPRs, modelFNRs, thresholds))

    Charting.PlotROCs(currentFPRs, currentFNRs, currentLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate",
                      yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-SMSSpamROCs-Part2-%s-%s" % (numMI, numFrequent))

Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate",
                  yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-SMSSpamROCs-Part2-All-%s" % totalFeatures)

headers = ["numMI", "FPR", "FNR", "threshold"]
print(tabulate(perf10, headers, tablefmt="github"))
print(tabulate(perf50, headers, tablefmt="github"))

"""

Frequent 0.5 (0.3739130434782609, 0.04597701149425287, 0.11)
MI 0.1 (0.09782608695652174, 0.14942528735632185, 0.17)
New 0.5 (0.3695652173913043, 0.08045977011494253, 0.04)
New 0.1 (0.1, 0.14942528735632185, 0.15)

total: 50 features
p10 values
|   numMI |       FPR |      FNR |   threshold |
|---------|-----------|----------|-------------|
|       0 | 0.0891304 | 0.264368 |        0.22 |
|       5 | 0.1       | 0.183908 |        0.21 |
|      10 | 0.0956522 | 0.183908 |        0.21 |
|      15 | 0.1       | 0.172414 |        0.21 |
|      20 | 0.076087  | 0.195402 |        0.22 |
|      25 | 0.0804348 | 0.16092  |        0.22 |
|      30 | 0.1       | 0.172414 |        0.21 |
|      35 | 0.0652174 | 0.16092  |        0.22 |
|      40 | 0.0652174 | 0.16092  |        0.21 |
|      45 | 0.0869565 | 0.172414 |        0.2  |
|      50 | 0.1       | 0.137931 |        0.18 |
p50 values
|   numMI |      FPR |       FNR |   threshold |
|---------|----------|-----------|-------------|
|       0 | 0.497826 | 0.0344828 |        0.1  |
|       5 | 0.493478 | 0.0344828 |        0.09 |
|      10 | 0.386957 | 0.045977  |        0.09 |
|      15 | 0.48913  | 0.0344828 |        0.09 |
|      20 | 0.373913 | 0.045977  |        0.09 |
|      25 | 0.356522 | 0.045977  |        0.09 |
|      30 | 0.35     | 0.045977  |        0.09 |
|      35 | 0.315217 | 0.0574713 |        0.1  |
|      40 | 0.328261 | 0.0574713 |        0.09 |
|      45 | 0.341304 | 0.0574713 |        0.09 |
|      50 | 0.386957 | 0.0344828 |        0.08 |

Total Features 25

|   numMI |       FPR |      FNR |   threshold |
|---------|-----------|----------|-------------|
|       0 | 0.1       | 0.333333 |        0.22 |
|       5 | 0.0913043 | 0.310345 |        0.24 |
|      10 | 0.0913043 | 0.275862 |        0.23 |
|      15 | 0.0978261 | 0.310345 |        0.24 |
|      20 | 0.0673913 | 0.310345 |        0.22 |
|      25 | 0.0978261 | 0.149425 |        0.17 |
|   numMI |      FPR |       FNR |   threshold |
|---------|----------|-----------|-------------|
|       0 | 0.373913 | 0.045977  |        0.11 |
|       5 | 0.376087 | 0.045977  |        0.1  |
|      10 | 0.347826 | 0.091954  |        0.1  |
|      15 | 0.319565 | 0.114943  |        0.1  |
|      20 | 0.334783 | 0.091954  |        0.1  |
|      25 | 0.36087  | 0.0804598 |        0.09 |


75
|   numMI |       FPR |      FNR |   threshold |
|---------|-----------|----------|-------------|
|      10 | 0.0978261 | 0.137931 |        0.2  |
|      15 | 0.1       | 0.16092  |        0.21 |
|      20 | 0.0956522 | 0.16092  |        0.21 |
|      25 | 0.0978261 | 0.126437 |        0.2  |
|      30 | 0.0956522 | 0.126437 |        0.2  |
|      35 | 0.1       | 0.091954 |        0.19 |
|      40 | 0.0934783 | 0.114943 |        0.2  |
|      45 | 0.076087  | 0.137931 |        0.21 |
|      50 | 0.0847826 | 0.137931 |        0.2  |
|      55 | 0.073913  | 0.16092  |        0.21 |
|      60 | 0.0978261 | 0.149425 |        0.2  |
|   numMI |      FPR |       FNR |   threshold |
|---------|----------|-----------|-------------|
|      10 | 0.386957 | 0.045977  |        0.09 |
|      15 | 0.382609 | 0.045977  |        0.09 |
|      20 | 0.376087 | 0.045977  |        0.09 |
|      25 | 0.363043 | 0.045977  |        0.09 |
|      30 | 0.363043 | 0.045977  |        0.09 |
|      35 | 0.491304 | 0.0344828 |        0.08 |
|      40 | 0.358696 | 0.045977  |        0.09 |
|      45 | 0.343478 | 0.045977  |        0.09 |
|      50 | 0.334783 | 0.045977  |        0.09 |
|      55 | 0.336957 | 0.045977  |        0.09 |
|      60 | 0.33913  | 0.045977  |        0.09 |
"""
