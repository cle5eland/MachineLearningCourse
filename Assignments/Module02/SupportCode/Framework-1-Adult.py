import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.Assignments.Module02.SupportCode.AdultFeaturize as AdultFeaturize
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.MLProjectSupport.Adult.AdultDataset as AdultDataset
import MachineLearningCourse.MLSolution.ParameterSweep as ParameterSweep
kOutputDirectory = "./temp/mod2/assignment4"


(xRaw, yRaw) = AdultDataset.LoadRawData()


(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw,
 yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent >50K." %
      (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent >50K." %
      (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent >50K." %
      (len(yTest), 100.0 * sum(yTest)/len(yTest)))

modelDefaults = {
    'convergence': 0.0001,
    'stepSize': 1
}

featurizerDefaults = {
    'useCategoricalFeatures': True,
    'useNumericFeatures': False
}


# ParameterSweep.hyperparameterSweep('convergence', xTrainRaw, yTrain, modelType=LogisticRegression.LogisticRegression,
#                                   featurizerType=AdultFeaturize.AdultFeaturize, featureCreateMethod='CreateFeatureSet', paramValues=[0.05, 0.01, 0.001, 0.0005, 0.0001], modelDefaults=modelDefaults, featurizerDefaults=featurizerDefaults)


# ParameterSweep.hyperparameterSweep('stepSize', xTrainRaw, yTrain, modelType=LogisticRegression.LogisticRegression,
#                                   featurizerType=AdultFeaturize.AdultFeaturize, featureCreateMethod='CreateFeatureSet', paramValues=[10.0, 8.0, 5.0, 1.0, 0.5, 0.1], modelDefaults=modelDefaults, featurizerDefaults=featurizerDefaults)


featurizer = AdultFeaturize.AdultFeaturize()
featurizer.CreateFeatureSet(
    xTrainRaw, yTrain, useCategoricalFeatures=True, useNumericFeatures=False)
for i in range(featurizer.GetFeatureCount()):
    print("%d - %s" % (i, featurizer.GetFeatureInfo(i)))

xTrain = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest = featurizer.Featurize(xTestRaw)

for i in range(10):
    print("%d - " % (yTrain[i]), xTrain[i])

############################

model = MostCommonClassModel.MostCommonClassModel()
model.fit(xTrain, yTrain)
yValidatePredicted = model.predict(xValidate)
validateAccuracy = EvaluateBinaryClassification.Accuracy(
    yValidate, yValidatePredicted)
errorBounds = ErrorBounds.GetAccuracyBounds(
    validateAccuracy, len(yValidate), 0.5)

print()
print("### 'Most Common Class' model validate set accuracy: %.4f (50%% %.4f - %.4f)" %
      (validateAccuracy, errorBounds[0], errorBounds[1]))
