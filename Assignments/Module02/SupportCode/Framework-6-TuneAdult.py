# Really? You still need framework code?

# Sorry, there isn't any. Because I think you've got this!


import MachineLearningCourse.MLUtilities.Learners.DecisionTree as DecisionTree
import MachineLearningCourse.MLSolution.ParameterSweep as ParameterSweep
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.MLProjectSupport.Adult.AdultDataset as AdultDataset
import MachineLearningCourse.Assignments.Module02.SupportCode.AdultFeaturize as AdultFeaturize
from MachineLearningCourse.MLUtilities.Learners.DecisionTreeWeighted import DecisionTreeWeighted
from MachineLearningCourse.MLUtilities.Learners.ADABoost import AdaBoost
from MachineLearningCourse.MLUtilities.Learners.LogisticRegression import LogisticRegression


(xRaw, yRaw) = AdultDataset.LoadRawData()

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw,
 yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)


featurizerDefaults = {
    'useCategoricalFeatures': True,
    'useNumericFeatures': True
}

rounds = [25, 35, 40, 45]
modelParams = {
    "maxDepth": 5
}

# rounds = [{"maxDepth": 1}, {"maxDepth": 2}, {
#   "maxDepth": 3}, {"maxDepth": 5}, {"maxDepth": 7}, {"maxDepth": 9}]
modelDefaults = {
    'rounds': 50,
    'modelType': DecisionTreeWeighted,
    'modelParams': modelParams
}

ParameterSweep.hyperparameterSweep('rounds', xTrainRaw, yTrain, modelType=AdaBoost,
                                   featurizerType=AdultFeaturize.AdultFeaturize, featureCreateMethod='CreateFeatureSet', paramValues=rounds, modelDefaults=modelDefaults, featurizerDefaults=featurizerDefaults, outputName="max-depth-5-round-param-sweep-lower-vals")
