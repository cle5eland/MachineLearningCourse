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

rounds = [5, 10, 20, 50, 100, 200, 1000, 2000]
modelParams = {
    "maxDepth": 2
}

# rounds = [{"maxDepth": 1}, {"maxDepth": 3}, {
#    "maxDepth": 5}, {"maxDepth": 8}, {"maxDepth": 10}, {"maxDepth": 15}]
modelDefaults = {
    'rounds': 10,
    'modelType': DecisionTreeWeighted,
    'modelParams': modelParams
}

ParameterSweep.hyperparameterSweep('rounds', xTrainRaw, yTrain, modelType=AdaBoost,
                                   featurizerType=AdultFeaturize.AdultFeaturize, featureCreateMethod='CreateFeatureSet', paramValues=rounds, modelDefaults=modelDefaults, featurizerDefaults=featurizerDefaults, outputName="adaboost-rounds-normalized-numerics")
