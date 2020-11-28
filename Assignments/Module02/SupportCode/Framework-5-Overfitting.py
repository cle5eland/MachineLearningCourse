
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
"""

modelDefaults = {
    'maxDepth': 5,
    'weights': [1.0 for _ in range(len(yTrain))]}


paramValues = [1, 5, 8, 10, 12, 15, 18]


ParameterSweep.hyperparameterSweep('maxDepth', xTrainRaw, yTrain, modelType=DecisionTreeWeighted,
                                   featurizerType=AdultFeaturize.AdultFeaturize, featureCreateMethod='CreateFeatureSet', paramValues=paramValues, modelDefaults=modelDefaults, featurizerDefaults=featurizerDefaults, xValidateRaw=xValidateRaw, yValidate=yValidate, outputName="weighted-tree-step-size")


modelDefaults = {
    'maxDepth': 5
}


ParameterSweep.hyperparameterSweep('maxDepth', xTrainRaw, yTrain, modelType=DecisionTree.DecisionTree,
                                   featurizerType=AdultFeaturize.AdultFeaturize, featureCreateMethod='CreateFeatureSet', paramValues=paramValues, modelDefaults=modelDefaults, featurizerDefaults=featurizerDefaults, xValidateRaw=xValidateRaw, yValidate=yValidate, outputName="normal-tree-step-size")

"""
modelDefaults = {
    "stepSize": 1.0,
    "convergence": 0.0001,
}

rounds = [0.01, 0.1, 1.0, 3.0, 8.0, 12.0]

ParameterSweep.hyperparameterSweep('stepSize', xTrainRaw, yTrain, modelType=LogisticRegression,
                                   featurizerType=AdultFeaturize.AdultFeaturize, featureCreateMethod='CreateFeatureSet', paramValues=rounds, modelDefaults=modelDefaults, featurizerDefaults=featurizerDefaults, xValidateRaw=xValidateRaw, yValidate=yValidate)

"""

rounds = [5, 10, 20, 50, 100, 200]
modelParams = {
    "maxDepth": 1
}
modelDefaults = {
    'rounds': 1,
    'modelType': DecisionTreeWeighted,
    'modelParams': modelParams
}

ParameterSweep.hyperparameterSweep('rounds', xTrainRaw, yTrain, modelType=AdaBoost,
                                   featurizerType=AdultFeaturize.AdultFeaturize, featureCreateMethod='CreateFeatureSet', paramValues=rounds, modelDefaults=modelDefaults, featurizerDefaults=featurizerDefaults, xValidateRaw=xValidateRaw, yValidate=yValidate)
"""
