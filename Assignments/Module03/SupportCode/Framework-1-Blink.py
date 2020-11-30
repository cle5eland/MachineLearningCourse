import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkFeaturize as BlinkFeaturize
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
from MachineLearningCourse.MLUtilities.Learners.ADABoost import AdaBoost
import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset
from MachineLearningCourse.MLUtilities.Learners.DecisionTreeWeighted import DecisionTreeWeighted
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
import MachineLearningCourse.MLSolution.ParameterSweep as ParameterSweep


kOutputDirectory = "./temp/mod3/assignment1"

(xRaw, yRaw) = BlinkDataset.LoadRawData()


(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw,
 yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent opened." %
      (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent opened." %
      (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent opened" %
      (len(yTest), 100.0 * sum(yTest)/len(yTest)))


model = AdaBoost()

featurizerDefaults = {
    'includeEdgeFeatures': True,
    'includeSubdividedFeatures': True
}


modelDefaults = {
    'rounds': 20,
    'modelParams': {
        "maxDepth": 5
    },
    'modelType': DecisionTreeWeighted
}

paramValues = [20, 30, 40, 50, 70, 100]
ParameterSweep.hyperparameterSweep('rounds', xTrainRaw, yTrain, modelType=AdaBoost, featurizerType=BlinkFeaturize.BlinkFeaturize,
                                   featureCreateMethod='CreateFeatureSet', paramValues=paramValues, modelDefaults=modelDefaults, featurizerDefaults=featurizerDefaults, xValidateRaw=xValidateRaw, yValidate=yValidate, outputName='rerun-rounds-sweep-assignment-1-depth5-actually-all-features')
