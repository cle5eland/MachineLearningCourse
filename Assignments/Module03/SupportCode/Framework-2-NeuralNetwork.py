from PIL import Image
import PIL
from joblib import Parallel, delayed
import time
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
from MachineLearningCourse.MLUtilities.Learners.NeuralNetworkFullyConnected import NeuralNetworkFullyConnected
import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkFeaturize as BlinkFeaturize
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
import MachineLearningCourse.MLSolution.ParameterSweep as ParameterSweep
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

kOutputDirectory = "./temp/mod3/assignment2"


(xRaw, yRaw) = BlinkDataset.LoadRawData()


(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw,
 yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent opened." %
      (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent opened." %
      (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent opened" %
      (len(yTest), 100.0 * sum(yTest)/len(yTest)))


featurizer = BlinkFeaturize.BlinkFeaturize()


sampleStride = 2
featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=False,
                            includeIntensities=True, intensitiesSampleStride=sampleStride)

xTrain = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest = featurizer.Featurize(xTestRaw)


def VisualizeWeights(weightArray, outputPath, sampleStride=2):
    imageDimension = int(24 / sampleStride)
    pixelSize = 2 * sampleStride
    imageSize = imageDimension * pixelSize

    # note the extra weight for the bias is where the +1 comes from
    if len(weightArray) != (imageDimension * imageDimension) + 1:
        raise UserWarning("size of the weight array is %d but it should be %d" % (
            len(weightArray), (imageDimension * imageDimension) + 1))

    if not outputPath.endswith(".jpg"):
        raise UserWarning(
            "output path should be a path to a file that ends in .jpg, it is currently: %s" % (outputPath))

    image = Image.new("RGB", (imageSize, imageSize), "White")

    pixels = image.load()

    for x in range(imageDimension):
        for y in range(imageDimension):
            weight = weightArray[1+(x*imageDimension) + y]

            # Add in the bias to help understand the weight's function
            weight += weightArray[0]

            if weight >= 0:
                color = (0, int(255 * abs(weight)), 0)
            else:
                color = (int(255 * abs(weight)), 0, 0)

            for i in range(pixelSize):
                for j in range(pixelSize):
                    pixels[(x * pixelSize) + i, (y * pixelSize) + j] = color

    image.save(outputPath)


"""
weights = model.layers[1].getWeights()
for filterNumber in range(hiddenStructure[0]):
    # update the first parameter based on your representation
    VisualizeWeights(weights[filterNumber], "%s/filters/epoch%d_neuron%d.jpg" % (
        kOutputDirectory, 0, filterNumber), sampleStride=sampleStride)
"""


def outputPlot(epochs: [], valAcc: [], trainAcc: [], outputName):
    yBotLimit = min(min(valAcc), min(trainAcc)) - 0.01
    yBotLimit = yBotLimit if yBotLimit > 0 else 0
    yTopLimit = max(max(valAcc), max(trainAcc)) + 0.01
    yTopLimit = yTopLimit if yTopLimit < 1 else 1
    Charting.PlotSeries([valAcc, trainAcc], ["Validation Accuracy", "Training Set Accuracy"], epochs,
                        chartTitle="Single Hidden Layer Width 10 Validation and Training Accuracies", xAxisTitle="Epoch", yAxisTitle="Accuracy", yBotLimit=yBotLimit, yTopLimit=yTopLimit, outputDirectory=kOutputDirectory, fileName=outputName)


maxEpochs = 50000
step = 0.05
convergence = 0.0001

hiddenStructure = [5, 5]
print('initing model...')
model = NeuralNetworkFullyConnected(
    numInputFeatures=len(xTrain[0]), hiddenLayersNodeCounts=hiddenStructure)
print('Done.')
numEpoch = []
trainAcc = []
valAcc = []
epoch = None
for i in range(maxEpochs):
    if not model.converged:
        epoch = i
        print('epoch ', i)
        model.incrementalFit(xTrain, yTrain, epochs=1,
                             stepSize=step, convergence=convergence)
        numEpoch.append(i)
        yPredicted = model.predict(xValidate)
        correct = CrossValidation.__countCorrect(yValidate, yPredicted)
        accuracy = correct / float(len(yValidate))
        valAcc.append(accuracy)
        yPredicted = model.predict(xTrain)
        correct = CrossValidation.__countCorrect(yTrain, yPredicted)
        accuracy = correct / float(len(yTrain))
        trainAcc.append(accuracy)

print('model fit.')

"""
weights = model.layers[1].getWeights()

for filterNumber in range(hiddenStructure[0]):
    # update the first parameter based on your representation
    VisualizeWeights(weights[filterNumber], "%s/filters/single-layer-epoch%d_neuron%d.jpg" % (
        kOutputDirectory, epoch, filterNumber), sampleStride=sampleStride)"""
outputPlot(numEpoch, valAcc, trainAcc, 'Double-Layer-Acc-Plot')

"""
featurizerDefaults = {
    'includeEdgeFeatures': False,
    'includeIntensities': True,
    'intensitiesSampleStride': sampleStride
}


modelDefaults = {
    'maxEpochs': 50000,
    'stepSize': 0.1,
    'convergence': 0.0001
}

hiddenStructure = [5, 5]

modelInitParams = {
    'numInputFeatures': len(xTrain[0]), 'hiddenLayersNodeCounts': hiddenStructure
}

paramValues = [0.1, 0.01, 0.05]
ParameterSweep.hyperparameterSweep('stepSize', xTrainRaw, yTrain, modelType=NeuralNetworkFullyConnected, modelInitParams=modelInitParams, featurizerType=BlinkFeaturize.BlinkFeaturize,
                                   featureCreateMethod='CreateFeatureSet', paramValues=paramValues, modelDefaults=modelDefaults, featurizerDefaults=featurizerDefaults, xValidateRaw=xValidateRaw, yValidate=yValidate, outputName='mid-5-5-size-double-layer')
"""
