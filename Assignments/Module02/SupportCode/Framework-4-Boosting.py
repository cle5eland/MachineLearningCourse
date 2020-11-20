import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCompound2D as ConceptCompound2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptLinear2D as ConceptLinear2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptSquare2D as ConceptSquare2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCircle2D as ConceptCircle2D
import MachineLearningCourse.MLUtilities.Data.Generators.SampleUniform2D as SampleUniform2D
from MachineLearningCourse.MLUtilities.Learners.ADABoost import AdaBoost
from MachineLearningCourse.MLUtilities.Learners.DecisionTreeWeighted import DecisionTreeWeighted
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLSolution.ParameterSweep as ParameterSweep
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
import time
from joblib import Parallel, delayed


kOutputDirectory = "./temp/mod2/assignment3"


# remember this helper function
# Charting.PlotSeriesWithErrorBars([yValues], [errorBars], [series names], xValues, chartTitle=", xAxisTitle="", yAxisTitle="", yBotLimit=0.5, outputDirectory=kOutputDirectory, fileName="")

# generat some synthetic data do help debug your learning code

generator = SampleUniform2D.SampleUniform2D(seed=100)
#conceptSquare = ConceptSquare2D.ConceptSquare2D(width=.2)
conceptLinear = ConceptLinear2D.ConceptLinear2D(bias=0.05, weights=[0.3, -0.3])
conceptCircle = ConceptCircle2D.ConceptCircle2D(radius=.3)

concept = ConceptCompound2D.ConceptCompound2D(
    concepts=[conceptLinear, conceptCircle])

xTest = generator.generate(1000)
yTest = concept.predict(xTest)

xTrain = generator.generate(1000)
yTrain = concept.predict(xTrain)

RunVisualize = True
if RunVisualize:
    # this code outputs the true concept.
    visualize = Visualize2D.Visualize2D(kOutputDirectory, "generated-concept")
    visualize.Plot2DDataAndBinaryConcept(xTest, yTest, concept)
    visualize.Save()

    # you can use this to visualize what your model is learning.
    visualize = Visualize2D.Visualize2D(
        kOutputDirectory, "boosted-tree", size=400)
    modelParams = {
        "maxDepth": 1
    }
    model = AdaBoost()
    model.fit(x=xTrain, y=yTrain, rounds=50,
              modelType=DecisionTreeWeighted, modelParams=modelParams)
    model.visualize()
    visualize.PlotBinaryConcept(model)
    visualize.Save()


def NewExecute(modelType, modelParams: dict,):
    totalCorrect = 0
    # Fit model
    model = modelType()
    model.fit(xTrain, yTrain, **modelParams)
    model.visualize()
    # Count accurate predictions
    totalCorrect += CrossValidation.__countCorrect(yTest,
                                                   model.predict(xTest))

    # Calculate total accuracy
    logAccuracy = totalCorrect/len(xTrain)
    return logAccuracy


def ExecuteEvaluationRun(runSpecification):
    startTime = time.time()
    modelSpecification = runSpecification['modelSpecification']
    paramSpecification = runSpecification['parameterSpecification']
    # just need raw accuracy here
    accuracy = NewExecute(**modelSpecification, **paramSpecification)

    runSpecification['accuracy'] = accuracy

    (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(
        accuracy, len(xTrain), 0.5)
    runSpecification['lowerBound'] = lowerBound
    runSpecification['upperBound'] = upperBound

    endTime = time.time()
    runSpecification['runtime'] = endTime - startTime

    return runSpecification


def hyperparameterSweep(parameterName: str, modelType, paramValues: list, modelDefaults: dict):
    evaluationRunSpecifications = []
    # paramValues = [1]
    # Step Size = 5
    modelSpecification = {}
    modelSpecification['modelType'] = modelType
    for param in paramValues:
        runSpecification = {}
        modelParams = modelDefaults.copy()
        modelParams[parameterName] = param

        parameterSpecification = {
            'modelParams': modelParams,
        }

        runSpecification['modelSpecification'] = modelSpecification
        runSpecification['parameterSpecification'] = parameterSpecification
        runSpecification['optimizing'] = parameterName
        runSpecification[parameterName] = param

        evaluationRunSpecifications.append(runSpecification)

    evaluations = Parallel(n_jobs=12)(delayed(ExecuteEvaluationRun)(
        runSpec) for runSpec in evaluationRunSpecifications)

    ParameterSweep.outputResult(parameterName, paramValues, evaluations)


RunParameterSweep = False
if RunParameterSweep:
    rounds = [1, 10, 20, 50, 100, 200, 500, 1000, 2000]
    modelParams = {
        "maxDepth": 1
    }
    defaults = {
        'rounds': 1,
        'modelType': DecisionTreeWeighted,
        'modelParams': modelParams
    }
    hyperparameterSweep('rounds', AdaBoost, rounds, defaults)
# Or you can use it to visualize individual models that you learened, e.g.:
# visualize.PlotBinaryConcept(model->modelLearnedInRound[2])

# you might like to see the training or test data too, so you might prefer this to simply calling 'PlotBinaryConcept'
# visualize.Plot2DDataAndBinaryConcept(xTrain,yTrain,model)

# And remember to save
# visualize.Save()
