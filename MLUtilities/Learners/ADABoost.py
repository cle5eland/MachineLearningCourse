from math import pow, log
from collections import Counter
from joblib import Parallel, delayed


def calculateEpsilon(y, yPredicted, p):
    total = 0.0
    for i in range(len(y)):
        if y[i] != yPredicted[i]:
            total += p[i]
    return total


def calculateNewWeights(y, yPredicted, w, beta):
    return [w[i]*pow(beta, 1-(1 if y[i] != yPredicted[i] else 0))
            for i in range(len(w))]


class AdaBoost(object):

    def __init__(self):
        self.models = []
        self.betas = []
        self.labelDistribution = Counter()

    def updateLabelDistribution(self, y):
        for label in y:
            self.labelDistribution[label] += 1

    def fit(self, x: [[float]], y: [int], rounds: int, modelType, modelParams: dict):
        self.updateLabelDistribution(y)
        w = [1.0/float(len(x)) for _ in x]
        for _ in range(rounds):
            weightTotal = sum(w)
            p = [w[i]/weightTotal for i in range(len(w))]
            model = modelType()
            model.fit(x, y, weights=p, **modelParams, verbose=False)
            yPredicted = model.predict(x)
            epsilon = calculateEpsilon(y, yPredicted, p)
            if epsilon > 0.5:
                print('Epsilon greater than 0.5 -- exiting ADABoost.')
                # exit
                return
            if (epsilon == 0):
                print(
                    'Epsilon is 0 -- no error. Exiting for now, in future maybe add prior')
                return
            beta = (epsilon)/(1.0-epsilon)
            self.betas.append(beta)
            self.models.append(model)
            w = calculateNewWeights(y, yPredicted, w, beta)
            """
            print("\n\n")
            print("round: ", rnd + 1)
            print("")
            print("weights: ", p)
            print("")
            print('epsilon: ', epsilon)
            print('beta: ', beta)
            print('\n\n')"""

    def visualize(self):
        for i in range(len(self.models)):
            print('model ', i)
            self.models[i].visualize()

    def predict(self, x: [[]], classificationThreshold=0.5):
        print('predicting %s classifications...' % len(x))
        # predictions: [[int]] = [model.predict(x) for model in self.models]
        predictions = Parallel(n_jobs=12)(delayed(lambda model: model.predict(x))(
            model) for model in self.models)
        print('predictions complete.')
        aggregatePrediction = [0.0 for _ in range(len(x))]
        for i in range(len(x)):
            # init results
            results = {}
            for (label, _) in self.labelDistribution.items():
                results[label] = 0.0
            for k in range(len(predictions)):
                # prediction for current model
                prediction = predictions[k]
                beta = self.betas[k]
                # this is really only 0 and 1
                for (label, _) in self.labelDistribution.items():
                    if prediction[i] == label:
                        results[label] += log(1/beta)

            aggregatePrediction[i] = max(results, key=results.get)

        return aggregatePrediction
