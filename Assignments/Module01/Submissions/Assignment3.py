

from tabulate import tabulate
totalFeatures = 50
numMIs = [x * 5
          for x in range(int(totalFeatures/5) + 1)]
print(numMIs)
table = [[0.01, 8, 0.841], [0.001, 57, 0.867],
         [0.0001, 186, 0.923], [0.00001, 522, 0.927]]
headers = ["convergence", "steps", "accuracy"]

print(tabulate(table, headers, tablefmt="github"))

"""
1 Point.

Tune the hyperparameter ‘convergence’ by trying [ 0.01, 0.001, 0.0001, 0.00001 ] (with stepSize of 1.0). Produce a table showing:

<convergence parameter>, <steps to convergence>, <validation set accuracy>

for each setting of the convergence hyperparameter.
"""
# Submission:
"""
|   convergence |   steps |   accuracy |
|---------------|---------|------------|
|        0.01   |       8 |      0.841 |
|        0.001  |      57 |      0.867 |
|        0.0001 |     186 |      0.923 |
|        1e-05  |     522 |      0.927 |
"""


"""
1 Point –

Describe in 2-3 sentences your interpretation of the output of this hyperparameter sweep. Include a justification for trying more parameters
or stopping there. If you do think you should try more values of the 'convergence' hyperparameter, what is the next one you would try?
"""
