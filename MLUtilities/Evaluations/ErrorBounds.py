import math

_zValues = {.5: .67, .68: 1.0, .8: 1.28, .9: 1.64, .95: 1.96, .98: 2.33, .99: 2.58}


def GetAccuracyBounds(mean: float, sampleSize: int, confidence: float):
    if mean < 0.0 or mean > 1.0:
        raise UserWarning("mean must be between 0 and 1")

    if sampleSize <= 0:
        raise UserWarning("sampleSize should be positive")

    # TODO: check if confidence in _zValues
    z = _zValues.get(confidence)
    val = math.sqrt((mean * (1.0-mean))/float(sampleSize)) * z
    lower = mean - val
    upper = mean + val

    return (lower, upper)
