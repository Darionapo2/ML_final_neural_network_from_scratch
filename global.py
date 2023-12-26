from math import pow, e

def sigmoid(x: float) -> float:
    return 1/(1 + pow(e, -x))