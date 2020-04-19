from numpy import ndarray, exp

def sigmoid(input: ndarray) -> ndarray:
    return 1/(1 + exp(-input))

def d_sigmoid(input: ndarray) -> ndarray:
        return sigmoid(input) * (1 - sigmoid(input))