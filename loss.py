import numpy as np

class Loss():
    def loss(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class MSE(Loss):
    def loss(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return 2 * (predicted - actual)