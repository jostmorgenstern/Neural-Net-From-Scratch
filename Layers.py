import typing
import numpy as np
from activation import sigmoid, d_sigmoid

F = typing.Callable

class Layer:
    def forward(input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LinearLayer(Layer):
    def __init__(self, output_size: int, input_size: int, weight_init: F[[int, int], np.ndarray]):
        self.W = weight_init(output_size, input_size)
        self.b = np.zeros(output_size)
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return self.W @ input + self.b
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.b_grad = grad
        self.W_grad = grad.reshape((grad.shape[0], 1)) @ self.input.reshape((1, self.input.shape[0]))
        return self.W.T @ grad

    def __repr__(self):
        return f"<LinearLayer {self.W.shape}>"

class ActivationLayer(Layer):
    def __init__(self, f: F[[np.ndarray], np.ndarray], derivative_f: F[[np.ndarray], np.ndarray]):
        self.f = f
        self.derivative_f = derivative_f
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return self.f(input)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.derivative_f(self.input)

class SigmoidLayer(ActivationLayer):
    def __init__(self):
        super().__init__(sigmoid, d_sigmoid)
    
    def __repr__(self):
        return "<SigmoidLayer>"