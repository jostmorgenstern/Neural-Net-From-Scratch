from typing import List
from layers import LinearLayer
from mathtypes import Vector, Tensor
import numpy as np

class Loss():
    def loss_func(self, predicted: Vector, actual: Vector) -> Vector:
        raise NotImplementedError

    def grad_func(self, predicted: Vector, actual: Vector) -> Vector:
        raise NotImplementedError

class SquaredError(Loss):
    def loss_func(self, predicted: Vector, actual: Vector) -> Vector:
        return np.sum((predicted - actual) ** 2)

    def grad_func(self, predicted: Vector, actual: Vector) -> Vector:
        return 2 * (predicted - actual) 

class NeuralNet:
    def __init__(self, layers: List, loss: Loss = SquaredError()):
        self.layers = layers
        self.loss = loss

    def predict(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def gradient_layers(self) -> List:
        return [layer for layer in self.layers if layer.has_params()]

    def gradient_step(self, learning_rate: int):
        for layer in self.gradient_layers():
            layer.gradient_step(learning_rate)