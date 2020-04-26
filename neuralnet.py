from typing import List
from layers import LinearLayer
from mathtypes import Vector, Tensor
import numpy as np

class NeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

class MLP(NeuralNet):    
    def linear_layers(self) -> List:
        return [layer for layer in self.layers if isinstance(layer, LinearLayer)]

    def gradient_step(self, learning_rate: int):
        for layer in self.linear_layers():
            layer.W -= learning_rate * layer.W_grad
            layer.b -= learning_rate * layer.b_grad

class Loss():
    def loss(self, predicted: Vector, actual: Vector) -> Vector:
        raise NotImplementedError

    def grad(self, predicted: Vector, actual: Vector) -> Vector:
        raise NotImplementedError

class SquaredError(Loss):
    def loss(self, predicted: Vector, actual: Vector) -> Vector:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Vector, actual: Vector) -> Vector:
        return 2 * (predicted - actual)