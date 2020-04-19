import typing
from Layers import LinearLayer
import numpy as np

class MLP:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def linear_layers(self):
        return [layer for layer in self.layers if isinstance(layer, LinearLayer)]

    def gradient_step(self, learning_rate: int):
        for layer in self.linear_layers():
            layer.W -= learning_rate * layer.W_grad
            layer.b -= learning_rate * layer.b_grad