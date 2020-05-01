from typing import Callable, Tuple
from mathtypes import Vector, Matrix, Tensor
import numpy as np
from scipy.ndimage import correlate

class Layer:
    def forward(input):
        raise NotImplementedError

    def backward(input):
        raise NotImplementedError

class LinearLayer(Layer):
    def __init__(self, output_size: int, input_size: int, weight_init: Callable):
        self.W = weight_init((output_size, input_size))
        self.b = np.zeros(output_size)
    
    def forward(self, input: Vector) -> Vector:
        self.input = input
        return self.W @ input + self.b
    
    def backward(self, grad: Vector) -> Vector:
        self.b_grad = grad
        self.W_grad = grad.reshape((grad.shape[0], 1)) @ self.input.reshape((1, self.input.shape[0]))
        return self.W.T @ grad

    def __repr__(self):
        return f"<LinearLayer {self.W.shape}>"

    def has_params(self):
        return True

    def gradient_step(self, learning_rate: int):
        self.W -= learning_rate * self.W_grad
        self.b -= learning_rate * self.b_grad

class ConvLayer(Layer):
    def __init__(self, filter_count: int, channel_count: int, filter_height: int, filter_width: int, weight_init: Callable[[Tuple[int, ...]], Tensor]):
        self.filters = weight_init((filter_count, channel_count, filter_height, filter_width))

    def forward(self, input: Tensor) -> Tensor:
        self.input = input   
        return self.convolve_all(input, self.filters)

    
    def convolve_all(self, feature_maps, kernels):
        return np.array([self.convolve_single_map(feature_map, kernel) for kernel in kernels for feature_map in feature_maps])

    def convolve_single(self, feature_map, kernel):
        return sum([correlate(image_channel, kernel_channel) for image_channel, kernel_channel in zip(feature_map, kernel)])
    
    def backward(self, grad: Tensor) -> Tensor:

    
    
    def has_params(self):
        return True
    
    def __repr__(self):
        return f"<ConvLayer {self.filters.shape}>"

class ActivationLayer(Layer):
    """ template for subclasses """
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return self.f(input)

    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.derivative_f(self.input)

    def has_params(self):
        return False

class SigmoidLayer(ActivationLayer):
    def __repr__(self):
        return "<SigmoidLayer>"

    def f(self, input: Tensor) -> Tensor:
        return 1/(1 + np.exp(-input))

    def derivative_f(self, input: Tensor) -> Tensor:
        return self.f(input) * (1 - self.f(input))

def random(shape: Tuple[int, ...]) -> Tensor:
    return np.random.randn(*shape)

def he(shape: Tuple[int, ...]) -> Tensor:
    return np.random.randn(*shape) * np.sqrt(2 / shape[1])

def xavier(shape: Tuple[int, ...]) -> Tensor:
    return np.random.randn(*shape) * np.sqrt(1 / shape[1])

def zero(shape: Tuple[int, ...]) -> Tensor:
    return np.zeros(shape)