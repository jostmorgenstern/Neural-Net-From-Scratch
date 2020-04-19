import typing, numpy as np

def random(height: int, width: int):
    return np.random.randn(height, width)

def he(height: int, width: int):
    return np.random.randn(height, width) * np.sqrt(2/width)

def xavier(height: int, width: int):
    return np.random.randn(height, width) * np.sqrt(1/width)

def zero(height: int, width: int):
    return np.zeros(height, width)