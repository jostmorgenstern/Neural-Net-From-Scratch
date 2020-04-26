import numpy as np, os
from layers import LinearLayer, SigmoidLayer, xavier
from mnistparser import load_data
from train import BatchIterator, train
from neuralnet import NeuralNet, SquaredError
from pandas import DataFrame
from test import test, ModelEvaluator

def mnist_label_as_one_hot(label: int) -> np.ndarray:
    vector = np.zeros(10)
    vector[int(label)] = 1
    return vector

def flatten_mnist_input(input: np.ndarray) -> np.ndarray:
    return input.reshape(28 ** 2)

def highest_output_neuron(output: np.ndarray) -> int:
    return np.argwhere(output == np.max(output))[0,0]

def main():
    mnist_path = os.path.join(os.getcwd(), "MNIST")
    (train_images, train_labels), (test_images, test_labels) = load_data(mnist_path)
    
    layers = [
        LinearLayer(32, 28 ** 2, xavier),
        SigmoidLayer(),
        LinearLayer(32, 32, xavier),
        SigmoidLayer(),
        LinearLayer(10, 32, xavier),
        SigmoidLayer()
    ]
    net = NeuralNet(layers)

    np.seterr(over='ignore')
    train(
        net, train_images, train_labels, flatten_mnist_input, mnist_label_as_one_hot,
        epoch_count=1000, batch_size=1
    )

    confusion_matrix = DataFrame(np.zeros((10, 10)), index=range(10), columns=range(10))
    evaluator = test(
        net, test_images, test_labels, confusion_matrix,
        flatten_mnist_input, highest_output_neuron, mnist_label_as_one_hot,
        title="POST-TRAIN"
    )
    evaluator.plot()

if __name__ == "__main__":
    main()