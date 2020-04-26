import numpy as np, os
from layers import LinearLayer, SigmoidLayer, xavier
from mnistparser import load_data
from train import BatchIterator, train
from modelevaluator import ModelEvaluator
from neuralnet import MLP, SquaredError

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
    train_set, test_set = load_data(mnist_path)
    train_images = train_set[0]
    train_labels = train_set[1]
    test_images = test_set[0]
    test_labels = test_set[1]

    layers = [
        LinearLayer(32, 28 ** 2, xavier),
        SigmoidLayer(),
        LinearLayer(32, 32, xavier),
        SigmoidLayer(),
        LinearLayer(10, 32, xavier),
        SigmoidLayer()
    ]
    net = MLP(layers)
    loss_func = SquaredError()

    train(
        net,
        train_images,
        train_labels,
        flatten_mnist_input,
        mnist_label_as_one_hot,
        epoch_count=2000,
        batch_size=32
    )

    post_train_evaluator = ModelEvaluator(np.arange(10), "POST-TRAIN")
    for image, label_float in zip(test_images, test_labels):
        output = net.predict(flatten_mnist_input(image))
        post_train_evaluator.receive(
            highest_output_neuron(output),
            int(label_float), 
            loss_func.loss(output, mnist_label_as_one_hot(label_float)))
    
    post_train_evaluator.plot()

if __name__ == "__main__":
    main()