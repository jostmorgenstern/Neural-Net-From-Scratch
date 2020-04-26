import numpy as np
from neuralnet import Loss, SquaredError, NeuralNet
from typing import Callable
from tqdm import tqdm

def BatchIterator(inputs: np.ndarray, labels: np.ndarray, batch_size: int):
    indices = np.arange(0, len(inputs))        
    np.random.shuffle(indices)
    current_start = 0
    
    while True:
        if current_start + batch_size > len(indices):
            np.random.shuffle(indices)
            current_start = 0
        else:
            slice = indices[current_start : current_start + batch_size]
            current_start = current_start + batch_size
            yield zip(inputs[slice], labels[slice])

def train(
    net: NeuralNet,
    train_inputs: np.ndarray,
    train_labels: np.ndarray, 
    input_converter: Callable,
    label_converter: Callable,
    loss_func: Loss = SquaredError(),
    epoch_count: int = 5000,
    batch_size: int = 32,
    learning_rate: int = 0.1):

    batch_iterator = BatchIterator(train_inputs, train_labels, batch_size)
    pbar = tqdm(total=epoch_count)
    for epoch in range(epoch_count):
        epoch_loss = 0
        batch = next(batch_iterator)
        for input, label in batch:
            vector_input = input_converter(input)
            vector_label = label_converter(label)
            output = net.predict(vector_input)
            epoch_loss += loss_func.loss(output, vector_label)
            grad = loss_func.grad(output, vector_label)
            net.backward(grad)
            net.gradient_step(learning_rate / batch_size)
        pbar.update()
        pbar.set_description(desc=f"Current epoch loss: {round(epoch_loss, 2)}")

