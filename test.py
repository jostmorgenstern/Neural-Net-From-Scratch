import numpy as np, seaborn as sn, matplotlib.pyplot as plt
from neuralnet import NeuralNet, Loss, SquaredError
from pandas import DataFrame
from tqdm import tqdm
from typing import Callable

class ModelEvaluator:
    """useful for testing models by plotting a confusion matrix and showing the total loss"""
    def __init__(self, confusion_matrix: DataFrame, title=''):
        """axis 0 is predicted, axis 1 is actual"""
        self.confusion_matrix = confusion_matrix
        self.total_loss = 0
        self.title = title

    def receive(self, predicted, actual, loss: int):
        self.confusion_matrix.loc[predicted, actual] += 1
        self.total_loss += loss

    def plot(self, path=None):
        sn.set(font_scale=1.4)
        heatmap = sn.heatmap(self.confusion_matrix, annot=True, annot_kws={"size": 16})
        heatmap.set(xlabel='actual', ylabel='predicted')
        heatmap_as_numpy = self.confusion_matrix.to_numpy()
        heatmap.set_title(
            f"{self.title} \n" +
            f"total loss: {self.total_loss} \n" +
            f"Correct: {np.trace(heatmap_as_numpy)} / {np.sum(heatmap_as_numpy)}"
        )
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

def test(
    net: NeuralNet,
    inputs: np.ndarray,
    labels: np.ndarray,
    confusion_matrix: DataFrame,
    input_converter: Callable,
    output_converter: Callable,
    label_converter: Callable,
    loss: Loss = SquaredError(),
    title=''
) -> ModelEvaluator:
    evaluator = ModelEvaluator(confusion_matrix, title=title)
    pbar = tqdm(total=len(labels))
    for input, label in zip(inputs, labels):
        output = net.predict(input_converter(input))
        evaluator.receive(output_converter(output), label, loss.loss(output, label_converter(label)))
        pbar.update()
        pbar.set_description(desc=f"Testing model")
    return evaluator