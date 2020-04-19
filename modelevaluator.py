import typing, numpy as np, pandas as pd, seaborn as sn, matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, classes: list, title=""):
        """axis 0 is predicted, axis 1 is actual"""
        self.confusion_matrix = pd.DataFrame(np.zeros((len(classes), len(classes))), classes, classes)
        self.total_loss = 0
        self.title = title

    def receive(self, predicted, actual, loss: int):
        self.confusion_matrix.loc[predicted, actual] += 1
        self.total_loss += loss

    def plot(self, path=None):
        sn.set(font_scale=1.4)
        heatmap = sn.heatmap(self.confusion_matrix, annot=True, annot_kws={"size": 16})
        heatmap.set(xlabel='actual', ylabel='predicted')
        heatmap.set_title(
            f"{self.title} \n" +
            f"total loss: {self.total_loss} \n"
        )   
        if path is None:
            plt.show()
        else:
            plt.savefig(path)