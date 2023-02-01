import evaluate
import datasets
import numpy as np
import math
from sklearn.metrics import accuracy_score

class CustomAccuracy(evaluate.Metric):
    def __init__(self, metrics_name='accuracy', **kwargs):
        super().__init__(**kwargs)
        self.metrics_name = metrics_name
    
    def _info(self):
        return evaluate.MetricInfo(
            description='',
            citation='',
            inputs_description='',
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32"),
                    "references": datasets.Value("float32")
                } if self.config_name != 'multilabel'
                else {
                    "predictions": datasets.Sequence(datasets.Value("float32")),
                    "references": datasets.Sequence(datasets.Value("float32")),
                }
            )
        )

    def _compute(self, predictions, references):
        accuracy = accuracy_score(references, predictions)
        return {f'{self.metrics_name}': accuracy}
