import torch
import evaluate
import datasets
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP

#evaluate 패키지에서 아직 지원되지 않는 mAP 를 사용하기 위해 evaluate 패키지의 사용자 정의 매트릭 클래스를 구현
#MeanAveragePrecision: 클래스별로 Average Precision (Precision-Recall 곡선의 아래쪽 면적) 를 구한후 그 값을 평균.
#mAP 값이 클수록 좋음.
class MeanAveragePrecision(evaluate.Metric):
    def __init__(self, metrics_name='mean_average_precision', **kwargs):
        super().__init__(**kwargs)
        self.metrics_name = metrics_name

    def _info(self):
        return evaluate.MetricInfo(
            description='',
            citation='',
            inputs_description='',
            features=datasets.Features(
                {
                    "predictions": {
                        'boxes': datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                        "scores": datasets.Sequence(datasets.Value("float32")),
                        "labels": datasets.Sequence(datasets.Value("int32")),
                    },
                    "references": {
                        'boxes': datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                        "labels": datasets.Sequence(datasets.Value("int32"))
                    }
                }
            )              
        )

    def _compute(self, predictions, references):
        predictions = [{key: torch.Tensor(value) for key, value in prediction.items()} for prediction in predictions]
        references = [{key: torch.Tensor(value) for key , value in reference.items()} for reference in references]
        metric = MAP()
        metric.update(predictions, references)
        return {"mean_average_precision": metric.compute()['map'].item()}
