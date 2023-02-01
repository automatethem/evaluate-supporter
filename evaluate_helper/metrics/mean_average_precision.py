import torch
import datasets
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP
#evaluate 패키지에서 아직 지원되지 않는 mAP 를 사용하기 위해 evaluate 패키지의 사용자 정의 매트릭 클래스를 구현
#MeanAveragePrecision: 클래스별로 AP (Precision-Recall 곡선의 아래 면적) 를 구한후 그 값을 평균.
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
            features=datasets.Features({
                "prediction_boxes": datasets.Sequence(datasets.Value("float32")),
                "prediction_scores": datasets.Value("float32"),
                "prediction_labels": datasets.Value("float32"),
                "reference_boxes": datasets.Sequence(datasets.Value("float32")),
                "reference_labels": datasets.Value("float32")
            })            
        )

    def _compute(self, prediction_boxes, prediction_scores, prediction_labels, reference_boxes, reference_labels):
        preds = [
            dict(
                boxes=torch.Tensor(prediction_boxes),
                scores=torch.Tensor(prediction_scores),
                labels=torch.Tensor(prediction_labels),
            )
        ]
        target = [
            dict(
                boxes=torch.Tensor(reference_boxes),
                labels=torch.Tensor(reference_labels),
            )
        ]
        #print(preds)
        #print(target)
        metric = MAP()
        metric.update(preds, target)
        return {f'{self.metrics_name}': metric.compute()['map'].item()}
    
