import torch
import evaluate
import datasets
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP

'''
#https://medium.com/data-science-at-microsoft/how-to-smoothly-integrate-meanaverageprecision-into-your-training-loop-using-torchmetrics-7d6f2ce0a2b3
import evaluate
import evaluate_helper

train_metrics = evaluate.combine([
    evaluate_helper.metrics.MeanAveragePrecision()
])

prediction_boxes = [[258.0, 41.0, 606.0, 285.0]]
prediction_scores = [0.536]
prediction_labels = [0]
reference_boxes = [[214.0, 41.0, 562.0, 285.0]]
reference_labels = [0]

#train_metrics_value = train_metrics.compute(prediction_boxes=prediction_boxes, prediction_scores=prediction_scores, prediction_labels=prediction_labels, reference_boxes=reference_boxes, reference_labels=reference_labels)
#print(train_metrics_value['mean_average_precision']) #0.6000000238418579
train_metrics.add_batch(prediction_boxes=prediction_boxes, prediction_scores=prediction_scores, prediction_labels=prediction_labels, reference_boxes=reference_boxes, reference_labels=reference_labels)
train_metrics_value = train_metrics.compute()
print(train_metrics_value['mean_average_precision']) #
'''
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
        return {"mean_average_precision": metric.compute()['map'].item()}
