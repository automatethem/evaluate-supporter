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

predictions = [
    {
        'boxes': torch.Tensor([[258.0, 41.0, 606.0, 285.0]]), 
        'scores': torch.Tensor([0.536]),
        'labels': torch.Tensor([0])
    }]
references=[
    {
        'boxes': torch.Tensor([[214.0, 41.0, 562.0, 285.0]]), 
        'labels': torch.Tensor([0])
    }
]

#train_metrics.add_batch(predictions=predictions, references=references)
#train_metrics_value = train_metrics.compute()
train_metrics_value = train_metrics.compute(predictions=[{predictions=predictions, references=references)

print(train_metrics_value['mean_average_precision']) #0.6000000238418579
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
                "predictions": {
                    'boxes': datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                    "scores": datasets.Sequence(datasets.Value("float32")),
                    "labels": datasets.Sequence(datasets.Value("int32")),
                },
                "references": {
                    'boxes': datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                    "labels": datasets.Sequence(datasets.Value("int32"))
                }
            })              
        )

    def _compute(self, predictions, references):
        predictions = [{key: torch.Tensor(value) for key, value in prediction.items()} for prediction in predictions]
        references = [{key: torch.Tensor(value) for key, value in reference.items()} for reference in references]
        metric = MAP()
        metric.update(predictions, references)
        return {"mean_average_precision": metric.compute()['map'].item()}
