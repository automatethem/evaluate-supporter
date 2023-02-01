from .custom_accuracy import *

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
    
# PSNR function: 모델의 출력값과 high-resoultion의 유사도를 측정합니다.
# PSNR 값이 클수록 좋습니다.
class Psnr(evaluate.Metric):
    def __init__(self, metrics_name='psnr', **kwargs):
        super().__init__(**kwargs)
        self.metrics_name = metrics_name
    
    def _info(self):
        return evaluate.MetricInfo(
            description='',
            citation='',
            inputs_description='',
            features=datasets.Features({
                "predictions": datasets.Sequence(datasets.Sequence(datasets.Sequence(datasets.Value("float32")))),
                "references": datasets.Sequence(datasets.Sequence(datasets.Sequence(datasets.Value("float32"))))
            })
        )

    def _compute(self, predictions, references, max_val=1.):
        references = np.array(references)
        predictions = np.array(predictions)
        img_diff = predictions - references
        rmse = math.sqrt(np.mean((img_diff)**2))
        if rmse == 0: # label과 output이 완전히 일치하는 경우
            psnr = 100
        else:
            psnr = 20 * math.log10(max_val/rmse)
        return {f'{self.metrics_name}': psnr}

import evaluate
import datasets

#train_metrics = evaluate.combine([
#    AverageMetrics('perplexity')
#])
#train_metrics.add_batch(predictions=[math.exp(loss.item())])
#train_metrics_value = train_metrics.compute()
class AverageMetrics(evaluate.Metric):
    def __init__(self, metrics_name='average_metrics', **kwargs):
        super().__init__(**kwargs)
        self.metrics_name = metrics_name

    def _info(self):
        return evaluate.MetricInfo(
            description='',
            citation='',
            inputs_description='',
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32")
                }
            )
        )

    def _compute(self, predictions):
        average = np.mean((predictions))
        return {f'{self.metrics_name}': average}
