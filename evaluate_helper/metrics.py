import evaluate
import datasets
import numpy as np
import math

from sklearn.metrics import accuracy_score
import evaluate
class Accuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description='',
            citation='',
            inputs_description='',
            features=datasets.Features({
                "predictions": datasets.Value("float32"),
                "references": datasets.Value("float32")
            })
        )

    def _compute(self, predictions, references):
        accuracy = accuracy_score(references, predictions)
        return {"accuracy": accuracy}

import datasets
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP
#evaluate 패키지에서 아직 지원되지 않는 mAP 를 사용하기 위해 evaluate 패키지의 사용자 정의 매트릭 클래스를 구현
#MeanAveragePrecision: 클래스별로 AP (Precision-Recall 곡선의 아래 면적) 를 구한후 그 값을 평균.
#mAP 값이 클수록 좋음.
class MeanAveragePrecision(evaluate.Metric):
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
        '''
        prediction_boxes = prediction_boxes
        prediction_scores = prediction_scores
        prediction_labels = prediction_labels
        reference_boxes = reference_boxes
        reference_labels = reference_labels
        '''
        '''
        confidences = []
        for confidence in prediction_scores[0]:
            confidences.append(confidence[confidence.argmax(-1)])
        confidences = torch.Tensor(confidences) 
        '''
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
        return {"map": metric.compute()['map']}
    
# PSNR function: 모델의 출력값과 high-resoultion의 유사도를 측정합니다.
# PSNR 값이 클수록 좋습니다.
class Psnr(evaluate.Metric):
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
        return {"psnr": psnr}
