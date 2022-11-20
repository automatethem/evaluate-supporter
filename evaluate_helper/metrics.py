import evaluate
import datasets
import numpy as np
import math

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
