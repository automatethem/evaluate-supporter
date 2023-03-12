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
