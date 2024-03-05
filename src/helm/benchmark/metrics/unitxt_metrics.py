from abc import abstractmethod
from typing import List, cast

from datasets import load_dataset
from unitxt import get_from_catalog
from unitxt.metrics import InstanceMetric, GlobalMetric

from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


def _generate_instances(scenario_state: ScenarioState):
    for request_state in scenario_state.request_states:
        yield {
            "references": [
                correct_reference.output.text for correct_reference in request_state.instance.all_correct_references
            ],
            "prediction": request_state.result.completions[0].text,
        }


class UnitxtMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__()
        dataset_name = ",".join(f"{key}={value}" for key, value in kwargs.items())
        dataset = load_dataset("unitxt/data", dataset_name, trust_remote_code=True)
        row = next(iter(dataset.values()))[0]
        post_processors_names = row.get("postprocessors", [])
        # TODO: Handle post processors
        metric_names = row.get("metrics", [])
        self.metrics = [get_from_catalog(metric_name) for metric_name in metric_names]

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        metric_result = super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism)

        stats = []
        for metric in self.metrics:
            if isinstance(metric, GlobalMetric):
                global_metric = cast(GlobalMetric, metric)
                score_dict = next(global_metric.process(_generate_instances(scenario_state)))
                score_name = score_dict["score"]["global"]["score_name"]
                score = score_dict["score"]["global"][score_name]
                stats.append(Stat(MetricName(score_name)).add(score))
        metric_result.aggregated_stats.extend(stats)
        return metric_result

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Evaluate free-form generation.  Override me!"""
        stats = []
        for metric in self.metrics:
            if isinstance(metric, InstanceMetric):
                correct_references = [
                    correct_reference.output.text for correct_reference in request_state.instance.all_correct_references
                ]
                metric_output = cast(InstanceMetric, metric).compute(
                    references=correct_references, prediction=request_state.result.completions[0].text, task_data={}
                )
                score_name = metric_output["score_name"]
                score = metric_output[score_name]
                stats.append(Stat(MetricName(score_name)).add(score))

        return stats
