from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_machine_translation_adapter_spec,
    get_multiple_choice_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_generation_metric_specs,
    get_basic_metric_specs,
    get_exact_match_metric_specs,
    get_f1_metric_specs,
    get_generative_harms_metric_specs,
    get_generic_metric_specs,
    get_open_ended_generation_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.runner import get_benchmark_output_path
from helm.benchmark.scenarios.scenario import ScenarioSpec, get_scenario_cache_path


@run_spec_function("unitxt")
def get_unitxt_spec(**kwargs) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.unitxt_scenario.UnitxtScenario", args=kwargs)
    adapter_spec = get_generation_adapter_spec()
    return RunSpec(
        name="unitxt",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=[
            MetricSpec(class_name="helm.benchmark.metrics.unitxt_metrics.UnitxtMetric", args=kwargs)
        ],  # get_basic_generation_metric_specs(["exact_match_indicator"])+ get_generic_metric_specs(),
        groups=["unitxt"],
    )
