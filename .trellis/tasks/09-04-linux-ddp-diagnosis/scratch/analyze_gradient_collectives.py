from pathlib import Path

from ncls.learning.methods import get_method_plugin
from ncls.learning.training import TrainingPlanResolver
from ncls.paths import PROJECT_ROOT


plan = TrainingPlanResolver(PROJECT_ROOT).resolve(
    Path("configs/training/runs/metal-linux-long.yaml")
)
config = plan.to_runtime_config()
plugin = get_method_plugin(plan.selection.method)
model = plugin.model_factory.create(config.model_context)
named_parameters = dict(model.named_parameters())

total_bytes = sum(
    parameter.numel() * parameter.element_size()
    for parameter in named_parameters.values()
)
print(
    f"model_parameter_tensors={len(named_parameters)} "
    f"total_numel={sum(parameter.numel() for parameter in named_parameters.values())} "
    f"total_mib={total_bytes / 2**20:.3f}"
)

for phase in config.phases:
    names = [
        name
        for group in phase.parameter_groups
        for name in plugin.descriptor.parameter_groups[group]
    ]
    sizes = [
        named_parameters[name].numel() * named_parameters[name].element_size()
        for name in names
    ]
    print(
        f"phase={phase.name} "
        f"groups={len(phase.parameter_groups)} "
        f"allreduce_calls_per_step={len(names)} "
        f"gradient_mib={sum(sizes) / 2**20:.3f} "
        f"tensors_le_4k={sum(size <= 4 * 1024 for size in sizes)} "
        f"tensors_le_64k={sum(size <= 64 * 1024 for size in sizes)} "
        f"max_tensor_mib={max(sizes) / 2**20:.3f}"
    )
