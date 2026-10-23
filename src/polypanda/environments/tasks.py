"""Task registry: which environment, config and observation schema belong to each skill."""

from dataclasses import dataclass

from polypanda.environments import grasp, reach, transport


@dataclass(frozen=True)
class TaskSpec:
    env_cls: type
    config_cls: type
    obs_schema: str


TASKS = {
    "reach": TaskSpec(reach.ReachEnv, reach.ReachConfig, reach.OBS_SCHEMA),
    "grasp": TaskSpec(grasp.GraspEnv, grasp.GraspConfig, grasp.OBS_SCHEMA),
    "transport": TaskSpec(transport.TransportEnv, transport.TransportConfig,
                          transport.OBS_SCHEMA),
}


def task_spec(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; known: {sorted(TASKS)}") from None
