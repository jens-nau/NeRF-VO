def set_logging_prefix(metrics: dict, prefix: str) -> dict:
    metrics_with_prefix = {}
    for key, value in metrics.items():
        new_key = prefix + str(key)
        metrics_with_prefix[new_key] = value
    return metrics_with_prefix


def step_check(step: int, step_size: int, run_at_zero: bool = False) -> bool:
    if step_size == 0:
        return False
    return (run_at_zero or step != 0) and step % step_size == 0
