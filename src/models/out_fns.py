import torch.nn.functional as F


def get_outfns(tasks: list[str]) -> dict:
    return {task: _outfn(task) for task in tasks}


def _outfn(task: str):
    if task == "water_mask":
        return lambda x: F.sigmoid(x)
    if task == "semantic":
        return lambda x: F.log_softmax(x, dim=1)
    if task == "depth":
        return lambda x: x
    if task == "normal":
        return lambda x: F.normalize(x, p=2, dim=1)
    raise ValueError(f"Unknown task: {task}")
