from typing import Iterable, Optional

import torch
import torch.nn as nn


def get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    lr: float,
    params: Optional[Iterable[torch.nn.Parameter]] = None,
) -> torch.optim.Optimizer:
    if params is None:
        params = model.parameters()

    if optimizer_name == 'momentum':
        optimizer = torch.optim.SGD(
            params=params,
            lr=lr,
            momentum=0.9,
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            params=params,
            lr=lr,
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            params=params,
            lr=lr,
        )
    else:
        raise NotImplementedError(f'{optimizer_name} is not a supported optimizer.')

    return optimizer
