import os
import typing as t

import numpy as np
import torch


class NumpySaver:
    """Saver for embedding saving and loading, all data will be converted to numpy and saved as .npz."""

    def __init__(self):
        pass

    def exists(self, path: str) -> bool:
        if not path.endswith('.npz'):
            path = path + '.npz'
        return os.path.exists(path)

    def save(self, embed: t.Dict[str, t.Union[torch.Tensor, int, float]], path: str):
        # For dict of tensors, convert each tensor to numpy and save as a compressed .npz
        np_dict = {}
        for k, v in embed.items():
            if isinstance(v, torch.Tensor):
                np_dict[k] = v.detach().cpu().numpy()
            else:
                np_dict[k] = np.array(v)
        np.savez_compressed(path, **np_dict)

    def load(self, path: str):
        if not path.endswith('.npz'):
            path = path + '.npz'
        # Load the .npz file and convert numpy arrays back to torch tensors
        data = np.load(path, allow_pickle=False)
        return {k: torch.from_numpy(v) for k, v in data.items()}


def get_saver(
    saver_name: str,
) -> NumpySaver:
    saver = None
    if saver_name == 'numpy':
        saver = NumpySaver()
    else:
        raise ValueError(f'{saver_name} is not a supported saver.')
    return saver
