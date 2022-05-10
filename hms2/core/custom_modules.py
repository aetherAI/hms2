"""
This module contains custom modules..
"""
import abc
from typing import Sequence, Tuple, Type, Union

import cv2
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn


class BaseAugmentorModule(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def randomize(self) -> None:
        pass


class HEDPerturbAugmentorModule(BaseAugmentorModule):
    """
    An image augmentor that implements HED perturbing.

    Args:
        stain_angle (float): The maximal angle applied on perturbing the stain matrix.
        concentration_multiplier (tuple-like):
            A two-element tuple defining the scaling range of concentration perturbing.
        skip_background (bool):
            Skip this augmentation on background since it's unneccesary.
    """

    def __init__(
        self,
        stain_angle: float = 10.0,
        concentration_multiplier: Tuple[float, float] = (0.5, 1.5),
        skip_background: bool = True,
    ):
        super().__init__()
        self.stain_angle = stain_angle
        self.concentration_multiplier = concentration_multiplier
        self.skip_background = skip_background

        self.eps = 1e-6
        rgb_from_hed = np.array(
            [
                [0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11],
                [0.27, 0.57, 0.78],
            ]
        )
        self.hed_from_rgb = scipy.linalg.inv(rgb_from_hed)
        self.postfix = None

    def randomize(self) -> None:
        stain_angle_rad = np.radians(self.stain_angle)
        hed_from_rgb_aug = []
        for stain_idx in range(self.hed_from_rgb.shape[1]):
            stain = self.hed_from_rgb[:, stain_idx]
            stain_rotation_vector = np.random.uniform(
                -stain_angle_rad, stain_angle_rad, size=(3,)
            )
            stain_rotation_matrix, _ = cv2.Rodrigues(np.array([stain_rotation_vector]))
            stain_aug = np.matmul(stain_rotation_matrix, stain[:, np.newaxis])
            hed_from_rgb_aug.append(stain_aug)
        hed_from_rgb_aug = np.concatenate(hed_from_rgb_aug, axis=1)
        rgb_from_hed_aug = scipy.linalg.inv(hed_from_rgb_aug)

        concentration_aug_matrix = np.diag(
            np.random.uniform(*self.concentration_multiplier, size=(3,)),
        )

        # image_od_aug = image_od . hed_from_rgb . concentration_aug_matrix .
        # rgb_from_hed_aug
        postfix = np.matmul(concentration_aug_matrix, rgb_from_hed_aug)
        postfix = np.matmul(self.hed_from_rgb, postfix)
        self.postfix = postfix

    @torch.no_grad()
    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        if self.postfix is None:
            raise RuntimeError("randomize() should be called before forward().")

        # When the image is all white, this augmentation will not make any change, so
        # skip it.
        if self.skip_background and torch.all(image_batch == 1.0).item():
            return image_batch

        image_batch = torch.clamp(image_batch, min=self.eps)
        image_batch_od = torch.log(image_batch) / np.log(self.eps)
        image_batch_od = image_batch_od.permute(0, 2, 3, 1).contiguous()
        postfix = torch.tensor(self.postfix, dtype=torch.float32).to(
            image_batch_od.device
        )
        image_batch_od_aug = torch.matmul(image_batch_od, postfix)
        image_batch_od_aug = image_batch_od_aug.permute(0, 3, 1, 2).contiguous()
        image_batch_od_aug = torch.clamp(image_batch_od_aug, min=0.0)
        image_batch_aug = torch.exp(image_batch_od_aug * np.log(self.eps))
        image_batch_aug = torch.ceil(image_batch_aug * 255.0) / 255.0

        return image_batch_aug


class FrozenBatchNorm2d(nn.BatchNorm2d):
    """
    Batch normalization for 2D tensors with a frozen running mean and variance. Use the
    classmethod `convert_frozen_batchnorm` to rapidly convert a module containing batch
    normalization layers.

    Args:
        Refer to the descriptions in `torch.nn.BatchNorm2d`.
    """

    _version = 1

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__(
            num_features=num_features,
            eps=eps,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward operations. Mean and variance calculations are removed.
        """
        self._check_input_dim(input_tensor)

        output = nn.functional.batch_norm(
            input=input_tensor,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=False,
            eps=self.eps,
        )

        return output

    @classmethod
    def convert_frozen_batchnorm(
        cls: Type["FrozenBatchNorm2d"], module: nn.Module
    ) -> nn.Module:
        """
        Convert a module with batch normalization layers to frozen one.
        """
        bn_module = (
            nn.modules.batchnorm.BatchNorm2d,
            nn.modules.batchnorm.SyncBatchNorm,
        )
        if isinstance(module, bn_module):
            frozen_bn = cls(
                num_features=module.num_features,
                eps=module.eps,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
            ).to(device=next(module.parameters()).device)
            if module.affine:
                with torch.no_grad():
                    frozen_bn.weight.copy_(module.weight)
                    frozen_bn.bias.copy_(module.bias)
            if module.track_running_stats:
                with torch.no_grad():
                    frozen_bn.running_mean.copy_(module.running_mean)
                    frozen_bn.running_var.copy_(module.running_var)
                    frozen_bn.num_batches_tracked.copy_(module.num_batches_tracked)
            module = frozen_bn
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    module.add_module(name, new_child)

        return module


class LogSumExpPool2d(nn.Module):
    def __init__(self, factor: float = 1.0):
        super().__init__()
        self.factor = factor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, _, height, width = inputs.shape

        max_pool = nn.functional.adaptive_max_pool2d(inputs, output_size=(1, 1))
        exp = torch.exp(self.factor * (inputs - max_pool))
        sumexp = torch.sum(exp, dim=(2, 3), keepdim=True) / (height * width)
        logsumexp = max_pool + torch.log(sumexp) / self.factor

        return logsumexp


class PermuteLayer(nn.Module):
    def __init__(self, dims: Sequence[int]):
        super().__init__()
        self.dims = dims

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs.permute(*self.dims)
        return output


class ScaleAndShift(nn.Module):
    __constants__ = ["scale", "bias"]

    def __init__(self, scale=1.0, bias=0.0):
        super().__init__()
        self.scale = scale
        self.bias = bias

    def forward(self, inputs):
        return inputs * self.scale + self.bias

    def extra_repr(self):
        return "scale={}, bias={}".format(self.scale, self.bias)


class ToDevice(nn.Module):
    def __init__(self, device: Union[torch.device, str]):
        super().__init__()
        self.device = device

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input.to(self.device)
        return output
