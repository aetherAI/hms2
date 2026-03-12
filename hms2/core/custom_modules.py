"""
This module contains custom modules..
"""
import abc
from typing import Optional, Sequence, Tuple, Type, Union

import cv2
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.transforms.functional


class BaseAugmentorModule(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def randomize(self) -> None:
        pass

    @abc.abstractmethod
    def forward(self, image_batch: torch.Tensor, is_background_tile: bool = False) -> torch.Tensor:
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
            ],
        )
        self.hed_from_rgb = scipy.linalg.inv(rgb_from_hed)
        self.postfix = None

    def randomize(self) -> None:
        stain_angle_rad = np.radians(self.stain_angle)
        hed_from_rgb_aug = []
        for stain_idx in range(self.hed_from_rgb.shape[1]):
            stain = self.hed_from_rgb[:, stain_idx]
            stain_rotation_vector = np.random.uniform(-stain_angle_rad, stain_angle_rad, size=(3,))
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
    def forward(self, image_batch: torch.Tensor, is_background_tile: bool = False) -> torch.Tensor:
        if self.postfix is None:
            raise RuntimeError('randomize() should be called before forward().')

        # When the image is all white, this augmentation will not make any change, so
        # skip it.
        if self.skip_background and is_background_tile:
            return image_batch

        image_batch = torch.clamp(image_batch, min=self.eps)
        image_batch_od = torch.log(image_batch) / np.log(self.eps)
        image_batch_od = image_batch_od.permute(0, 2, 3, 1).contiguous()
        postfix = torch.tensor(self.postfix, dtype=torch.float32).to(image_batch_od.device)
        image_batch_od_aug = torch.matmul(image_batch_od, postfix)
        image_batch_od_aug = image_batch_od_aug.permute(0, 3, 1, 2).contiguous()
        image_batch_od_aug = torch.clamp(image_batch_od_aug, min=0.0)
        image_batch_aug = torch.exp(image_batch_od_aug * np.log(self.eps))
        image_batch_aug = torch.ceil(image_batch_aug * 255.0) / 255.0

        return image_batch_aug


class GaussianBlurAugmentorModule(BaseAugmentorModule):
    """
    An image augmentor that implements random Gaussian blur.

    Args:
        kernel_size: Kernel size for convolutional operations.
        sigma: Range of radius for Gaussian kernels.
    """

    sigma: Optional[float]

    def __init__(self, kernel_size: int = 3, sigma_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

        self.sigma = None

    def randomize(self) -> None:
        self.sigma = torch.empty(1).uniform_(self.sigma_range[0], self.sigma_range[1]).item()

    def forward(self, image_batch: torch.Tensor, is_background_tile: bool = False) -> torch.Tensor:
        if self.sigma is None:
            raise RuntimeError('randomize() should be called before forward().')

        if is_background_tile:
            return image_batch

        blurred_image_batch = torchvision.transforms.functional.gaussian_blur(
            image_batch,
            kernel_size=[self.kernel_size, self.kernel_size],
            sigma=[self.sigma, self.sigma],
        )
        return blurred_image_batch


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
    def convert_frozen_batchnorm(cls: Type['FrozenBatchNorm2d'], module: nn.Module) -> nn.Module:
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
                if not isinstance(frozen_bn.running_mean, torch.Tensor):
                    raise ValueError
                if not isinstance(frozen_bn.running_var, torch.Tensor):
                    raise ValueError
                if not isinstance(frozen_bn.num_batches_tracked, torch.Tensor):
                    raise ValueError
                if not isinstance(module.running_mean, torch.Tensor):
                    raise ValueError
                if not isinstance(module.running_var, torch.Tensor):
                    raise ValueError
                if not isinstance(module.num_batches_tracked, torch.Tensor):
                    raise ValueError
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
        logsumexp = torch.where(
            max_pool == -np.inf,
            torch.tensor(-np.inf, device=inputs.device),
            logsumexp,
        )
        return logsumexp


class SoftplusInv(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        eps = torch.tensor(self.eps, device=input.device)
        threshold = torch.log(eps) + 2.0
        is_too_small = input < torch.exp(threshold)
        is_too_large = input > -threshold
        too_small_value = torch.where(
            input == 0,
            torch.tensor(-np.inf, device=input.device),
            torch.log(input),
        )
        too_large_value = input
        medium_value = input + torch.log(-torch.expm1(-input))
        output = torch.where(
            is_too_small,
            too_small_value,
            torch.where(
                is_too_large,
                too_large_value,
                medium_value,
            ),
        )
        return output


class ReLSEPool2d(nn.Module):
    def __init__(
        self,
        threshold: float = 20.0,
    ) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        re_lse = _compute_re_lse(input, threshold=self.threshold)
        return re_lse


def _compute_re_lse(input: torch.Tensor, threshold: float) -> torch.Tensor:
    # When the scaled_input is small
    secured_scaled_input = torch.clamp(input, max=threshold)
    small_value = torch.log1p(
        torch.expm1(secured_scaled_input).sum(dim=(2, 3), keepdim=True),
    )

    # When the input is large
    max_pool = input.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    large_value = max_pool + torch.log(torch.exp(input - max_pool).sum(dim=(2, 3), keepdim=True))

    # Select the formula
    re_lse = torch.where(
        large_value < threshold,
        small_value,
        large_value,
    )

    return re_lse


class ReCAM(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        n: float = 1e6,
        threshold: float = 20.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n = n
        self.threshold = threshold
        self.eps = eps

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            # When the input is small
            secured_input = torch.clamp(input, max=self.threshold)
            small_value = torch.log1p(self.n * torch.expm1(secured_input))

            # When the input is large
            large_value = input + torch.log(torch.tensor(self.n, device=input.device))

            # Select the formula
            re_cam = torch.where(
                large_value < self.threshold,
                small_value,
                large_value,
            )

        return re_cam


class PermuteLayer(nn.Module):
    def __init__(self, dims: Sequence[int]):
        super().__init__()
        self.dims = dims

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs.permute(*self.dims)
        return output


class ScaleAndShift(nn.Module):
    __constants__ = ['scale', 'bias']

    def __init__(self, scale=1.0, bias=0.0):
        super().__init__()
        self.scale = scale
        self.bias = bias

    def forward(self, inputs):
        return inputs * self.scale + self.bias

    def extra_repr(self):
        return f'scale={self.scale}, bias={self.bias}'


class ToDevice(nn.Module):
    def __init__(self, device: Union[torch.device, str]):
        super().__init__()
        self.device = device

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input.to(self.device)
        return output


class EuclideanFarthestPointPrePooling(torch.nn.Module):

    def __init__(self, side_sample_ratio: float = 1 / 7) -> None:
        super().__init__()
        self.r = side_sample_ratio

    def _flatten_hw(self, x):
        # (B, C, H, W) -> (B, H*W, C)
        return x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))

    def _reshape_hw(self, x, b, c, h, w, r):
        # (B, M, C) -> (B, C, H*r, W*r)
        return x.reshape(b, round(h * r), round(w * r), c).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Farthest point sampling on GPU.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H*r, W*r), where r is the sample ratio.

            Note that even if the output shape is (B, C, H*r, W*r), the actual spatial locations may not form
            a regular grid.  This is because the farthest point sampling selects points based on their feature
            distances, which may not correspond to a regular grid.
        """
        b, c, h, w = x.shape
        x_flat = self._flatten_hw(x)  # (B, N, C)
        n = x_flat.size(1)
        m = round(h * self.r) * round(w * self.r)
        m = max(min(m, n), 1)

        outs = []
        for i in range(b):
            # Precompute norm (without sqrt) for each vector
            vectors = x_flat[i]
            norms = (vectors * vectors).sum(dim=1)

            # Initialize indices for sampled points and min distances
            idx = torch.zeros(m, dtype=torch.long, device=x.device)
            dist = torch.full((n,), float('inf'), device=x.device)

            for j in range(1, m):
                # Update min distances after adding a new vector
                # 1. Compute the distance from all points to the newly added point
                # 2. Update the min distance
                last_idx = idx[j - 1]
                y = vectors[last_idx]

                # (x - y)^2 = x^2 + y^2 - 2xy
                y_norm = norms[last_idx]
                dot = vectors @ y

                d = norms + y_norm - 2 * dot

                dist = torch.minimum(dist, d)

                # Select the next point as the farthest point
                idx[j] = torch.argmax(dist)

            outs.append(vectors[idx])
        out = torch.stack(outs, dim=0)  # (B, M, C)
        return self._reshape_hw(out, b, c, h, w, self.r)
