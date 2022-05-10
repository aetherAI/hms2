import abc
import concurrent.futures
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from .custom_modules import BaseAugmentorModule


class BaseLoaderModule(nn.Module, metaclass=abc.ABCMeta):
    """
    An abstract loader module, handling region reading, augmentation, and CPU-GPU
    transfer. It accepts a torch.Tensor with NHWC format and uint8 type as an input
    image batch. The module returns a torch.Tensor on CUDA with the FP32 type and NCHW
    shape. For image augmentation, all the random variables should be kept as data
    members and get re-randomized upon `randomize` is called.
    """

    @abc.abstractmethod
    def forward(
        self,
        image_batch: torch.Tensor,
        coord: Tuple[int, int],
        size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Define the forward function to read a region from an image.

        Args:
            image_batch (torch.Tensor): A tensor with NHWC format and uint8 type.
            coord (Tuple[int, int]): A 2-element tuple defining (x, y).
            size (Tuple[int, int]): A 2-element tuple defining (width, height).

        Returns:
            region (torch.Tensor): A tensor in NCHW and FP32.
        """

    @abc.abstractmethod
    def randomize(self) -> None:
        """
        Randomize all the variables for augmentations.
        """

    def hint_future_accesses(
        self,
        image_batch: torch.Tensor,
        coords: Sequence[Tuple[int, int]],
        sizes: Sequence[Tuple[int, int]],
    ) -> None:
        """
        Hint the loader module the requesting regions of future accesses and their
        order. The order will be (image_batch[0], coords[0], sizes[0]) ->
        (image_batch[1], coords[0], sizes[0]) -> ... -> (image_batch[N - 1],
        coords[0], sizes[0]) -> (image_batch[0], coords[1], sizes[1]) -> ...

        Args:
            image_batch (torch.Tensor):
                The format is defined in each derived class.
            coords (list):
                A list of tuples, each of which is a 2-element tuple defining (x, y).
            sizes (list):
                A list of tuples, each of which is a 2-element tuple definiing (w, h).
        """

    def prefetch_next(self) -> None:
        """
        Available when `hint_future_accesses` is called. Let the loader module to
        prefetch the next region. This method should be called before the next region
        is requested by `forward`, or an error will be raised.
        """

    @abc.abstractmethod
    def record_snapshot(self) -> None:
        """
        Start recording the snapshot for debugging.
        """

    @abc.abstractmethod
    def get_snapshot(self) -> np.ndarray:
        """
        Stop recording the snapshot and return it.

        Returns:
            snapshots:
                A batch of snapshots with the shape [B, H, W, 3], RGB uint8 format.
        """


class PlainLoaderModule(BaseLoaderModule):
    """
    A plain loader module that simply does region reading, CPU-GPU data transfer, and
    normalization. If needed, augmentation should be done before an image is fed into
    this module.
    """

    def __init__(self):
        super().__init__()

        self.register_buffer("device_indicator", torch.empty(0))

        self.prefetch_idx = 0
        self.prefetch_image_batch = None
        self.prefetch_coords = None
        self.prefetch_sizes = None
        self.prefetch_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.prefetch_task = None

        self.snapshot_enabled = False
        self.snapshots = []

    def forward(
        self,
        image_batch: torch.Tensor,
        coord: Tuple[int, int],
        size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        See the description in `BaseLoaderModule.forward`.
        """
        if self.prefetch_task is not None:
            # If hint_future_accesses was called.
            if (
                coord != self.prefetch_coords[self.prefetch_idx]
                or size != self.prefetch_sizes[self.prefetch_idx]
                or image_batch is not self.prefetch_image_batch
            ):
                raise ValueError(
                    "The arguments of hint_future_accesses does not consist with those"
                    " of forward."
                )
            patch = self.prefetch_task.result()

            self.prefetch_next()
        else:
            # If hint_future_accesses was not called.
            patch = self._read_region(
                image_batch,
                coord,
                size,
                self.device_indicator.device,
            )

        return patch

    def randomize(self) -> None:
        """
        Do nothing since no random variable exist in this loader module.
        """

    def hint_future_accesses(
        self,
        image_batch: torch.Tensor,
        coords: Sequence[Tuple[int, int]],
        sizes: Sequence[Tuple[int, int]],
    ) -> None:
        """
        See the description in `BaseLoaderModule.hint_future_accesses`.
        """
        self.prefetch_idx = -1
        self.prefetch_image_batch = image_batch
        self.prefetch_coords = coords
        self.prefetch_sizes = sizes

        self.prefetch_next()

    def prefetch_next(self) -> None:
        """
        See the description in `BaseLoaderModule.prefetch_next`.
        """
        self.prefetch_idx += 1
        if self.prefetch_idx < len(self.prefetch_coords):
            next_coord = self.prefetch_coords[self.prefetch_idx]
            next_size = self.prefetch_sizes[self.prefetch_idx]

            if self.prefetch_task is not None:
                self.prefetch_task.cancel()
            self.prefetch_task = self.prefetch_thread_pool.submit(
                self._read_region,
                self.prefetch_image_batch,
                next_coord,
                next_size,
                self.device_indicator.device,
            )
        else:
            self.prefetch_task = None

    def record_snapshot(self) -> None:
        """
        See the description in `BaseLoaderModule.record_snapshot`.
        """
        self.snapshot_enabled = True

    def get_snapshot(self) -> np.ndarray:
        """
        See the description in `BaseLoaderModule.get_snapshot`.
        """
        width = max(
            [snapshot["coord"][0] + snapshot["size"][0] for snapshot in self.snapshots]
        )
        height = max(
            [snapshot["coord"][1] + snapshot["size"][1] for snapshot in self.snapshots]
        )

        batch_size = 0
        for snapshot in self.snapshots:
            batch_size_this = int(snapshot["patch_batch"].shape[0])
            if batch_size == 0:
                batch_size = batch_size_this
            elif batch_size != batch_size_this:
                raise RuntimeError("Batch sizes are not consistent in the snapshots.")

        canvases = []
        for idx in range(batch_size):
            canvas = Image.new(
                mode="RGB",
                size=(width, height),
                color=(0, 255, 0),
            )
            for snapshot in self.snapshots:
                patch = Image.fromarray(snapshot["patch_batch"][idx])
                box = (
                    snapshot["coord"][0],
                    snapshot["coord"][1],
                    snapshot["coord"][0] + snapshot["size"][0],
                    snapshot["coord"][1] + snapshot["size"][1],
                )
                canvas.paste(patch, box=box)
            canvas = np.array(canvas)
            canvases.append(canvas)
        canvases_array = np.array(canvases)

        self.snapshot_enabled = False
        self.snapshots = []

        return canvases_array

    def __del__(self) -> None:
        self.prefetch_thread_pool.shutdown()

    @torch.no_grad()
    def _read_region(
        self,
        image_batch: torch.Tensor,
        coord: Tuple[int, int],
        size: Tuple[int, int],
        device: Union[torch.device, str, None],
    ) -> torch.Tensor:
        patch = image_batch[
            :,
            coord[1] : coord[1] + size[1],
            coord[0] : coord[0] + size[0],
            :,
        ]
        patch = patch.to(device=device)  # To GPU
        patch = patch.permute(0, 3, 1, 2).contiguous()  # To NCHW
        patch = patch.float().div(255.0)  # To FP32

        if self.snapshot_enabled:
            patch_snapshot = (
                (patch * 255.0)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
                .cpu()
                .numpy()
            )
            self.snapshots.append(
                {
                    "coord": coord,
                    "size": size,
                    "patch_batch": patch_snapshot,
                }
            )

        patch = transforms.functional.normalize(
            tensor=patch,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return patch


class GPUAugmentationLoaderModule(PlainLoaderModule):
    """
    A loader module that does region reading, CPU-GPU data transfer, on-GPU
    augmentation, and normalization.

    Args:
        random_rotation (bool): Enable random rotation. Default is True.
        random_translation (tuple or NoneType):
            A tuple with 2 elements (x, y). Set None to disalbe the augmentation.
            Default is (-32.0, 32.0).
        random_flip (bool): Enable random flipping. Default is True.
        other_augmentations (sequence or NoneType):
            A list of torch.nn.Module. Each consumes a torch.Tensor as an input image
            batch with NCHW, FP32, [0.0, 1.0] formats, and produces a torch.Tensor
            with the same shape and format. Both tensors are on GPU. These modeuls
            should implement a randomize() method. Set None (default) to indicate no
            further augmentations to apply.
        skip_background_tile_aug (bool):
            Skip unnecessary augmentations, including rotation, translation, and
            flipping, for background tiles.
    """

    def __init__(
        self,
        random_rotation: bool = True,
        random_translation: Optional[Tuple[float, float]] = (-32.0, 32.0),
        random_flip: bool = True,
        other_augmentations: Optional[Sequence[BaseAugmentorModule]] = None,
        skip_background_tile_aug: bool = True,
    ):
        super().__init__()
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.random_flip = random_flip
        self.other_augmentations = (
            other_augmentations if other_augmentations is not None else []
        )
        self.skip_background_tile_aug = skip_background_tile_aug

        self.rotation_angle = 0.0
        self.translation_pixels = np.zeros(shape=[2])
        self.do_flip = False
        self.affine_matrix = np.identity(3)

    def randomize(self) -> None:
        if self.random_rotation:
            self.rotation_angle = np.random.uniform(-180.0, 180.0)
        if self.random_translation:
            self.translation_pixels = np.random.uniform(
                self.random_translation[0],
                self.random_translation[1],
                size=(2,),
            )
        if self.random_flip:
            self.do_flip = np.random.rand() > 0.5
        for module in self.other_augmentations:
            module.randomize()

        self.affine_matrix = self._calculate_affine_matrix()

    def _calculate_affine_matrix(self) -> np.ndarray:
        """
        The order of the augmentations is rotate -> translate -> flip.
        Calculating affine_matrix should be in the reverse way.
        """
        affine_matrix = np.identity(3)

        if self.do_flip:
            transform = np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            )
            affine_matrix = np.matmul(transform, affine_matrix)

        if self.random_translation:
            transform = np.array(
                [
                    [1.0, 0.0, -self.translation_pixels[0]],
                    [0.0, 1.0, -self.translation_pixels[1]],
                    [0.0, 0.0, 1.0],
                ]
            )
            affine_matrix = np.matmul(transform, affine_matrix)

        if self.random_rotation:
            angle_in_rad = np.radians(self.rotation_angle)
            transform = np.array(
                [
                    [np.cos(angle_in_rad), -np.sin(angle_in_rad), 0.0],
                    [np.sin(angle_in_rad), np.cos(angle_in_rad), 0.0],
                    [0.0, 0.0, 1.0],
                ],
            )
            affine_matrix = np.matmul(transform, affine_matrix)

        return affine_matrix

    @torch.no_grad()
    def _read_region(
        self,
        image_batch: torch.Tensor,
        coord: Tuple[int, int],
        size: Tuple[int, int],
        device: Union[torch.device, str, None],
    ):
        # If in evaluation mode, disable the augmentations.
        if not self.training:
            return super()._read_region(image_batch, coord, size, device)

        # Calculate the region center regarding the image center
        batch_size, height, width, channels = image_batch.shape
        coord = np.array(coord)
        size = np.array(size)
        image_center = np.array([width, height]) / 2.0
        region_center = coord + size / 2.0
        norm_region_center = region_center - image_center

        # Get the new region center after an affine transformation
        new_norm_region_center = np.matmul(
            self.affine_matrix,
            np.array([norm_region_center[0], norm_region_center[1], 1.0]),
        )[:2]
        new_region_center = new_norm_region_center + image_center

        # Calculate the coordinates of the region to read
        min_read_size = np.max(size * np.sqrt(2.0))
        read_l = np.floor(new_region_center[0] - min_read_size / 2.0).astype(np.int32)
        read_r = np.ceil(new_region_center[0] + min_read_size / 2.0).astype(np.int32)
        read_t = np.floor(new_region_center[1] - min_read_size / 2.0).astype(np.int32)
        read_b = np.ceil(new_region_center[1] + min_read_size / 2.0).astype(np.int32)
        new_region_center_patch = new_region_center - np.array([read_l, read_t])

        is_background_tile = None
        if read_l > width or read_r < 0 or read_t > height or read_b < 0:
            # When the reading region is totally out-of-range, just create a blank
            # tesnor on GPU.
            is_background_tile = True
            patch = torch.full(
                size=(
                    batch_size,
                    channels,
                    read_b - read_t,
                    read_r - read_l,
                ),
                fill_value=1.0,
                device=device,
            )
        else:
            # When the reading region is contentful or partially out-of-range, crop
            # valid region, send it to GPU, and pad blank.

            # Deal with partially out-of-range issue
            patch_width = read_r - read_l
            patch_height = read_b - read_t
            pad_l = np.maximum(0, -read_l)
            read_l = np.maximum(0, read_l)
            pad_r = np.maximum(0, read_r - width)
            read_r = np.minimum(width, read_r)
            pad_t = np.maximum(0, -read_t)
            read_t = np.maximum(0, read_t)
            pad_b = np.maximum(0, read_b - height)
            read_b = np.minimum(height, read_b)

            # Read the region
            patch = image_batch[
                :,
                read_t:read_b,
                read_l:read_r,
                :,
            ]

            # Determine if patch is blank.
            if patch.nelement() == 0 or torch.min(patch) == 255:
                # If so, just create a blank tesnor on GPU.
                is_background_tile = True
                patch = torch.full(
                    size=(
                        batch_size,
                        channels,
                        patch_height,
                        patch_width,
                    ),
                    fill_value=1.0,
                    device=device,
                )
            else:
                # Send the patch to GPU.
                is_background_tile = False
                patch = patch.to(device=device)  # To GPU
                patch = patch.permute(0, 3, 1, 2).contiguous()  # To NCHW
                patch = patch.float().div(255.0)  # To FP32

                # Pad white color for out-of-range region reading
                patch = nn.functional.pad(
                    patch,
                    pad=(pad_l, pad_r, pad_t, pad_b),
                    mode="constant",
                    value=1.0,
                )

        # Rotate the patch if needed
        if self.random_rotation and not is_background_tile:
            patch = transforms.functional.rotate(
                img=patch,
                angle=self.rotation_angle,
                interpolation=transforms.InterpolationMode.BILINEAR,
                center=list(new_region_center_patch),
                fill=1.0,
            )

        # Translate the patch
        if not is_background_tile:
            translate = size / 2.0 - new_region_center_patch
            patch = transforms.functional.affine(
                patch,
                angle=0.0,
                translate=list(translate),
                scale=1.0,
                shear=0.0,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=1.0,
            )

        # Crop out the real region
        patch = patch[:, :, : size[1], : size[0]]

        # Flip the patch if needed
        if self.do_flip and not is_background_tile:
            patch = torch.flip(patch, dims=(3,))

        # Apply other augmentations
        for module in self.other_augmentations:
            patch = module(patch)

        if self.snapshot_enabled:
            patch_snapshot = (
                (patch * 255.0)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
                .cpu()
                .numpy()
            )
            self.snapshots.append(
                {
                    "coord": coord,
                    "size": size,
                    "patch_batch": patch_snapshot,
                }
            )

        # Normalize the patch
        patch = transforms.functional.normalize(
            tensor=patch,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return patch


class NoLoaderModule(BaseLoaderModule):
    """
    A loader module that simply does normalization.
    """

    def __init__(self, augmentations: Optional[Sequence[nn.Module]] = None):
        super().__init__()

        self.augmentations = nn.ModuleList(
            augmentations if augmentations is not None else []
        )

        self.register_buffer("device_indicator", torch.empty(0))

    def randomize(self):
        """
        Do nothing since no random variable exist in this loader module.
        """

    @torch.no_grad()
    def forward(
        self,
        image_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        See the description in `BaseLoaderModule.forward`.
        """
        # Do augmentations
        for augmentation in self.augmentations:
            image_batch = augmentation(image_batch)

        # Do normalization
        image_batch = image_batch.permute(0, 3, 1, 2).contiguous()  # To NCHW
        image_batch = image_batch.float().div(255.0)  # To FP32
        image_batch = image_batch - torch.tensor(
            [0.485, 0.456, 0.406],
            device=image_batch.device,
        ).view(-1, 1, 1)
        image_batch = image_batch / torch.tensor(
            [0.229, 0.224, 0.225],
            device=image_batch.device,
        ).view(-1, 1, 1)
        return image_batch

    def record_snapshot(self) -> None:
        """
        See the description in `BaseLoaderModule.record_snapshot`.
        """
        raise NotImplementedError()

    def get_snapshot(self) -> np.ndarray:
        """
        See the description in `BaseLoaderModule.get_snapshot`.
        """
        raise NotImplementedError()
