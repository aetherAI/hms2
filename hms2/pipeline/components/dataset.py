"""
This module defines the dataset.
"""

from __future__ import annotations

import bisect
import csv
import logging
import math
import os
import random
import time
import traceback
import typing as t
from pathlib import Path

import affine
import cv2
import numpy as np
import numpy.typing as npt
import pydantic
import rasterio.features
import shapely
import torch
import torch.utils.data
import torch.utils.data.distributed

from .utils import HEDPerturbAugmentor, find_optimal_scale

from .official_openslide import BoundingBox, open_slide


class ResizeRatioByPixelSpacing(pydantic.BaseModel):
    target_pixel_spacing: float
    loose_pixel_spacing: bool = True


class ClassWeight(pydantic.BaseModel):
    class_index: int
    positivity: bool
    weight: float


class Dataset(torch.utils.data.Dataset):
    """
    A dataset for HMS2.

    Args:
        csv_path (str):
            The path to the CSV file recording the slide names and the labels.
        slide_dir (str): The directory path that stores slide data.
        slide_file_extension (str):
            The extension suffix of slide data.
        contour_dir (str or NoneType):
            The directory path that stores contour data. If given (a.k.a. not None),
            the program will look for the contour json files first by the contour name
            provided in the CSV file. Then, the program will get the slide name by
            parsing the json file. If this argument is None, the names in the CSV file
            will directly correspond to slides in slide_dir.
        format (one of 'safe_rotate', 'trim', 'fit'):
            This determines the shape of output images. 'fit' will only apply
            resizing and background removal to the input image. 'trim' will
            additionally trim background margins. 'safe_rotate' will pad the trimmed
            image with white color to make it friendly to rotation augmentation.
        resize_ratio (float):
            The ratio between the target size and the original size.
        snapshot_path (str or NoneType):
            The path to store loaded images for debugging. The default value, None,
            disables the snapshot storing.
        augment_list (list or NoneType):
            A list of strings specifying which augmentations should be performed. The
            tokens include 'flip', 'rigid', 'hed_perturb'. Set None to disable
            augmentations.
        default_pixel_spacing (float):
            The default pixel spacing to use if the slide does not contain pixel
            spacing information. Default to 0.25.
        embedding_dir (str or NoneType):
            The directory path that stores embedding data. If given (a.k.a. not None),
            the dataset will skip the slides that are already in the embedding directory.
            This argument is used to avoid re-extracting the embeddings, and should be set
            only when the dataset is used for embedding extraction.
    """

    def __init__(
        self,
        csv_path: str,
        slide_dir: str,
        slide_file_extension: str,
        contour_dir: t.Optional[str] = None,
        format: t.Union[t.Literal['fit'], t.Literal['trim'], t.Literal['safe_rotate']] = 'fit',
        resize_ratio: t.Union[float, ResizeRatioByPixelSpacing] = 1.0,
        snapshot_path: t.Optional[str] = None,
        augment_list: t.Optional[t.Sequence[str]] = None,
        default_pixel_spacing: float = 0.25,
        embedding_dir: t.Optional[str] = None,
    ):
        self.format = format
        self.resize_ratio = resize_ratio
        self.snapshot_path = snapshot_path
        self.augment_list = augment_list if (augment_list is not None) else []
        self.default_pixel_spacing = default_pixel_spacing

        # Parse CSV to get paths
        self.slide_name_list = []
        self.slide_path_list = []
        self.contours_list = []
        self.y_true_list = []

        embedding_slide_names = set()
        if embedding_dir is not None:
            for file in os.listdir(embedding_dir):
                embedding_slide_names.add(file.rsplit('.', 1)[0])

        # Read the CSV for slide_name, slide_path, contours (optional), and y_true.
        with open(csv_path) as file:
            reader = csv.reader(file)
            for row in reader:
                slide_name = row[0]
                if embedding_dir is not None and slide_name in embedding_slide_names:
                    # Skip the slide if it is already in the embedding directory.
                    continue

                if len(row[1:]) == 1:
                    y_true = int(row[1])
                else:
                    y_true = np.array(
                        [np.nan if label == 'X' else float(label) for label in row[1:]],
                        dtype=np.float32,
                    )

                if contour_dir is None:
                    slide_path = Path(slide_dir) / f'{slide_name}{slide_file_extension}'
                    if not os.path.exists(slide_path):
                        raise FileNotFoundError(f'{slide_path} not found while parsing {csv_path}.')
                    contours = None
                else:
                    contour_json_path = os.path.join(contour_dir, f'{slide_name}.json')
                    if not os.path.exists(contour_json_path):
                        raise FileNotFoundError(f'{contour_json_path} not found while parsing {csv_path}.')
                    contours_descriptor = ContoursDescriptor.parse_file(
                        contour_json_path,
                    )
                    raw_slide_name = contours_descriptor.slide_name
                    contours = contours_descriptor.contours
                    slide_path = Path(slide_dir) / f'{raw_slide_name}{slide_file_extension}'
                    if not os.path.exists(slide_path):
                        raise FileNotFoundError(f'{slide_path} not found while parsing {contour_json_path}.')

                self.slide_name_list.append(slide_name)
                self.slide_path_list.append(slide_path)
                self.contours_list.append(contours)
                self.y_true_list.append(y_true)

    def __len__(self) -> int:
        """
        The number of slides.

        Returns:
            length (int): The number.
        """
        return len(self.slide_path_list)

    def __getitem__(self, idx: int) -> t.Tuple[npt.NDArray[np.uint8], t.Union[int, npt.NDArray[np.int_]]]:
        """
        Get the pair of a slide image and a label with an index.

        Args:
            idx (int): The index.

        Returns:
            img (numpy.ndarray): The RGB image in HWC, uint8 format.
            y_true (t.Union[int, numpy.ndarray]):
                The label of the slide. If only one number is presented, an integer is
                returned. Otherwise, a ndarray is returned with the shape [C].
        """
        slide_path = self.slide_path_list[idx]
        contours = self.contours_list[idx]
        y_true = self.y_true_list[idx]

        slide = open_slide(slide_path)
        slide.max_region_nbytes = np.inf
        resize_ratio = self.get_resize_ratio(idx)

        img = None
        while img is None:
            try:
                img = _read_region(
                    slide,
                    resize_ratio=resize_ratio,
                    contours=contours,
                    format=self.format,
                )

            except RuntimeError:
                print(f'Error raises while reading {slide_path}. Trying reading level-0 data.')
                try:
                    img = _read_region(
                        slide,
                        resize_ratio=1.0,
                        contours=contours,
                        format=self.format,
                    )
                    img = cv2.resize(
                        img,
                        (int(img.shape[1] * resize_ratio), int(img.shape[0] * resize_ratio)),
                        interpolation=cv2.INTER_AREA,
                    )
                except RuntimeError:
                    print(traceback.format_exc())
                    print(f'Error raises while reading {slide_path}. Retry in 5 seconds.')
                    time.sleep(5)

        for augment in self.augment_list:
            if augment == 'flip':
                do_flip = np.random.choice([True, False])
                if do_flip:
                    img = cv2.flip(img, flipCode=1)
            elif augment == 'rigid':
                angle = np.random.uniform(-180.0, 180.0)
                shift = np.random.uniform(-32.0, 32.0, size=(2,))
                matrix = cv2.getRotationMatrix2D(
                    center=(img.shape[1] / 2.0, img.shape[0] / 2.0),
                    angle=angle,
                    scale=1.0,
                )
                matrix[:, 2] += shift
                img = cv2.warpAffine(
                    img,
                    matrix,
                    (img.shape[1], img.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderValue=(255, 255, 255),
                )
            elif augment == 'hed_perturb':
                img = HEDPerturbAugmentor()(img)

        if self.snapshot_path is not None:
            os.makedirs(self.snapshot_path, exist_ok=True)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(self.snapshot_path, 'dataset_snapshot.tiff'),
                img_bgr,
            )

        return img, y_true

    def get_slide_path(self, idx: int) -> str:
        """
        Get the file path of a slide with an index.

        Args:
            idx (int): The index.

        Returns:
            path (str): The path to the slide file.
        """
        return self.slide_path_list[idx]

    def get_slide_name(self, idx: int) -> str:
        """
        Get the slide name given an index.

        Args:
            idx (int): The index.

        Returns:
            slide_name (str): The slide name.
        """
        return self.slide_name_list[idx]

    def get_y_true(self, idx: int) -> t.Union[int, np.ndarray]:
        """
        Get the ground truth of a slide with an index.

        Args:
            idx (int): The index.

        Returns:
            y_true (int): The ground truth.
        """
        return self.y_true_list[idx]

    def get_contours(self, idx: int) -> t.Optional[t.List[t.List[t.Tuple[int, int]]]]:
        """
        Get the contours of a slide with an index.

        Args:
            idx (int): The index.

        Returns:
            contours: The contours of the slide.
        """
        return self.contours_list[idx]

    def get_resize_ratio(self, idx: int) -> float:
        """
        Get the resize ratio of a slide with an index.

        Args:
            idx (int): The index.

        Returns:
            resize_ratio: The resize ratio.
        """
        slide = open_slide(self.slide_path_list[idx])
        if isinstance(self.resize_ratio, float):
            resize_ratio = self.resize_ratio
        elif isinstance(self.resize_ratio, ResizeRatioByPixelSpacing):
            if slide.metadata.pixel_width_nm is None:
                source_pixel_dim = self.default_pixel_spacing
                print(
                    'Warning: The slide contains no pixel spacing infomation: '
                    f'{self.slide_name_list[idx]}. Set it as default, {self.default_pixel_spacing}.',
                )
            else:
                source_pixel_dim = slide.metadata.pixel_width_nm / 1000.0
            resize_ratio = find_optimal_scale(
                source_pixel_dim=source_pixel_dim,
                target_pixel_dim=self.resize_ratio.target_pixel_spacing,
                eps=(np.inf if self.resize_ratio.loose_pixel_spacing else None),
            )
        else:
            raise ValueError(f'Unsupported type of resize_ratio: {self.resize_ratio}.')

        return resize_ratio


class EmbeddingDataset(Dataset):
    """
    A dataset for loading extracted embeddings.
    Args:
        csv_path (str):
            The path to the CSV file recording the slide names and the labels.
        slide_dir (str):
            The directory path that stores slide data.
        slide_file_extension (str):
            The extension suffix of slide data.
        embed_dir (str):
            The directory path that stores embedding data.
        saver:
            The saver object for loading embeddings.
    """

    def __init__(
        self,
        csv_path: str,
        slide_dir: str,
        slide_file_extension: str,
        embed_dir: str,
        saver,
        **kwargs,
    ):
        super().__init__(csv_path=csv_path, slide_dir=slide_dir, slide_file_extension=slide_file_extension, **kwargs)
        self.embed_dir = embed_dir
        self.saver = saver

        # Check if the embedding directory exists
        if not os.path.exists(self.embed_dir):
            raise FileNotFoundError(
                f'Embedding directory {self.embed_dir} does not exist, please extract the embeddings first.',
            )

        # Check all embeddings exist in the embedding directory
        for slide_name in self.slide_name_list:
            embed_path = os.path.join(self.embed_dir, f'{slide_name}')
            if not saver.exists(embed_path):
                raise FileNotFoundError(
                    f'Embedding file {embed_path} does not exist, please ensure all embeddings are extracted.',
                )

    def __getitem__(self, idx: int) -> t.Tuple[t.Dict[torch.Tensor], t.Union[int, npt.NDArray[np.int_]]]:
        slide_name = self.slide_name_list[idx]
        y_true = self.y_true_list[idx]
        embed = self.saver.load(os.path.join(self.embed_dir, f'{slide_name}'))
        return embed, y_true


class ConcatDataset(torch.utils.data.Dataset):
    """Concatednated dataset

    Args:
        datasets (t.Sequence[Dataset]): A list of datasets.
    """

    def __init__(
        self,
        datasets: t.Sequence[Dataset],
    ):
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(dataset) for dataset in datasets])

    def _get_ori_dataset_idx(self, idx: int) -> t.Tuple[int, int]:
        """Convert global index to local index.

        Args:
            idx: Global index of ``ConcatDataset``.

        Returns:
            t.Tuple[int, int]: The index of ``self.datasets`` and the local
            index of data.
        """
        # Get `dataset_idx` to tell idx belongs to which dataset.
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        # Get the inner index of single dataset.
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return dataset_idx, sample_idx

    def __len__(self) -> int:
        """
        The number of slides.

        Returns:
            length (int): The number.
        """
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> t.Tuple[npt.NDArray[np.uint8], t.Union[int, npt.NDArray[np.int_]]]:
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx][sample_idx]

    def get_slide_path(self, idx: int) -> str:
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_slide_path(sample_idx)

    def get_slide_name(self, idx: int) -> str:
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_slide_name(sample_idx)

    def get_y_true(self, idx: int) -> t.Union[int, np.ndarray]:
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_y_true(sample_idx)

    def get_contours(self, idx: int) -> t.Optional[t.List[t.List[t.Tuple[int, int]]]]:
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_contours(sample_idx)

    def get_resize_ratio(self, idx: int) -> float:
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_resize_ratio(sample_idx)


class ContoursDescriptor(pydantic.BaseModel):
    slide_name: str
    contours: t.List[t.List[t.Tuple[int, int]]]


class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):
    dataset: Dataset

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: t.Optional[int] = None,
        rank: t.Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        class_weights: t.Optional[t.List[ClassWeight]] = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.class_weights = class_weights
        self.num_samples_per_class_per_replica_list = self._calculate_num_samples_per_class_per_replica_list()

    def __iter__(self) -> t.Iterator[t.Any]:
        iter_without_class_weights = super().__iter__()
        if self.class_weights is None or self.num_samples_per_class_per_replica_list is None:
            return iter_without_class_weights

        list_without_class_weights = list(iter_without_class_weights)
        list_with_class_weights = []
        for class_index, class_weight in enumerate(self.class_weights):
            candidate_dataset_indices = []
            for dataset_index in list_without_class_weights:
                y_true = self.dataset.get_y_true(dataset_index)
                if self._is_class_met(y_true, class_weight):
                    candidate_dataset_indices.append(dataset_index)

            num_samples_per_class_per_replica = self.num_samples_per_class_per_replica_list[class_index]
            for index in range(num_samples_per_class_per_replica):
                candidate_index = index % len(candidate_dataset_indices)
                candidate = candidate_dataset_indices[candidate_index]
                list_with_class_weights.append(candidate)

        random.shuffle(list_with_class_weights)
        return iter(list_with_class_weights)

    def __len__(self) -> int:
        if self.num_samples_per_class_per_replica_list is not None:
            return sum(self.num_samples_per_class_per_replica_list)
        else:
            return self.num_samples

    def _calculate_num_samples_per_class_per_replica_list(self) -> t.Optional[t.List[int]]:
        if self.class_weights is None:
            return None

        num_samples_per_class_per_replica_list: t.List[int] = []
        for class_weight in self.class_weights:
            num_samples_per_class = 0
            for dataset_index in range(len(self.dataset)):
                y_true = self.dataset.get_y_true(dataset_index)
                if self._is_class_met(y_true, class_weight):
                    num_samples_per_class += 1
            num_samples_per_class_per_replica = math.ceil(
                num_samples_per_class * class_weight.weight / self.num_replicas,
            )
            num_samples_per_class_per_replica_list.append(num_samples_per_class_per_replica)

        return num_samples_per_class_per_replica_list

    def _is_class_met(self, y_true: t.Union[int, np.ndarray], class_weight: ClassWeight) -> bool:
        if self.class_weights is None:
            raise ValueError

        is_class_met = False
        if isinstance(y_true, int):
            if class_weight.positivity:
                if y_true == class_weight.class_index:
                    is_class_met = True
            else:
                if y_true != class_weight.class_index:
                    is_class_met = True
        elif isinstance(y_true, np.ndarray):
            if class_weight.positivity:
                if y_true[class_weight.class_index] == 1:
                    is_class_met = True
            else:
                if y_true[class_weight.class_index] == 0:
                    is_class_met = True
        else:
            raise ValueError

        return is_class_met


def _read_region(
    slide_reader: t.Any,
    resize_ratio: float,
    pad_color: t.Tuple[int, int, int] = (255, 255, 255),
    contours: t.Optional[t.Sequence[t.Sequence[t.Union[int, float]]]] = None,
    format: t.Union[t.Literal['fit'], t.Literal['trim'], t.Literal['safe_rotate']] = 'fit',
) -> np.ndarray:
    """
    Read a region of a slide if `contours` is given. Otherwise, read the entire slide.

    Args:
        slide_reader:
            A slide reader object.
        resize_ratio:
            The resizing ratio, typically <= 1 to shrink the image.
        pad_color:
            The color of padding in (R, G, B). Default to `(255, 255, 255)`.
        format (one of 'safe_rotate', 'trim', 'fit'):
            This determines the shape of output images. 'fit' will only apply
            resizing and background removal to the input image. 'trim' will
            additionally trim background margins. 'safe_rotate' will pad the trimmed
            image with white color to make it friendly to rotation augmentation.

    Returns:
        image (np.ndarray):
            The read region with the shape as `(output_size[1], output_size[0], 3)`
            and the dtype as np.uint8.
    """
    # Return a blank region given that `contours` contains no contours.
    if contours is not None and len(contours) == 0:
        shape = (
            (math.ceil(slide_reader.height * resize_ratio), math.ceil(slide_reader.width * resize_ratio), 3)
            if format == 'fit'
            else (512, 512, 3)
        )
        return np.full(
            shape=shape,
            fill_value=pad_color,
            dtype=np.uint8,
        )

    # Calculate the effective area
    contours_np = None
    if contours is not None:
        contours_np = [np.array(contour) for contour in contours]

    if contours_np is None:
        left = 0
        top = 0
        input_width = slide_reader.width
        input_height = slide_reader.height
    else:
        coordinates = []
        for contour in contours_np:
            coordinates.append(contour.reshape((-1, 2)))
        coordinates = np.concatenate(coordinates)

        left = max(math.floor(coordinates[:, 0].min()), 0)
        top = max(math.floor(coordinates[:, 1].min()), 0)
        right = min(math.ceil(coordinates[:, 0].max()), slide_reader.width)
        bottom = min(math.ceil(coordinates[:, 1].max()), slide_reader.height)
        input_width = right - left
        input_height = bottom - top

    # Read the effective area
    box = BoundingBox(left=left, top=top, width=input_width, height=input_height)
    effective_image = slide_reader.get_region(
        box,
        scale=resize_ratio,
        padding=True,
    )
    effective_width = effective_image.shape[1]
    effective_height = effective_image.shape[0]

    # Crop contours
    if contours_np is not None:
        effective_mask = rasterio.features.rasterize(
            [shapely.MultiPolygon(polygons=[shapely.Polygon(shell=contour) for contour in contours_np])],  # noqa: S604
            out_shape=(effective_height, effective_width),
            transform=(affine.Affine.translation(xoff=left, yoff=top) * affine.Affine.scale(1.0 / resize_ratio)),
            all_touched=True,
            dtype=np.uint8,
        ).astype(bool)
        effective_image = np.where(
            effective_mask[:, :, np.newaxis],
            effective_image,
            np.array(pad_color, dtype=np.uint8)[np.newaxis, np.newaxis, :],
        )
    else:
        effective_mask = np.ones((effective_height, effective_width), dtype=bool)

    # Calculate the output size
    if format == 'fit':
        output_width = math.ceil(slide_reader.width * resize_ratio)
        output_height = math.ceil(slide_reader.height * resize_ratio)
    elif format == 'trim':
        output_width = effective_width
        output_height = effective_height
    elif format == 'safe_rotate':
        diagnal_length = math.ceil((effective_width**2.0 + effective_height**2.0) ** 0.5)
        output_width = diagnal_length
        output_height = diagnal_length
    else:
        raise ValueError

    # Do padding
    if format == 'fit':
        padding_left = int(left * resize_ratio)
        padding_top = int(top * resize_ratio)
        padding_right = output_width - effective_image.shape[1] - padding_left
        padding_bottom = output_height - effective_image.shape[0] - padding_top
        image = cv2.copyMakeBorder(
            effective_image,
            padding_top,
            padding_bottom,
            padding_left,
            padding_right,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )
    elif format == 'trim':
        image = effective_image
    elif format == 'safe_rotate':
        padding_left = (output_width - effective_image.shape[1]) // 2
        padding_top = (output_height - effective_image.shape[0]) // 2
        padding_right = output_width - effective_image.shape[1] - padding_left
        padding_bottom = output_height - effective_image.shape[0] - padding_top
        image = cv2.copyMakeBorder(
            effective_image,
            padding_top,
            padding_bottom,
            padding_left,
            padding_right,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )

    else:
        raise ValueError

    return image
