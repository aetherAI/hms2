from __future__ import annotations

import argparse
import math
import typing as t
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.lib.stride_tricks
import numpy.typing as npt
import pydantic
from PIL import Image
from tqdm import tqdm

from ..components.config import TestConfig
from ..components.dataset import ConcatDataset, Dataset


class RepresentativePatchMetadata(pydantic.BaseModel):
    output_path: Path
    class_idx: int
    rank_idx: int
    score: float
    slide_name: str
    slide_path: Path
    viewport_coordinate: t.Tuple[int, int]  # (left, top) on the level-0 slide image
    viewport_size: t.Tuple[int, int]  # (width, height) on the level-0 slide image


class RepresentativePatchesMetadata(pydantic.BaseModel):
    __root__: t.List[RepresentativePatchMetadata]


def main() -> None:
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Config file.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help=('Folder to store the outputs. default: ' '${RESULT_DIR}/representative_patches .'),
    )
    parser.add_argument(
        '--output-metadata',
        type=Path,
        default=None,
        help='Path to store the metadata output. default: ${RESULT_DIR}/representative_patches_metadata.json',
    )
    parser.add_argument(
        '--nms',
        type=_positive_odd_number,
        default=17,
        help=(
            'The kernel size of non max suppresion to avoid sampling overlapped '
            'patches. Should be positive and odd. Set 1 to disable NMS.'
        ),
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=30,
        help='The number of representative patches to extract per class.',
    )
    parser.add_argument(
        '--size',
        type=int,
        default=256,
        help='The image size of the output representative patches.',
    )
    parser.add_argument(
        '--output_filename_template',
        type=str,
        default='class{class_idx}_rank{rank_idx}.png',
        help='The template of output filenames.',
    )
    args = parser.parse_args()

    # Read the config
    config = TestConfig.from_yaml(Path(args.config))

    # Set defaults
    if args.output is None:
        args.output = Path(config.RESULT_DIR) / 'representative_patches'
    if args.output_metadata is None:
        args.output_metadata = Path(config.RESULT_DIR) / 'representative_patches_metadata.json'

    args.output.mkdir(parents=True, exist_ok=True)

    # Read the test dataset
    test_subdatasets = [
        Dataset(
            csv_path=dataset_config.TEST_CSV_PATH,
            slide_dir=dataset_config.SLIDE_DIR,
            slide_file_extension=dataset_config.SLIDE_FILE_EXTENSION,
            contour_dir=dataset_config.CONTOUR_DIR,
            format='fit',
            resize_ratio=config.RESIZE_RATIO,
        )
        for dataset_config in config.TEST_DATASET_CONFIGS
    ]
    test_dataset = ConcatDataset(datasets=test_subdatasets)

    # For each class
    metadata_list: t.List[RepresentativePatchMetadata] = []
    for class_idx in range(config.NUM_CLASSES):
        # Read the visualization results for each test image
        visualization_results: t.List[npt.NDArray[np.float_]] = []
        for idx in range(len(test_dataset)):
            slide_name = test_dataset.get_slide_name(idx)
            path = Path(config.VIZ_RESULT_DIR) / f'{slide_name}.npy'
            heatmap = np.load(path)[..., class_idx]
            heatmap = np.nan_to_num(heatmap, nan=0.0)
            visualization_results.append(heatmap)

        # Apply NMS on the visualization result to prevent the sampling of
        # overlapped patches.
        if args.nms != 1:
            for idx in tqdm(
                range(len(test_dataset)),
                desc=f'Applying NMS of class {class_idx}',
            ):
                heatmap = visualization_results[idx]
                heatmap = _apply_nms_to_prevent_overlapping(
                    heatmap=heatmap,
                    kernel_size=args.nms,
                )
                visualization_results[idx] = heatmap

        # Get the threshold
        flattened_visualization_results = np.concatenate(
            [result.reshape([-1]) for result in visualization_results],
            axis=0,
        )  # [N]
        largest_k = np.partition(flattened_visualization_results, -args.top_k, axis=0)[-args.top_k:]  # [K]
        threshold = largest_k.min()  # []
        print(f'The threshold of class {class_idx}: {threshold}')

        # Get representative patches for each slide
        patches: t.List[_Patch] = []
        for dataset_idx, result in enumerate(
            tqdm(visualization_results, desc=f'Reading patches of class {class_idx}'),
        ):
            # Get the indices of patches
            representative_map = result >= threshold  # [H_R, W_R]
            representative_ys, representative_xs = np.nonzero(representative_map)
            if len(representative_ys) == 0:
                continue

            # Read regions
            image, _ = test_dataset[dataset_idx]  # [H, W, C]
            resize_ratio = test_dataset.get_resize_ratio(dataset_idx)
            x_stride = _calculate_heatmap_stride(
                heatmap_size=representative_map.shape[1],
                image_size=image.shape[1],
            )
            y_stride = _calculate_heatmap_stride(
                heatmap_size=representative_map.shape[0],
                image_size=image.shape[0],
            )
            for x, y in zip(representative_xs, representative_ys):
                patch_image, viewport_coordinate, viewport_size = _extract_representative_patch(
                    image,
                    coordinate=(x, y),
                    stride=(x_stride, y_stride),
                    patch_size=args.size,
                    resize_ratio=resize_ratio,
                )
                patches.append(
                    _Patch(
                        patch=patch_image,
                        score=result[y, x],
                        slide_name=test_dataset.get_slide_name(dataset_idx),
                        slide_path=Path(test_dataset.get_slide_path(dataset_idx)),
                        coordinate=(x, y),
                        viewport_coordinate=viewport_coordinate,
                        viewport_size=viewport_size,
                    ),
                )

        # Sort the representative patches and save them
        patches = sorted(patches, key=lambda patch: patch.score, reverse=True)
        for idx in range(args.top_k):
            if idx >= len(patches):
                break

            # Save the patch
            patch = patches[idx]
            filename = args.output_filename_template.replace('{class_idx}', str(class_idx)).replace(
                '{rank_idx}',
                str(idx),
            )
            path = args.output / filename
            Image.fromarray(patch.patch).save(path)
            print(
                f'Rank {idx}. Slide: {patch.slide_name}. '
                f'Coordinate: {patch.coordinate}. '
                f'Viewport coordinate: {patch.viewport_coordinate}. '
                f'Viewport size: {patch.viewport_size}. '
                f'Score: {patch.score:.3f}. Output: {filename}.',
            )

            # Update the metadata
            metadata = RepresentativePatchMetadata(
                output_path=path,
                class_idx=class_idx,
                rank_idx=idx,
                score=patch.score,
                slide_name=patch.slide_name,
                slide_path=patch.slide_path,
                viewport_coordinate=patch.viewport_coordinate,
                viewport_size=patch.viewport_size,
            )
            metadata_list.append(metadata)

    # Save the metadata
    representative_patches_metadata = RepresentativePatchesMetadata(
        __root__=metadata_list,
    )
    with open(args.output_metadata, 'w') as file:
        file.write(representative_patches_metadata.json(indent=4))
    print(f'The metadata has been saved as {args.output_metadata}.')


def _positive_odd_number(number: int) -> int:
    if number <= 0:
        raise RuntimeError('Not positive.')
    if number % 2 == 0:
        raise RuntimeError('Not odd.')
    return number


def _calculate_heatmap_stride(heatmap_size: int, image_size: int) -> int:
    if heatmap_size == 1:
        return 2 ** math.ceil(math.log2(image_size))
    else:
        return 2 ** math.floor(math.log2((image_size - 1) / (heatmap_size - 1)))


def _apply_nms_to_prevent_overlapping(
    heatmap: npt.NDArray[np.float_],
    kernel_size: int,
) -> npt.NDArray[np.float_]:
    kernel_size = _positive_odd_number(kernel_size)
    padding = kernel_size // 2
    padded_heatmap = np.pad(heatmap, padding, mode='constant')

    shape_w = (
        heatmap.shape[0],
        heatmap.shape[1],
        kernel_size,
        kernel_size,
    )
    strides_w = (
        padded_heatmap.strides[0],
        padded_heatmap.strides[1],
        padded_heatmap.strides[0],
        padded_heatmap.strides[1],
    )
    heatmap_w = numpy.lib.stride_tricks.as_strided(padded_heatmap, shape_w, strides_w)
    max_pooled = heatmap_w.max(axis=(2, 3))

    nms = np.where(
        heatmap == max_pooled,
        heatmap,
        np.zeros_like(heatmap),
    )
    return nms


@dataclass
class _Patch:
    patch: npt.NDArray[np.uint8]
    score: float
    slide_name: str
    slide_path: Path
    coordinate: t.Tuple[int, int]  # coordinate on the heatmap
    viewport_coordinate: t.Tuple[int, int]  # coordinate on the level-0 slide image
    viewport_size: t.Tuple[int, int]  # size on the level-0 slide image


def _extract_representative_patch(
    image: npt.NDArray[np.uint8],
    coordinate: t.Tuple[int, int],
    stride: t.Tuple[int, int],
    patch_size: int,
    resize_ratio: float,
) -> t.Tuple[npt.NDArray[np.uint8], t.Tuple[int, int], t.Tuple[int, int]]:
    center_x = stride[0] * (coordinate[0] + 0.5)
    center_y = stride[1] * (coordinate[1] + 0.5)
    left = int(center_x - patch_size / 2.0)
    top = int(center_y - patch_size / 2.0)
    right = left + patch_size
    bottom = top + patch_size

    true_left = max(left, 0)
    true_top = max(top, 0)
    true_right = min(right, image.shape[1])
    true_bottom = min(bottom, image.shape[0])

    patch = image[
        true_top:true_bottom,
        true_left:true_right,
        :,
    ].copy()
    patch = cv2.copyMakeBorder(
        patch,
        true_top - top,
        bottom - true_bottom,
        true_left - left,
        right - true_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    viewport_coordinate = (
        math.floor(true_left / resize_ratio),
        math.floor(true_top / resize_ratio),
    )
    viewport_size = (
        math.ceil((true_right - true_left) / resize_ratio),
        math.ceil((true_bottom - true_top) / resize_ratio),
    )
    return patch, viewport_coordinate, viewport_size


if __name__ == '__main__':
    main()
