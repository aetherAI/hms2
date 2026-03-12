import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import affine
import cv2
import matplotlib.cm
import numpy as np
import numpy.typing as npt
import rasterio.features
import scipy.linalg
import shapely
import torch
import torch.cuda.amp
import torch.nn as nn


try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


class HEDPerturbAugmentor:
    """
    An image augmentor that implements HED perturbing.

    Args:
        stain_angle (float): The maximal angle applied on perturbing the stain matrix.
        concentration_multiplier (tuple-like):
            A two-element tuple defining the scaling range of concentration perturbing.
    """

    def __init__(
        self,
        stain_angle: float = 10.0,
        concentration_multiplier: Tuple[float, float] = (0.5, 1.5),
    ):
        self.stain_angle = stain_angle
        self.concentration_multiplier = concentration_multiplier

        self.eps = 1e-6
        rgb_from_hed = np.array(
            [
                [0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11],
                [0.27, 0.57, 0.78],
            ],
        )
        self.hed_from_rgb = scipy.linalg.inv(rgb_from_hed)
        self.rgb_to_od_lut = self._get_rgb_to_od_lut()
        self.postfix = None

    def _get_rgb_to_od_lut(self) -> np.ndarray:
        rgb = np.arange(0, 256, dtype=np.uint8)
        rgb = rgb.astype(np.float32) / np.float32(255.0)
        rgb = np.maximum(rgb, self.eps)
        img_od = np.log(rgb) / np.log(self.eps)
        return img_od

    def _get_postfix(self) -> np.ndarray:
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

        # img_od_aug = img_od . hed_from_rgb . concentration_aug_matrix .
        # rgb_from_hed_aug
        postfix = np.matmul(concentration_aug_matrix, rgb_from_hed_aug)
        postfix = np.matmul(self.hed_from_rgb, postfix)

        self.postfix = postfix
        return postfix

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img_od = self.rgb_to_od_lut[img]

        postfix = self._get_postfix()
        img_od_aug = np.matmul(img_od, postfix)

        img_aug = np.digitize(img_od_aug, self.rgb_to_od_lut[:-1], right=True).astype(np.uint8)
        return img_aug


def draw_pred_map(
    pred_map: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Draw a heat map for a prediciton map.

    Args:
        pred_map (np.ndarray):
            A prediction map with the shape as (H, W) and the value range in [0.0, 1.0].
        size (NoneType or tuple-like):
            The desired size for the heat map. The default is None which sets the size
            of the heat map the same as that of the input prediction map.
    Return:
        heat_map (np.ndarray): An RGBA heat map with the shape (H, W, 4) and uint8.
    """

    if size is not None:
        pred_map = cv2.resize(pred_map, tuple(size), interpolation=cv2.INTER_CUBIC)
    pred_map = np.maximum(0.0, np.minimum(1.0, pred_map))

    color_map = matplotlib.cm.get_cmap(name='jet')(pred_map)[:, :, :3]  # type: ignore
    alpha_map = pred_map[:, :, np.newaxis]
    heat_map = np.concatenate([color_map, alpha_map], axis=-1)
    heat_map = np.maximum(0.0, np.minimum(255.0, heat_map * 255.0)).astype(np.uint8)

    return heat_map


class GradScaler(torch.cuda.amp.GradScaler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_step_skipped = None

    def _maybe_opt_step(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_state: dict,
        *args,
        **kwargs,
    ):
        retval = None
        if sum(v.item() for v in optimizer_state['found_inf_per_device'].values()) == 0:
            retval = optimizer.step(*args, **kwargs)
            self._last_step_skipped = False
        else:
            self._last_step_skipped = True

        return retval

    @property
    def last_step_skipped(self):
        return self._last_step_skipped


class TestResults:
    def __init__(
        self,
        test_result_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if test_result_list is None:
            test_result_list = []

        self.test_result_list = test_result_list

    @staticmethod
    def open(path: str) -> 'TestResults':
        with open(path) as file:
            test_result_list = json.load(file)

        test_results = TestResults(test_result_list)
        return test_results

    def __iter__(self) -> Iterable[Tuple[str, Union[int, List[int]], List[float]]]:
        for row in self.test_result_list:
            yield row['slide_name'], row['y_true'], row['y_pred']

    def dump(self, path: str) -> None:
        with open(path, 'w') as file:
            json.dump(self.test_result_list, file, indent=4)

    def append(
        self,
        slide_name: str,
        y_true: Union[int, List[int]],
        y_pred: List[float],
    ) -> None:
        self.test_result_list.append(
            {
                'slide_name': slide_name,
                'y_true': y_true,
                'y_pred': y_pred,
            },
        )


def find_optimal_scale(
    source_pixel_dim: float,
    target_pixel_dim: float,
    eps: Optional[float] = None,
) -> float:
    """
    Calculate the scale given pixel dimensions with some tolerance so that the returned
    scale is more likely to be a power of 2 (e.g., 1.0, 0.5, 0.25).

    Args:
        source_pixel_dim (float): The pixel dimension of a source image.
        target_pixel_dim (float): The pixel dimension of a target image.
        eps (Optional[float]):
            The tolerance to a power of 2. Set None to disable tolerance. Set np.inf
            to always return a power of 2.
    """
    scale = source_pixel_dim / target_pixel_dim
    if eps is not None:
        lg_scale = math.log(scale, 2)
        nearest_lg_power_of_2 = round(lg_scale)
        if math.isclose(lg_scale, nearest_lg_power_of_2, abs_tol=eps):
            scale = 2.0**nearest_lg_power_of_2

    return scale


def get_lora_config_modules(
    model,
    root_modules: List[str],
    modules_for_tune: List[str],
) -> Union[list, list]:
    """
    Get the module names for 1) freezed with LORA and tunable

    Args:
        root_modules (list[str]): root name of modules to apply LORA (e.g. conv_module)
        modules_for_tune (list[str]): Tunable operations root name (e.g. dense_module)
    """
    target_modules = []
    full_finetune_modules = []
    for k, v in model.named_modules():
        root_module_identify = re.findall(pattern='|'.join(root_modules), string=k)
        if (
            isinstance(v, (nn.Conv2d, nn.Linear))
            and root_module_identify
            and root_module_identify not in modules_for_tune
        ):
            # Operations to apply LORA
            target_modules.append(k)

        full_finetune_module_identify = re.findall(pattern='|'.join(modules_for_tune), string=k)
        if full_finetune_module_identify and (len(list(v.children())) == 0):
            full_finetune_modules.append(k)
    return target_modules, full_finetune_modules


def remove_unmatched_from_state_dict(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    matched_state_dict = {}
    for src_key, src_value in state_dict.items():
        if src_key not in model.state_dict() or src_value.shape != model.state_dict()[src_key].shape:
            print(f'{src_key} in state_dict has a different shape from that in model.')
            continue
        matched_state_dict[src_key] = src_value
    return matched_state_dict


def load_peft_model_separate(
    model,
    base_checkpoint: Union[str, dict],
    model_dir: str,
    adapter_name: str = 'default',
):
    """
    Instead of reading from whole finetuned state-dict,
        load the checkpoint from `pretrained` one and `peft` part.
    This is used for deployment that we only keep one copy of source model
        and have various finetuned parts.

    Args:
        model: The torch model
        base_checkpoint: Path or state_dict of the foundation model
        model_dir: Path to the result model directory that contains `adater_[model/config].bin`
        adapter_name: The appended module name  (default)
    Return:
        model: The modified and weight loaded model
    """
    if PeftModel is None:
        raise ImportError('You should install `peft` by `poe install-peft`.')
    if isinstance(base_checkpoint, str):
        base_checkpoint = torch.load(base_checkpoint)

    base_checkpoint = remove_unmatched_from_state_dict(base_checkpoint, model)
    model.load_state_dict(
        base_checkpoint,
        strict=False,
    )
    model = PeftModel.from_pretrained(
        model,
        model_dir,
        adapter_name=adapter_name,
        is_trainable=False,
    )
    return model


def get_backbone_stride(backbone: str) -> int:
    if backbone in [
        'resnet50_frozenbn',
        'resnet18_frozenbn',
        'resnet50_fixup',
        'resnet50_frozenbn_linear',
        'resnet50_frozenall_linear',
        'resnet50_frozenall',
    ]:
        return 32
    else:
        raise ValueError(f'Unrecognized backbone: {backbone}')


def make_invalid_nan(
    cam: npt.NDArray[np.float32],
    contours: List[List[Tuple[int, int]]],
    resize_ratio: float,
) -> npt.NDArray[np.float32]:
    mask = rasterio.features.rasterize(
        [shapely.MultiPolygon(polygons=[shapely.Polygon(shell=contour) for contour in contours])],  # noqa: S604
        out_shape=cam.shape[:2],
        transform=affine.Affine.scale(1.0 / resize_ratio),
        all_touched=True,
        dtype=np.uint8,
    ).astype(bool)
    new_cam = np.where(
        mask[:, :, np.newaxis],
        cam,
        np.nan,
    )
    return new_cam
