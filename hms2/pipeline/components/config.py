from __future__ import annotations

import re
import typing as t
from pathlib import Path

import pydantic
import yaml

from .dataset import ClassWeight, ResizeRatioByPixelSpacing
from .losses import LossConfig


T = t.TypeVar('T')


class TorchvisionPretrainOption(pydantic.BaseModel):
    type: t.Literal['torchvision'] = 'torchvision'
    weights: str


class BackbonePretrainOption(pydantic.BaseModel):
    type: t.Literal['backbone'] = 'backbone'
    path: str


class Hms2PretrainOption(pydantic.BaseModel):
    type: t.Literal['hms2'] = 'hms2'
    path: str


class NoPretrainOption(pydantic.BaseModel):
    type: t.Literal['no'] = 'no'


class TrainDatasetConfig(pydantic.BaseModel):
    TRAIN_CSV_PATH: str
    VAL_CSV_PATH: t.Optional[str] = None
    SLIDE_DIR: str
    SLIDE_FILE_EXTENSION: str
    CONTOUR_DIR: t.Optional[str] = None


class TrainConfig(pydantic.BaseModel):
    RESULT_DIR: str
    MODEL_PATH: str = '${RESULT_DIR}/model.pt'
    OPTIMIZER_STATE_PATH: str = '${RESULT_DIR}/opt_state.pt'
    HISTORY_DIR: t.Optional[str] = '${RESULT_DIR}/history/'
    STATES_PATH: str = '${RESULT_DIR}/states.pt'
    CONFIG_RECORD_PATH: str = '${RESULT_DIR}/config.yaml'
    TRAIN_EVENT_LOG_PATH: str = '${RESULT_DIR}/train_event_log.json'

    USE_MIXED_PRECISION: bool = False
    USE_HMS2: bool = True
    GPU_MEMORY_LIMIT_GB: t.Optional[float] = None

    RESIZE_RATIO: t.Union[float, ResizeRatioByPixelSpacing]
    GPU_AUGMENTS: t.List[str] = ['flip', 'rigid', 'hed_perturb']
    AUGMENTS: t.List[str] = []
    CLASS_WEIGHTS: t.Optional[t.List[ClassWeight]] = None

    NUM_CLASSES: int
    GPUS: t.Optional[int] = None
    MODEL: str = 'resnet50_frozenbn'
    PRETRAINED: t.Union[
        TorchvisionPretrainOption,
        BackbonePretrainOption,
        Hms2PretrainOption,
        NoPretrainOption,
        None,
    ] = TorchvisionPretrainOption(weights='IMAGENET1K_V1')
    PRE_POOLING: t.Optional[str] = 'no'
    POOL_USE: str = 're_lse'
    USE_LORA_FINETUNE: bool = False
    CUSTOM_DENSE: t.Optional[str] = None
    BATCH_SIZE: int = 1
    EPOCHS: int = 200
    LOSS: t.Union[LossConfig, str, None] = LossConfig(name='ce')
    METRIC_LIST: t.List[str] = ['accuracy']
    OPTIMIZER: str = 'adamw'
    INIT_LEARNING_RATE: float = 0.00001
    REDUCE_LR_FACTOR: float = 0.1
    REDUCE_LR_PATIENCE: int = 8

    TRAIN_DATASET_CONFIGS: pydantic.conlist(TrainDatasetConfig, min_items=1)

    CLASS_NAMES: t.Optional[t.List[str]] = None

    DEBUG_PATH: t.Optional[str] = None

    USE_EMBED_MODE: bool = False
    EMBED_MODE_CACHE_DIR: str = '${RESULT_DIR}/embedding'
    EMBED_MODE_GPU_COMPRESSOR: str = 'farthest_point_1/7'
    EMBED_MODE_COMPRESSORS: t.Sequence[str] = ['int8', 'unique_vector']

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Macro replacement
        macro_lut = {
            'RESULT_DIR': self.RESULT_DIR,
        }
        for key in self.__fields__:
            setattr(self, key, _apply_macro(getattr(self, key), macro_lut))

    @staticmethod
    def from_yaml(yaml_path: Path) -> 'TrainConfig':
        # Read YAML
        with open(yaml_path) as file:
            yaml_config = yaml.safe_load(file)

        # To pydantic
        config = TrainConfig.parse_obj(yaml_config)
        return config

    def save_yaml(self, yaml_path: Path) -> None:
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as file:
            yaml.dump(self.dict(), file)

    def save_yaml_snapshot(self) -> None:
        self.save_yaml(Path(self.CONFIG_RECORD_PATH))


class TestDatasetConfig(pydantic.BaseModel):
    TEST_CSV_PATH: str
    SLIDE_DIR: str
    SLIDE_FILE_EXTENSION: str
    CONTOUR_DIR: t.Optional[str] = None


class TestConfig(TrainConfig):
    TEST_RESULT_PATH: str = '${RESULT_DIR}/test_result.json'
    VIZ_RESULT_DIR: str = '${RESULT_DIR}/viz_results'
    TEST_EVENT_LOG_PATH: str = '${RESULT_DIR}/test_event_log.json'
    VIZ_EVENT_LOG_PATH: str = '${RESULT_DIR}/viz_event_log.json'

    TEST_DATASET_CONFIGS: pydantic.conlist(TestDatasetConfig, min_items=1)

    VIZ_POOL_USE: str = 'cam'

    @staticmethod
    def from_yaml(yaml_path: Path) -> 'TestConfig':
        # Read YAML
        with open(yaml_path) as file:
            yaml_config = yaml.safe_load(file)

        # If TRAIN_CONFIG_PATH exists, extract parameters from it.
        if 'TRAIN_CONFIG_PATH' in yaml_config:
            with open(yaml_config['TRAIN_CONFIG_PATH']) as file:
                train_yaml_config = yaml.safe_load(file)
            for key in train_yaml_config:
                if key in yaml_config:
                    print(
                        f'Repetitive key found: {key} in the training and test config. '
                        f'Adopted {key}={yaml_config[key]} from the test config.',
                    )
                    continue
                yaml_config[key] = train_yaml_config[key]
            del yaml_config['TRAIN_CONFIG_PATH']

        # To pydantic
        config = TestConfig.parse_obj(yaml_config)
        return config


def _apply_macro(
    data: T,
    macro_lut: dict[str, str],
) -> T:
    if isinstance(data, str):
        pattern = '\\${([^}]+)}'

        def _replace(matched: re.Match) -> str:
            variable = matched.group(1)
            if variable not in macro_lut:
                raise ValueError(f'Cannot find the macro: {variable}')
            return macro_lut[variable]

        data = re.sub(pattern, _replace, data)

    elif isinstance(data, dict):
        for key in data:
            data[key] = _apply_macro(data[key], macro_lut)

    elif isinstance(data, list):
        for index, element in enumerate(data):
            data[index] = _apply_macro(element, macro_lut)

    return data
