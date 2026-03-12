"""
This module provides a model building tool.
"""

import functools
import typing as t
from collections import namedtuple

import torch
import torch.nn as nn
import torchvision

from .compressors import get_compressor
from .custom_modules import (
    EuclideanFarthestPointPrePooling, FrozenBatchNorm2d,
    GaussianBlurAugmentorModule, HEDPerturbAugmentorModule, LogSumExpPool2d,
    PermuteLayer, ReCAM, ReLSEPool2d, ScaleAndShift, SoftplusInv,
)
from .custom_vit_modules import SimpleViT
from .fixup_resnet import fixup_resnet50
from .loader_modules import (
    EmbeddingLoaderModule, GPUAugmentationLoaderModule, NoLoaderModule, PlainLoaderModule,
)
from .model import Hms2Model
from .resnetv1c import resnetv1c50


class Hms2ModelBuilder:
    def __init__(self):
        self.Augmentation = namedtuple(
            'Augmentation',
            ['build_func'],
        )
        self.Backbone = namedtuple(
            'Backbone',
            ['build_func', 'output_channels', 'get_hms2_parameters', 'get_normalization_parameters'],
        )
        self.PrePooling = namedtuple(
            'PrePooling',
            ['build_func'],
        )
        self.Pooling = namedtuple(
            'Pooling',
            ['local_pooling_build_func', 'pooling_build_func'],
        )
        self.CustomDense = namedtuple(
            'CustomDense',
            ['custom_dense_build_func'],
        )

        self.augmentation_registry = {}
        self.backbone_registry = {}
        self.pooling_registry = {}
        self.pre_pooling_registry = {}
        self.custom_dense_registry = {}

        self._register_builtins()

    def build(
        self,
        n_classes: int,
        augmentation_list: t.Optional[t.Sequence[str]] = None,
        backbone: str = 'resnet50_frozenbn',
        pretrained: t.Union[None, t.OrderedDict[str, torch.Tensor], str] = 'DEFAULT',
        pre_pooling: t.Optional[str] = 'no',
        pooling: str = 'gmp',
        custom_dense: t.Optional[str] = None,
        use_hms2: bool = True,
        device: t.Optional[t.Union[torch.device, str, int]] = None,
        use_cpu_for_dense: bool = False,
        gpu_memory_budget: float = 32.0,
    ) -> t.Union['_PlainModel', Hms2Model]:
        """
        Build a model given parameters.

        Args:
            n_classes (int): The number of classes.
            augmentation_list (list or NoneType):
                A list of str, each of which specify an augmentation process, including
                'flip', 'rigid', and 'hed_perturb'. The default is None that disables
                GPU augmentations.
            backbone (str):
                Specify the backbone structure. One of 'resnet50_frozenbn' (default).
            pretrained (Union[None, OrderedDict[str, torch.Tensor], str]):
                Specify `None` for random init. Specify a `dict` to init from loading a
                state dict. Specify a `str` to use official pre-trained weights. Default
                to 'DEFAULT' to use the default pre-trained weight provided by
                torchvision.
            pre_pooling (str or NoneType):
                Specify the pre_pooling function. One of 'no' (default) and 'conv_1x1'.
            pooling (str or NoneType):
                Specify the pooling function. One of 'gmp', 'gmp_scaled', 'gap', 'lse',
                'cam', and 'no'.
            custom_dense (Optional[str]):
                Specify the module after pooling if not using the standard single dense
                layer. One of 'no'.
            use_hms2 (bool): Whether to enable HMS2. The default is True.
            device (torch.device):
                The device to place modules. If None (default), it calls
                torch.cuda.current_device() to get the device.
            use_cpu_for_dense: Whether to compute dense layers using CPU.
            gpu_memory_budget (float):
                The GPU memory capacity to let the builder determine the parameters of
                HMS2.
        """
        # Default arguments
        if augmentation_list is None:
            augmentation_list = []

        if pre_pooling is None:
            pre_pooling = 'no'

        if device is None:
            device = torch.cuda.current_device()

        # Build components
        loader_module = self._build_loader_module(
            use_hms2=use_hms2,
            augmentation_list=augmentation_list,
            backbone=backbone,
            device=device,
        )
        backbone_module = self._build_backbone_module(
            backbone=backbone,
            pretrained=pretrained,
            device=device,
        )
        pre_pooling_module = self._build_pre_pooling_module(
            backbone=backbone,
            pre_pooling=pre_pooling,
            device=device,
        )
        local_pooling_module = self._build_local_pooling_module(
            pooling=pooling,
            device=device,
        )
        dense_module = self._build_dense_module(
            backbone=backbone,
            pooling=pooling,
            custom_dense=custom_dense,
            n_classes=n_classes,
            device=('cpu' if use_cpu_for_dense else device),
        )

        # Build the model
        model: nn.Module
        if use_hms2:
            hms2_parameters = self.backbone_registry[backbone].get_hms2_parameters(
                gpu_memory_budget=gpu_memory_budget,
            )

            model = Hms2Model(
                loader_module=loader_module,
                conv_module=backbone_module,
                pre_pooling_module=pre_pooling_module,
                local_pooling_module=local_pooling_module,
                use_cpu_for_dense=use_cpu_for_dense,
                dense_module=dense_module,
                **hms2_parameters,
            )
        else:
            model = _PlainModel(
                loader_module=loader_module,
                conv_module=backbone_module,
                pre_pooling_module=pre_pooling_module,
                local_pooling_module=local_pooling_module,
                dense_module=dense_module,
            )

        return model

    def build_embedding(
        self,
        n_classes: int,
        backbone: str = 'resnet50_frozenbn',
        pretrained: t.Union[None, t.OrderedDict[str, torch.Tensor], str] = 'DEFAULT',
        pooling: str = 'gmp',
        custom_dense: t.Optional[str] = None,
        device: t.Optional[t.Union[torch.device, str, int]] = None,
        use_cpu_for_dense: bool = False,
        pre_pooling: t.Optional[str] = 'no',
        compressors: t.Sequence[str] = (),
    ) -> Hms2Model:
        """
        Build a model for training with embedding.
        Args:
            n_classes (int): The number of classes.
            backbone (str):
                Specify the backbone structure. One of 'resnet50_frozenbn' (default).
            pretrained (Union[None, OrderedDict[str, torch.Tensor], str]):
                Specify `None` for random init. Specify a `dict` to init from loading a
                state dict. Specify a `str` to use official pre-trained weights. Default
                to 'DEFAULT' to use the default pre-trained weight provided by
                torchvision.
            pooling (str or NoneType):
                Specify the pooling function. One of 'gmp', 'gmp_scaled', 'gap', 'lse',
                'cam', and 'no'.
            custom_dense (Optional[str]):
                Specify the module after pooling if not using the standard single dense
                layer. One of 'no'.
            device (torch.device):
                The device to place modules. If None (default), it calls
                torch.cuda.current_device() to get the device.
            use_cpu_for_dense: Whether to compute dense layers using CPU.
            pre_pooling (str or NoneType):
                Specify the pre_pooling function. One of 'no' (default) and 'conv_1x1'.
            compressors (List[str]):
                A list of compressors for embedding decompression.

        Returns:
            model (Hms2Model):
                The built model.
        """
        if device is None:
            device = torch.cuda.current_device()

        if pre_pooling is None:
            pre_pooling = 'no'

        # Build components
        loader_module = EmbeddingLoaderModule(compressors=[get_compressor(c) for c in compressors]).to(device)
        backbone_module = self._build_backbone_module(
            backbone=backbone,
            pretrained=pretrained,
            device=device,
        )
        pre_pooling_module = self._build_pre_pooling_module(
            backbone=backbone,
            pre_pooling=pre_pooling,
            device=device,
        )
        local_pooling_module = self._build_local_pooling_module(
            pooling=pooling,
            device=device,
        )
        dense_module = self._build_dense_module(
            backbone=backbone,
            pooling=pooling,
            custom_dense=custom_dense,
            n_classes=n_classes,
            device=('cpu' if use_cpu_for_dense else device),
        )

        # Build the model
        hms2_parameters = {
            'tile_size': 128 * 128,
            'emb_crop_size': 0,
            'emb_stride_size': 1,
        }
        model = Hms2Model(
            loader_module=loader_module,
            conv_module=backbone_module,
            pre_pooling_module=pre_pooling_module,
            local_pooling_module=local_pooling_module,
            use_cpu_for_dense=use_cpu_for_dense,
            dense_module=dense_module,
            skip_no_grad=False,
            cache_background_forward=False,
            cache_background_backward=False,
            **hms2_parameters,
        )
        return model

    def register_augmentation(
        self,
        signature: str,
        build_func: t.Callable,
    ) -> None:
        """Register an augmentation.

        Args:
            signature: The name of the augmentation.
            build_func: Calling build_func() will yield an nn.Module.
        """
        self.augmentation_registry[signature] = self.Augmentation(
            build_func=build_func,
        )

    def register_backbone(
        self,
        signature: str,
        build_func: t.Callable,
        output_channels: int,
        get_hms2_parameters: t.Callable,
        get_normalization_parameters: t.Optional[t.Callable] = None,
    ) -> None:
        """Register a backbone.

        Args:
            signature: The name of the backbone.
            build_func:
                Calling build_func(pretrained=xxx) will yield an nn.Module.
                'pretrained: bool' must be included as an argument.
            output_channels: The number of the output channels.
            get_hms2_parameters:
                A callable with a parameter `gpu_memory_budget`. It returns a dict
                with the keys 'tile_size', 'emb_crop_size', and 'emb_stride_size'.
            get_normalization_parameters:
                A callable that returns a dict with the keys 'mean' and 'std'.
                If None, it will be set to the default ImageNet values.
        """
        if get_normalization_parameters is None:
            # set to imagenet default if not provided
            get_normalization_parameters = lambda: {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
            }

        self.backbone_registry[signature] = self.Backbone(
            build_func=build_func,
            output_channels=output_channels,
            get_hms2_parameters=get_hms2_parameters,
            get_normalization_parameters=get_normalization_parameters,
        )

    def register_pre_pooling(
        self,
        signature: str,
        build_func: t.Callable[[int], nn.Module],
    ) -> None:
        """Register a pre_pooling.

        Args:
            signature: The name of the pre_pooling.
            build_func:
                Calling build_func() will yield an nn.Module. This pooling will be
                applied before the main pooling.
        """
        self.pre_pooling_registry[signature] = self.PrePooling(
            build_func=build_func,
        )

    def register_pooling(
        self,
        signature: str,
        local_pooling_build_func: t.Optional[t.Callable[[], nn.Module]],
        pooling_build_func: t.Callable[[], nn.Module],
    ) -> None:
        """Register a pooling.

        Args:
            signature: The name of the pooling.
            local_pooling_build_func:
                Calling local_pooling_build_func() will yield an nn.Module. This
                pooling will be applied before HMS2 aggregation.
            pooling_build_func:
                Calling pooling_build_func() will yield an nn.Module. This pooling will
                be applied before linear layers.
        """
        self.pooling_registry[signature] = self.Pooling(
            local_pooling_build_func=local_pooling_build_func,
            pooling_build_func=pooling_build_func,
        )

    def register_custom_dense(
        self,
        signature: str,
        custom_dense_build_func: t.Callable[[int, int], nn.Module],
    ) -> None:
        """Register a custom dense.

        Args:
            signature: The name of the custom dense.
            custom_dense_build_func:
                Calling custom_dense_build_func(channels, num_classes) will yeild an
                nn.Module.
        """
        self.custom_dense_registry[signature] = self.CustomDense(
            custom_dense_build_func=custom_dense_build_func,
        )

    def _register_builtins(self):  # noqa: C901
        # Augmentations
        self.register_augmentation('hed_perturb', HEDPerturbAugmentorModule)
        self.register_augmentation('gaussian_blur', GaussianBlurAugmentorModule)

        # Backbones
        def backbone_with_frozenbn_build_func(
            pretrained: t.Union[None, t.OrderedDict[str, torch.Tensor], str],
            backbone_build_func: _BackboneBuildFunc,
            frozen_all: bool = False,
            post_avg_pool: bool = False,
            post_linear_in_channels: t.Optional[int] = None,
            post_linear_out_channels: t.Optional[int] = None,
        ) -> nn.Module:
            if pretrained is None:
                backbone = backbone_build_func(weights=None)
            elif isinstance(pretrained, str):
                backbone = backbone_build_func(weights=pretrained)
            else:
                backbone = backbone_build_func(weights=None)
                missing_keys, unexpected_keys = backbone.load_state_dict(
                    pretrained,
                    strict=False,
                )
                if len(missing_keys) > 0:
                    print(f'Missing keys: {missing_keys}')
                if len(unexpected_keys) > 0:
                    print(f'Unexpected keys: {unexpected_keys}')

            backbone = FrozenBatchNorm2d.convert_frozen_batchnorm(backbone)

            if frozen_all:
                backbone = backbone.requires_grad_(False)

            module_list = list(backbone.children())[:-2]
            if post_avg_pool:
                module_list.append(
                    nn.AvgPool2d(
                        kernel_size=(7, 7),
                        stride=(1, 1),
                        padding=(3, 3),
                    ),
                )
            if post_linear_in_channels is not None and post_linear_out_channels is not None:
                module_list += [
                    nn.Conv2d(
                        in_channels=post_linear_in_channels,
                        out_channels=post_linear_out_channels,
                        kernel_size=(1, 1),
                    ),
                    nn.ReLU(),
                ]
            module = nn.Sequential(*module_list)

            torch.cuda.empty_cache()
            return module

        def resnet50_frozenbn_get_hms2_parameters(gpu_memory_budget: float) -> dict:
            if gpu_memory_budget >= 32:
                parameters = {
                    'tile_size': 3136,
                    'emb_crop_size': 7,
                    'emb_stride_size': 32,
                }
            else:
                parameters = {
                    'tile_size': 2016,
                    'emb_crop_size': 7,
                    'emb_stride_size': 32,
                }
            return parameters

        def resnet50_frozenall_get_hms2_parameters(gpu_memory_budget: float) -> dict:
            del gpu_memory_budget
            parameters = {
                'tile_size': 3136,
                'emb_crop_size': 7,
                'emb_stride_size': 32,
            }
            return parameters

        self.register_backbone(
            signature='resnet50V1c_frozenbn',
            build_func=functools.partial(
                backbone_with_frozenbn_build_func,
                backbone_build_func=lambda weights: resnetv1c50(),
            ),
            output_channels=2048,
            get_hms2_parameters=resnet50_frozenbn_get_hms2_parameters,
        )

        self.register_backbone(
            signature='resnet50_frozenbn',
            build_func=functools.partial(
                backbone_with_frozenbn_build_func,
                backbone_build_func=torchvision.models.resnet50,
            ),
            output_channels=2048,
            get_hms2_parameters=resnet50_frozenbn_get_hms2_parameters,
        )

        self.register_backbone(
            signature='resnet50_frozenbn_linear',
            build_func=functools.partial(
                backbone_with_frozenbn_build_func,
                backbone_build_func=torchvision.models.resnet50,
                post_linear_in_channels=2048,
                post_linear_out_channels=2048,
            ),
            output_channels=2048,
            get_hms2_parameters=resnet50_frozenbn_get_hms2_parameters,
        )

        self.register_backbone(
            signature='resnet50_frozenall',
            build_func=functools.partial(
                backbone_with_frozenbn_build_func,
                backbone_build_func=torchvision.models.resnet50,
                frozen_all=True,
            ),
            output_channels=2048,
            get_hms2_parameters=resnet50_frozenall_get_hms2_parameters,
        )

        self.register_backbone(
            signature='resnet50_frozenall_linear',
            build_func=functools.partial(
                backbone_with_frozenbn_build_func,
                backbone_build_func=torchvision.models.resnet50,
                frozen_all=True,
                post_linear_in_channels=2048,
                post_linear_out_channels=2048,
            ),
            output_channels=2048,
            get_hms2_parameters=resnet50_frozenall_get_hms2_parameters,
        )

        self.register_backbone(
            signature='resnet50_frozenall_ap_linear',
            build_func=functools.partial(
                backbone_with_frozenbn_build_func,
                backbone_build_func=torchvision.models.resnet50,
                frozen_all=True,
                post_avg_pool=True,
                post_linear_in_channels=2048,
                post_linear_out_channels=2048,
            ),
            output_channels=2048,
            get_hms2_parameters=resnet50_frozenall_get_hms2_parameters,
        )

        def resnet18_frozenbn_get_hms2_parameters(gpu_memory_budget: float) -> dict:
            del gpu_memory_budget
            parameters = {
                'tile_size': 3136,
                'emb_crop_size': 7,
                'emb_stride_size': 32,
            }
            return parameters

        self.register_backbone(
            signature='resnet18_frozenbn',
            build_func=functools.partial(
                backbone_with_frozenbn_build_func,
                backbone_build_func=torchvision.models.resnet18,
            ),
            output_channels=512,
            get_hms2_parameters=resnet18_frozenbn_get_hms2_parameters,
        )

        self.register_backbone(
            signature='resnet50_fixup',
            build_func=functools.partial(
                backbone_with_frozenbn_build_func,
                backbone_build_func=lambda weights: fixup_resnet50(),
            ),
            output_channels=2048,
            get_hms2_parameters=resnet50_frozenbn_get_hms2_parameters,
        )

        # Pre pooling
        self.register_pre_pooling(
            signature='avg_pool_7x7',
            build_func=(lambda channels: nn.AvgPool2d(kernel_size=(7, 7), ceil_mode=True)),
        )

        self.register_pre_pooling(
            signature='farthest_point_1/7',
            build_func=(lambda channels: EuclideanFarthestPointPrePooling(side_sample_ratio=1 / 7)),
        )

        self.register_pre_pooling(
            signature='identity',
            build_func=(lambda channels: nn.Identity()),
        )

        self.register_pre_pooling(
            signature='no',
            build_func=(lambda channels: nn.Identity()),
        )

        self.register_pre_pooling(
            signature='conv_1x1',
            build_func=(
                lambda channels: nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=(1, 1),
                    ),
                )
            ),
        )

        self.register_pre_pooling(
            signature='conv_1x1_relu',
            build_func=(
                lambda channels: nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=(1, 1),
                    ),
                    nn.ReLU(),
                )
            ),
        )

        # Poolings
        self.register_pooling(
            'gmp',
            local_pooling_build_func=(lambda: nn.AdaptiveMaxPool2d((1, 1))),
            pooling_build_func=(
                lambda: nn.Sequential(
                    nn.AdaptiveMaxPool2d((1, 1)),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            'gmp_scaled',
            local_pooling_build_func=(
                lambda: nn.Sequential(
                    ScaleAndShift(scale=3.79, bias=(-17.7)),
                    nn.AdaptiveMaxPool2d((1, 1)),
                )
            ),
            pooling_build_func=(
                lambda: nn.Sequential(
                    nn.AdaptiveMaxPool2d((1, 1)),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            'gmp_scaled_1k',
            local_pooling_build_func=(
                lambda: nn.Sequential(
                    ScaleAndShift(scale=3.933, bias=(-19.14)),
                    nn.AdaptiveMaxPool2d((1, 1)),
                )
            ),
            pooling_build_func=(
                lambda: nn.Sequential(
                    nn.AdaptiveMaxPool2d((1, 1)),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            'gmp_scaled_2k',
            local_pooling_build_func=(
                lambda: nn.Sequential(
                    ScaleAndShift(scale=4.135, bias=(-21.23)),
                    nn.AdaptiveMaxPool2d((1, 1)),
                )
            ),
            pooling_build_func=(
                lambda: nn.Sequential(
                    nn.AdaptiveMaxPool2d((1, 1)),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            'gap',
            local_pooling_build_func=(lambda: nn.AdaptiveAvgPool2d((1, 1))),
            pooling_build_func=(
                lambda: nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            'lse',
            local_pooling_build_func=(lambda: LogSumExpPool2d(factor=1.0)),
            pooling_build_func=(
                lambda: nn.Sequential(
                    LogSumExpPool2d(factor=1.0),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            'lsem1',
            local_pooling_build_func=(
                lambda: nn.Sequential(
                    SoftplusInv(),
                    LogSumExpPool2d(factor=1.0),
                )
            ),
            pooling_build_func=(
                lambda: nn.Sequential(
                    LogSumExpPool2d(factor=1.0),
                    nn.Softplus(),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            're_lse',
            local_pooling_build_func=(lambda: ReLSEPool2d()),
            pooling_build_func=(
                lambda: nn.Sequential(
                    ReLSEPool2d(),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            'cam',
            local_pooling_build_func=None,
            pooling_build_func=(lambda: PermuteLayer(dims=(0, 2, 3, 1))),
        )
        self.register_pooling(
            're_cam',
            local_pooling_build_func=(lambda: ReCAM()),
            pooling_build_func=(lambda: PermuteLayer(dims=(0, 2, 3, 1))),
        )
        self.register_pooling(
            'no',
            local_pooling_build_func=None,
            pooling_build_func=(lambda: nn.Identity()),
        )

        # Custom Dense
        self.register_custom_dense(
            'no',
            custom_dense_build_func=(lambda channels, n_classes: nn.Identity()),
        )
        self.register_custom_dense(
            'simple_vit_ti_ap16',
            custom_dense_build_func=(
                lambda channels, n_classes: nn.Sequential(
                    nn.AvgPool2d(kernel_size=16, ceil_mode=True),
                    SimpleViT(
                        num_classes=n_classes,
                        dim=192,
                        depth=12,
                        heads=3,
                        mlp_dim=768,
                        channels=channels,
                    ),
                )
            ),
        )

    def _build_loader_module(
        self,
        use_hms2: bool,
        augmentation_list: t.Sequence[str],
        backbone: str,
        device: t.Union[torch.device, str, int],
    ) -> nn.Module:
        # Translate the augmentation_list
        augmentation_modules = []

        # Get normalization_parameters
        if backbone not in self.backbone_registry:
            raise RuntimeError(f'{backbone} has not yet registered as a backbone.')

        normalization_parameters = self.backbone_registry[backbone].get_normalization_parameters()
        normalization_mean = normalization_parameters['mean']
        normalization_std = normalization_parameters['std']

        for augmentation in augmentation_list:
            if use_hms2 and augmentation in ['flip', 'rigid']:
                # 'flip' and 'rigid' are built-in of HMS2 to enable patch-based
                # affine transformation. Skip initiating a module.
                pass

            elif augmentation in self.augmentation_registry:
                module = self.augmentation_registry[augmentation].build_func()
                augmentation_modules.append(module)

            else:
                raise RuntimeError(f'{augmentation} has not yet been registered as an augmentation.')

        # Build the loader module
        loader_module: nn.Module
        if use_hms2:
            if augmentation_list is None:
                loader_module = PlainLoaderModule(normalization_mean, normalization_std)
            else:
                random_rotation = 'rigid' in augmentation_list
                random_translation = (-32.0, 32.0) if 'rigid' in augmentation_list else None
                random_flip = 'flip' in augmentation_list

                loader_module = GPUAugmentationLoaderModule(
                    normalization_mean=normalization_mean,
                    normalization_std=normalization_std,
                    random_rotation=random_rotation,
                    random_translation=random_translation,
                    random_flip=random_flip,
                    other_augmentations=augmentation_modules,
                )
        else:
            loader_module = NoLoaderModule(
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
                augmentations=augmentation_modules,
            )

        loader_module = loader_module.to(device)
        return loader_module

    def _build_backbone_module(
        self,
        backbone: str,
        pretrained: t.Union[None, t.OrderedDict[str, torch.Tensor], str],
        device: t.Union[torch.device, str, int],
    ) -> nn.Module:
        if backbone not in self.backbone_registry:
            raise RuntimeError(f'{backbone} has not yet registered as a backbone.')

        backbone_module = self.backbone_registry[backbone].build_func(pretrained=pretrained)
        backbone_module = backbone_module.to(device)
        return backbone_module

    def _build_pre_pooling_module(
        self,
        backbone: str,
        pre_pooling: t.Optional[str],
        device: t.Union[torch.device, str, int],
    ) -> nn.Module:
        if backbone not in self.backbone_registry:
            raise RuntimeError(f'{backbone} has not yet registered as a backbone.')

        output_channels = self.backbone_registry[backbone].output_channels

        if pre_pooling not in self.pre_pooling_registry:
            raise RuntimeError(f'{pre_pooling} has not yet registered as a pre-pooling.')

        pre_pooling_build_func = self.pre_pooling_registry[pre_pooling].build_func
        if pre_pooling_build_func is None:
            return nn.Identity()

        module = pre_pooling_build_func(channels=output_channels)
        return module.to(device)

    def _build_local_pooling_module(
        self,
        pooling: str,
        device: t.Union[torch.device, str, int],
    ) -> t.Optional[nn.Module]:
        if pooling not in self.pooling_registry:
            raise RuntimeError(f'{pooling} has not yet registered as a pooling.')

        local_pooling_build_func = self.pooling_registry[pooling].local_pooling_build_func
        if local_pooling_build_func is None:
            return None

        local_pooling_module = local_pooling_build_func()
        local_pooling_module = local_pooling_module.to(device)

        return local_pooling_module

    def _build_dense_module(
        self,
        backbone: str,
        pooling: str,
        custom_dense: t.Optional[str],
        n_classes: int,
        device: t.Union[torch.device, str, int],
    ) -> nn.Module:
        if backbone not in self.backbone_registry:
            raise RuntimeError(f'{backbone} has not yet registered as a backbone.')

        output_channels = self.backbone_registry[backbone].output_channels

        if pooling not in self.pooling_registry:
            raise RuntimeError(f'{pooling} has not yet registered as a pooling.')

        pooling_module = self.pooling_registry[pooling].pooling_build_func()

        if custom_dense is None:
            dense_layer = nn.Linear(output_channels, n_classes, bias=True)
            with torch.no_grad():
                dense_layer.weight.div_(10.0)
                dense_layer.bias.div_(10.0)
        else:
            dense_layer_build_func = self.custom_dense_registry[custom_dense].custom_dense_build_func
            dense_layer = dense_layer_build_func(
                channels=output_channels,
                n_classes=n_classes,
            )

        dense_module = nn.Sequential(
            pooling_module,
            dense_layer,
        )
        dense_module = dense_module.to(device)

        return dense_module


class _PlainModel(nn.Module):
    """
    Plain model with a similar interface as Hms2Model.

    Args:
        See the descriptions in `Hms2Model`.
    """

    def __init__(
        self,
        loader_module: nn.Module,
        conv_module: nn.Module,
        pre_pooling_module: nn.Module,
        local_pooling_module: t.Optional[nn.Module],
        dense_module: nn.Module,
    ):
        super().__init__()

        self.loader_module = loader_module
        self.conv_module = conv_module
        self.pre_pooling_module = pre_pooling_module
        self.local_pooling_module = local_pooling_module
        self.dense_module = dense_module

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        """
        Implementation of a plain model.
        """
        if isinstance(img_batch, torch.Tensor):
            if len(img_batch.size()) != 4:
                raise ValueError('img_batch should have 4 dimensions')
        else:
            raise ValueError('img_batch should be torch.Tensor')

        loaded = self.loader_module(img_batch)
        conved = self.conv_module(loaded)

        if self.pre_pooling_module is not None:
            conved = self.pre_pooling_module(conved)

        if self.local_pooling_module is not None:
            local_pooled = self.local_pooling_module(conved)
        else:
            local_pooled = conved
        output = self.dense_module(local_pooled)

        return output


class _BackboneBuildFunc(t.Protocol):
    def __call__(self, weights: t.Optional[str]) -> nn.Module:
        ...
