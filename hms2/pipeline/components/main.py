import gc
import itertools
import math
import os
import shutil
import sys
import typing as t
from pathlib import Path

import devtools
import horovod.torch as hvd
import numpy as np
import torch
import torch.cuda
import torch.utils.data
import torch.utils.data.distributed
from mpi4py import MPI
from scoping import scoping
from tqdm import tqdm

from ...core.builder import Hms2ModelBuilder
from .callbacks import (
    Callback, CallbackList, ContinueModule, DataBoundWarningCallback, EarlyStoppingOnLRDecay,
    EventLoggingCallback, MetricHandler, ModelCheckpoint, RollbackOnLRDecay,
)
from .config import (
    BackbonePretrainOption, Hms2PretrainOption, NoPretrainOption,
    TestConfig, TorchvisionPretrainOption, TrainConfig,
)
from .dataset import ConcatDataset, Dataset, DistributedWeightedSampler
from .event_logging import EventLogger, TestBatchEvent, VisualizeBatchEvent
from .losses import get_activation_fn, get_loss_fn
from .main_emb import train_main_embed_mode
from .optimizers import get_optimizer
from .utils import (
    GradScaler, TestResults, get_backbone_stride, get_lora_config_modules,
    make_invalid_nan, remove_unmatched_from_state_dict,
)


try:
    import peft
except ImportError:
    peft = None


def _initialize_horovod() -> t.Tuple[int, int, bool]:
    # Initialize Pytorch and Horovod if required.
    hvd.init()
    torch.cuda.set_device(hvd.local_rank() % torch.cuda.device_count())

    worker_rank = hvd.rank()
    num_workers = hvd.size()
    is_rank0 = hvd.rank() == 0
    return worker_rank, num_workers, is_rank0


_ConfigType = t.TypeVar('_ConfigType', TrainConfig, TestConfig)


def _get_config(config_path: str, is_rank0: bool, config_type: t.Type[_ConfigType]) -> _ConfigType:
    # Read the config
    config = config_type.from_yaml(Path(config_path))

    if config.USE_LORA_FINETUNE and peft is None:
        raise ImportError('You should install `peft` by `poe install-peft`.')

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    if is_rank0:
        print('### Config ###')
        devtools.debug(config)

        try:
            shutil.copyfile(config_path, config.CONFIG_RECORD_PATH)
        except shutil.SameFileError:
            pass

    return config


def _set_gpu_memory_limit(rank: int, gpu_memory_limit_gb: t.Optional[float]) -> None:
    _, total_gpu_memory = torch.cuda.mem_get_info()
    total_gpu_memory_gb = total_gpu_memory / 1024**3
    if gpu_memory_limit_gb is None:
        memory_fraction = 1.0 / math.ceil(hvd.local_size() / torch.cuda.device_count())
    else:
        memory_fraction = min(1.0, gpu_memory_limit_gb / total_gpu_memory_gb)
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    actual_gpu_memory_limit_gb = total_gpu_memory_gb * memory_fraction
    print(f'On rank {rank}, total GPU memory: {total_gpu_memory_gb} GB. Limit: {actual_gpu_memory_limit_gb} GB.')


def train_main(config_path: str, continue_mode: bool) -> None:  # noqa: C901
    # Initialize and get the config
    worker_rank, num_workers, is_rank0 = _initialize_horovod()
    config = _get_config(config_path, is_rank0, TrainConfig)
    _set_gpu_memory_limit(worker_rank, config.GPU_MEMORY_LIMIT_GB)

    # Check the number of GPUs
    if config.GPUS is not None:
        if num_workers != config.GPUS:
            raise ValueError(f'{config.GPUS} GPUs are required, but we got {num_workers}.')

    if config.USE_EMBED_MODE:
        train_main_embed_mode(config, continue_mode, worker_rank, num_workers, is_rank0)
    else:
        train_main_standard_mode(config, continue_mode, worker_rank, num_workers, is_rank0)


def train_main_standard_mode(  # noqa: C901
    config: TrainConfig,
    continue_mode: bool,
    worker_rank: int,
    num_workers: int,
    is_rank0: bool,
) -> None:
    # Initialize and get the config
    _set_gpu_memory_limit(worker_rank, config.GPU_MEMORY_LIMIT_GB)

    # Check the number of GPUs
    if config.GPUS is not None:
        if num_workers != config.GPUS:
            raise ValueError(f'{config.GPUS} GPUs are required, but we got {num_workers}.')

    # Initialize datasets
    if is_rank0:
        print('### Initializing datasets ###')
    train_subdatasets = [
        Dataset(
            csv_path=dataset_config.TRAIN_CSV_PATH,
            slide_dir=dataset_config.SLIDE_DIR,
            slide_file_extension=dataset_config.SLIDE_FILE_EXTENSION,
            contour_dir=dataset_config.CONTOUR_DIR,
            format='safe_rotate',
            resize_ratio=config.RESIZE_RATIO,
            snapshot_path=(config.DEBUG_PATH if is_rank0 else None),
            augment_list=config.AUGMENTS,
        )
        for dataset_config in config.TRAIN_DATASET_CONFIGS
    ]
    train_dataset = ConcatDataset(datasets=train_subdatasets)
    if is_rank0:
        print(f'Training dataset contains {len(train_dataset)} slides.')

    val_subdatasets = [
        Dataset(
            csv_path=dataset_config.VAL_CSV_PATH,
            slide_dir=dataset_config.SLIDE_DIR,
            slide_file_extension=dataset_config.SLIDE_FILE_EXTENSION,
            contour_dir=dataset_config.CONTOUR_DIR,
            format='trim',
            resize_ratio=config.RESIZE_RATIO,
            snapshot_path=(config.DEBUG_PATH if is_rank0 else None),
        )
        for dataset_config in config.TRAIN_DATASET_CONFIGS
        if dataset_config.VAL_CSV_PATH is not None
    ]
    val_dataset = ConcatDataset(datasets=val_subdatasets) if len(val_subdatasets) > 0 else None
    if is_rank0:
        if val_dataset is None:
            print('No validation dataset.')
        else:
            print(f'Validation dataset contains {len(val_dataset)} slides.')

    # Initialize dataloaders.
    if is_rank0:
        print('### Initializing dataloaders ###')
    if config.CLASS_WEIGHTS is not None:
        train_sampler = DistributedWeightedSampler(
            train_dataset,
            num_replicas=num_workers,
            rank=worker_rank,
            shuffle=True,
            class_weights=config.CLASS_WEIGHTS,
        )
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=num_workers,
            rank=worker_rank,
            shuffle=True,
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=False,
    )
    print(f'Train dataloader contains {len(train_dataloader)} slides on rank {worker_rank}.')

    if val_dataset is not None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=num_workers,
            rank=worker_rank,
            shuffle=False,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=val_sampler,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=False,
        )
        print(f'Val dataloader contains {len(val_dataloader)} slides on rank {worker_rank}.')
    else:
        val_sampler = None
        val_dataloader = None

    # Initialize the model
    if is_rank0:
        print('### Initializing the model ###')

    pretrained: t.Union[None, t.OrderedDict[str, torch.Tensor], str]
    if not is_rank0:
        # Only the rank0 worker should load the pretrained weight. This avoids race
        # condition.
        pretrained = None
    elif isinstance(config.PRETRAINED, NoPretrainOption):
        # Random init
        pretrained = None
    elif isinstance(config.PRETRAINED, TorchvisionPretrainOption):
        # Load official pretrained weights
        pretrained = config.PRETRAINED.weights
    elif isinstance(config.PRETRAINED, BackbonePretrainOption):
        # Load backbone pretrained weights
        pretrained = torch.load(
            config.PRETRAINED.path,
            map_location='cpu',
        )
        if is_rank0:
            print(
                f'The pretrained weight {config.PRETRAINED.path} was loaded',
            )
    elif isinstance(config.PRETRAINED, Hms2PretrainOption):
        # Load the weights later
        pretrained = None
    else:
        raise ValueError

    model = Hms2ModelBuilder().build(
        n_classes=config.NUM_CLASSES,
        augmentation_list=config.GPU_AUGMENTS,
        backbone=config.MODEL,
        pretrained=pretrained,
        pre_pooling=config.PRE_POOLING,
        pooling=config.POOL_USE,
        custom_dense=config.CUSTOM_DENSE,
        use_hms2=config.USE_HMS2,
    )
    if is_rank0 and isinstance(config.PRETRAINED, Hms2PretrainOption):
        pretrained = torch.load(
            config.PRETRAINED.path,
            map_location='cpu',
        )
        pretrained = remove_unmatched_from_state_dict(pretrained, model)
        model.load_state_dict(pretrained, strict=False)
    if config.USE_LORA_FINETUNE:
        target_modules, full_finetune_modules = get_lora_config_modules(
            model=model,
            root_modules=['conv_module'],
            modules_for_tune=['dense_module'],
        )
        lora_config = peft.LoraConfig(
            r=32,
            lora_alpha=8,
            target_modules=target_modules,
            modules_to_save=full_finetune_modules,  # It is not to meant to `save`, it means to `tune`
        )
        model = peft.get_peft_model(model, lora_config)
        if is_rank0:
            lora_config.save_pretrained(config.RESULT_DIR)  # Save configuration
            model.print_trainable_parameters()

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    if is_rank0:
        print(model)

    # Initialize the loss and activation function
    loss_fn = get_loss_fn(config.LOSS)
    activation_fn = get_activation_fn(config.LOSS)

    # Initialize the optimizer
    if config.USE_LORA_FINETUNE:
        named_params = [(p_name, p_value) for p_name, p_value in model.named_parameters() if p_value.requires_grad]
        params = [param[1] for param in named_params]
    else:
        named_params = itertools.chain(model.named_parameters(), loss_fn.named_parameters())
        params = itertools.chain(model.parameters(), loss_fn.parameters())

    optimizer_local = get_optimizer(
        optimizer_name=config.OPTIMIZER,
        model=model,
        lr=config.INIT_LEARNING_RATE,
        params=params,
    )
    optimizer = hvd.DistributedOptimizer(
        optimizer_local,
        named_parameters=named_params,
    )

    # Initialize the gradient scaler for mixed precision training
    grad_scaler = None
    if config.USE_MIXED_PRECISION:
        grad_scaler = GradScaler(
            init_scale=65536.0,
            growth_interval=100,
        )

    # Initialize the LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        verbose=is_rank0,
    )

    # Setup callbacks
    callback_list: t.List[Callback] = []
    callback_list.append(
        MetricHandler(
            metric_list=config.METRIC_LIST,
            use_horovod=True,
            activation_fn=activation_fn,
            verbose=is_rank0,
        ),
    )
    callback_list.append(DataBoundWarningCallback())
    if config.REDUCE_LR_FACTOR > 0.0:
        callback_list.append(
            RollbackOnLRDecay(
                model=model,
                optimizer=optimizer,
                filepath=config.MODEL_PATH,
                optimizer_state_filepath=config.OPTIMIZER_STATE_PATH,
                verbose=is_rank0,
            ),
        )
    else:
        callback_list.append(
            EarlyStoppingOnLRDecay(
                optimizer=optimizer,
                verbose=is_rank0,
            ),
        )
    if is_rank0:
        callback_list.append(
            ModelCheckpoint(
                model=model,
                filepath=config.MODEL_PATH,
                monitor='val_loss',
                mode='min',
                save_best_only=(val_dataset is not None),
                optimizer=optimizer,
                optimizer_state_filepath=config.OPTIMIZER_STATE_PATH,
                is_lora_model=config.USE_LORA_FINETUNE,
                verbose=True,
                history_folder=config.HISTORY_DIR,
            ),
        )
        callback_list.append(
            EventLoggingCallback(
                log_path=Path(config.TRAIN_EVENT_LOG_PATH),
            ),
        )
    callback_list.append(
        ContinueModule(
            model=model,
            filepath=config.STATES_PATH,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            callbacks=callback_list,
            load_states=continue_mode,
            save_states=is_rank0,
            verbose=is_rank0,
        ),
    )
    callback_list_object = CallbackList(callback_list)

    # Load the model if required
    if continue_mode:
        last_epoch_idx = torch.load(
            config.STATES_PATH,
            map_location='cpu',
        )['epoch_idx']
        epoch_begin = last_epoch_idx + 1
    else:
        epoch_begin = 0

    # Start training
    if is_rank0:
        print('### Start training ###')

    for epoch_idx in range(epoch_begin, config.EPOCHS):
        if is_rank0:
            print(f'Epoch {epoch_idx}/{config.EPOCHS}')

        # Training phase
        with scoping():
            callback_list_object.on_train_epoch_begin(epoch_idx)
            train_sampler.set_epoch(epoch_idx)
            model.train()

            losses = []
            progress_bar = tqdm(
                train_dataloader,
                desc='train',
                disable=(not is_rank0),
                leave=False,
                file=sys.stdout,
            )
            for batch_idx, (img_batch, y_true) in enumerate(progress_bar):
                callback_list_object.on_train_batch_begin(batch_idx)

                if not config.USE_HMS2:
                    # If HMS2 is disabled, the input should be manually moved to GPU.
                    img_batch = img_batch.cuda()

                step_success = False
                loss = None
                y_pred = None
                while not step_success:
                    optimizer.zero_grad()
                    with torch.autocast('cuda', enabled=config.USE_MIXED_PRECISION):
                        y_pred = model(img_batch)
                        y_true = y_true.cuda()
                        loss = loss_fn.cuda()(y_pred, y_true)

                    if config.USE_MIXED_PRECISION:
                        if grad_scaler is None:
                            raise RuntimeError
                        grad_scaler.scale(loss).backward()  # type: ignore
                        optimizer.synchronize()  # Sync gradients
                        with optimizer.skip_synchronize():
                            grad_scaler.step(optimizer)  # Step or abandon the batch
                        grad_scaler.update()
                        step_success = not grad_scaler.last_step_skipped
                    else:
                        loss.backward()
                        optimizer.step()
                        step_success = True

                with torch.no_grad():
                    loss_aggregated = hvd.allreduce(loss).item()
                losses.append(loss_aggregated)

                callback_list_object.on_train_batch_end(
                    batch_idx,
                    logs={
                        'y_true': y_true.detach().cpu().numpy(),
                        'y_pred': y_pred.detach().cpu().numpy(),
                        'epoch_idx': epoch_idx,
                        'total_batches': len(train_dataloader),
                    },
                )

                progress_bar.set_postfix(
                    {
                        'train_loss': np.mean(losses),
                    },
                )
                progress_bar.refresh()

            train_loss = np.mean(losses)
            if is_rank0:
                print(f'Training loss: {train_loss}')

            callback_list_object.on_train_epoch_end(
                epoch_idx,
                logs={
                    'train_loss': train_loss,
                },
            )

        gc.collect()

        # Validation phase
        if val_dataloader is not None:
            callback_list_object.on_validation_epoch_begin(epoch_idx)
            model.eval()

            y_preds = []
            y_trues = []
            progress_bar = tqdm(
                val_dataloader,
                desc='val',
                disable=(not is_rank0),
                leave=False,
                file=sys.stdout,
            )
            for batch_idx, (img_batch, y_true) in enumerate(progress_bar):
                callback_list_object.on_validation_batch_begin(batch_idx)

                if not config.USE_HMS2:
                    # If HMS2 is disabled, the input should be manually moved to GPU.
                    img_batch = img_batch.cuda()

                with torch.no_grad():
                    y_pred = model(img_batch)

                y_preds.append(y_pred.cpu())
                y_trues.append(y_true)

                callback_list_object.on_validation_batch_end(
                    batch_idx,
                    logs={
                        'y_true': y_true.detach().cpu().numpy(),
                        'y_pred': y_pred.detach().cpu().numpy(),
                        'epoch_idx': epoch_idx,
                        'total_batches': len(val_dataloader),
                    },
                )

            with torch.no_grad():
                y_preds_tensor = torch.cat(y_preds, dim=0)
                y_trues_tensor = torch.cat(y_trues, dim=0)
                val_loss_local = loss_fn.cpu()(y_preds_tensor, y_trues_tensor)
                val_loss = hvd.allreduce(val_loss_local)
            val_loss = val_loss.item()

            if is_rank0:
                print(f'Validation loss: {val_loss}')

            lr_scheduler.step(val_loss)
            callback_list_object.on_validation_epoch_end(
                epoch_idx,
                logs={
                    'val_loss': val_loss,
                },
            )

        gc.collect()


def test_main(config_path: str) -> None:
    # Initialize and get the config
    worker_rank, num_workers, is_rank0 = _initialize_horovod()
    config = _get_config(config_path, is_rank0, TestConfig)
    _set_gpu_memory_limit(worker_rank, config.GPU_MEMORY_LIMIT_GB)

    # Create results folder
    os.makedirs(os.path.dirname(config.TEST_RESULT_PATH), exist_ok=True)

    # Initialize datasets
    if is_rank0:
        print('### Initializing datasets ###')
    test_subdatasets = [
        Dataset(
            csv_path=dataset_config.TEST_CSV_PATH,
            slide_dir=dataset_config.SLIDE_DIR,
            slide_file_extension=dataset_config.SLIDE_FILE_EXTENSION,
            contour_dir=dataset_config.CONTOUR_DIR,
            format='trim',
            resize_ratio=config.RESIZE_RATIO,
            snapshot_path=(config.DEBUG_PATH if is_rank0 else None),
        )
        for dataset_config in config.TEST_DATASET_CONFIGS
    ]
    test_dataset = ConcatDataset(datasets=test_subdatasets)
    if is_rank0:
        print(f'Test dataset contains {len(test_dataset)} slides.')

    # Initialize samplers
    if is_rank0:
        print('### Initializing samplers ###')
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=num_workers,
        rank=worker_rank,
        shuffle=False,
    )
    print(f'Test sampler contains {len(test_sampler)} slides on rank {worker_rank}.')

    # Initialize the model
    if is_rank0:
        print('### Initializing the model ###')

    model = Hms2ModelBuilder().build(
        n_classes=config.NUM_CLASSES,
        backbone=config.MODEL,
        pretrained=None,
        pre_pooling=config.PRE_POOLING,
        pooling=config.POOL_USE,
        custom_dense=config.CUSTOM_DENSE,
        use_hms2=config.USE_HMS2,
    )
    if config.USE_LORA_FINETUNE:
        # Load peft/lora config
        peft_cfg = peft.PeftConfig.from_pretrained(config.RESULT_DIR)
        model = peft.PeftModel(model, peft_config=peft_cfg)

    missing_keys, unexpected_keys = model.load_state_dict(
        torch.load(config.MODEL_PATH, map_location='cpu'),
        strict=False,
    )
    if len(missing_keys) > 0:
        raise ValueError(f'Missing keys: {missing_keys}.')
    if len(unexpected_keys) > 0:
        print(f'Unexpected keys: {unexpected_keys}.')

    if is_rank0:
        print(f'The weight file {config.MODEL_PATH} is loaded.')
    if is_rank0:
        print(model)

    # Determine the activation applied on predictions
    activation_fn = get_activation_fn(config.LOSS)

    # Initialize the event logger
    event_logger = None
    if is_rank0:
        event_logger = EventLogger(Path(config.TEST_EVENT_LOG_PATH))

    # Start testing
    if is_rank0:
        print('### Start testing ###')

    model.eval()
    y_preds = {}
    progress_bar = tqdm(
        test_sampler,
        desc='test',
        disable=(not is_rank0),
        leave=False,
        file=sys.stdout,
    )
    for batch_idx, dataset_idx in enumerate(progress_bar):
        img, _ = test_dataset[dataset_idx]
        img_batch = torch.tensor(img[np.newaxis, ...])

        if not config.USE_HMS2:
            # If HMS2 is disabled, the input should be manually move to GPU.
            img_batch = img_batch.cuda()

        y_pred = model(img_batch)[0]
        y_pred = activation_fn(y_pred.detach().cpu().numpy()).tolist()
        y_preds[dataset_idx] = y_pred

        if is_rank0:
            if event_logger is None:
                raise RuntimeError
            event_logger.append_and_write(
                TestBatchEvent(
                    batch=batch_idx,
                    total_batches=len(test_sampler),
                ),
            )

    # Aggregate results and save them
    y_preds = MPI.COMM_WORLD.gather(y_preds, root=0)
    y_preds = MPI.COMM_WORLD.bcast(y_preds, root=0)
    y_preds = {dataset_idx: subdict[dataset_idx] for subdict in y_preds for dataset_idx in subdict}

    test_results = TestResults()
    for dataset_idx in y_preds:
        slide_name = test_dataset.get_slide_name(dataset_idx)
        y_true = test_dataset.get_y_true(dataset_idx)
        y_pred = y_preds[dataset_idx]

        if isinstance(y_true, np.ndarray):
            y_true = y_true.tolist()

        test_results.append(
            slide_name=slide_name,
            y_true=y_true,
            y_pred=y_pred,
        )

    if is_rank0:
        test_results.dump(config.TEST_RESULT_PATH)
        print(f'The testing results were saved as {config.TEST_RESULT_PATH}.')


def visualize_main(config_path: str) -> None:
    # Initialize and get the config
    worker_rank, num_workers, is_rank0 = _initialize_horovod()
    config = _get_config(config_path, is_rank0, TestConfig)
    _set_gpu_memory_limit(worker_rank, config.GPU_MEMORY_LIMIT_GB)

    # Initialize datasets
    if is_rank0:
        print('### Initializing datasets ###')
    test_subdatasets = [
        Dataset(
            csv_path=dataset_config.TEST_CSV_PATH,
            slide_dir=dataset_config.SLIDE_DIR,
            slide_file_extension=dataset_config.SLIDE_FILE_EXTENSION,
            contour_dir=dataset_config.CONTOUR_DIR,
            format='fit',
            resize_ratio=config.RESIZE_RATIO,
            snapshot_path=(config.DEBUG_PATH if is_rank0 else None),
        )
        for dataset_config in config.TEST_DATASET_CONFIGS
    ]
    test_dataset = ConcatDataset(datasets=test_subdatasets)
    if is_rank0:
        print(f'Test dataset contains {len(test_dataset)} slides.')

    # Initialize samplers
    if is_rank0:
        print('### Initializing samplers ###')
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=num_workers,
        rank=worker_rank,
        shuffle=False,
    )
    print(f'Test sampler contains {len(test_sampler)} slides on rank {worker_rank}.')

    if is_rank0:
        print('### Initializing the model ###')
    model_cam = Hms2ModelBuilder().build(
        n_classes=config.NUM_CLASSES,
        backbone=config.MODEL,
        pretrained=None,
        pre_pooling=config.PRE_POOLING,
        pooling=config.VIZ_POOL_USE,
        use_hms2=config.USE_HMS2,
        use_cpu_for_dense=True,
    )

    missing_keys, unexpected_keys = model_cam.load_state_dict(
        torch.load(config.MODEL_PATH, map_location='cpu'),
        strict=False,
    )
    if len(missing_keys) > 0:
        raise ValueError(f'Missing keys: {missing_keys}.')
    if len(unexpected_keys) > 0:
        print(f'Unexpected keys: {unexpected_keys}.')
    if is_rank0:
        print(f'The weight file {config.MODEL_PATH} is loaded.')
    if is_rank0:
        print(model_cam)

    # Determine the activation applied on predictions
    activation_fn = get_activation_fn(config.LOSS)

    # Initialize the event logger
    event_logger = None
    if is_rank0:
        event_logger = EventLogger(Path(config.VIZ_EVENT_LOG_PATH))

    # Start visualizing
    if is_rank0:
        print('### Generate visualization results ###')

    os.makedirs(config.VIZ_RESULT_DIR, exist_ok=True)

    model_cam.eval()
    progress_bar = tqdm(
        test_sampler,
        desc='visualize',
        disable=(not is_rank0),
        leave=False,
        file=sys.stdout,
    )
    for batch_idx, dataset_idx in enumerate(progress_bar):
        img, _ = test_dataset[dataset_idx]
        with torch.no_grad():
            img_batch = torch.tensor(img[np.newaxis, ...])

            if not config.USE_HMS2:
                # If HMS2 is disabled, the input should be manually move to GPU.
                img_batch = img_batch.cuda()

            cam = model_cam(img_batch)[0]
            cam = activation_fn(cam.detach().cpu().numpy())

        if (contours := test_dataset.get_contours(dataset_idx)) is not None:
            cam = make_invalid_nan(
                cam=cam,
                contours=contours,
                resize_ratio=(test_dataset.get_resize_ratio(dataset_idx) / get_backbone_stride(config.MODEL)),
            )
        slide_name = test_dataset.get_slide_name(dataset_idx)
        viz_path = Path(config.VIZ_RESULT_DIR) / f'{slide_name}.npy'
        np.save(viz_path, cam)

        if is_rank0:
            if event_logger is None:
                raise RuntimeError
            event_logger.append_and_write(
                VisualizeBatchEvent(
                    batch=batch_idx,
                    total_batches=len(test_sampler),
                ),
            )

    MPI.COMM_WORLD.Barrier()
    if is_rank0:
        print(f'The raw visualization data were saved in {config.VIZ_RESULT_DIR}.')
