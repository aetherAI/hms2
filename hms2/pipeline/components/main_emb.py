import gc
import itertools
import math
import os
import shutil
import sys
import typing as t
import warnings
from concurrent.futures import ThreadPoolExecutor
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
from ...core.compressors import BaseCompressor, get_compressor
from .callbacks import (
    Callback, CallbackList, ContinueModule, EarlyStoppingOnLRDecay,
    MetricHandler, ModelCheckpoint, RollbackOnLRDecay,
)
from .config import (
    BackbonePretrainOption, Hms2PretrainOption, NoPretrainOption,
    TestConfig, TorchvisionPretrainOption, TrainConfig,
)
from .dataset import ConcatDataset, Dataset, DistributedWeightedSampler, EmbeddingDataset
from .event_logging import EventLogger, TestBatchEvent
from .losses import get_activation_fn, get_loss_fn
from .optimizers import get_optimizer
from .saver import get_saver
from .utils import TestResults, remove_unmatched_from_state_dict


warnings.filterwarnings('ignore', message='TypedStorage is deprecated')


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

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    if is_rank0:
        print('### Config ###')
        devtools.debug(config)

        try:
            shutil.copyfile(config_path, config.CONFIG_RECORD_PATH)
        except shutil.SameFileError:
            pass

    return config


def _validate_config(config: t.Union[TrainConfig, TestConfig]) -> None:
    if not config.USE_EMBED_MODE:
        raise RuntimeError('USE_EMBED_MODE must be true.')

    if not config.EMBED_MODE_CACHE_DIR:
        raise RuntimeError('EMBED_MODE_CACHE_DIR must be set for embedding mode.')

    if not config.USE_HMS2:
        raise RuntimeError('USE_HMS2 must be set for embedding mode.')

    if config.USE_LORA_FINETUNE:
        raise RuntimeError('LORA is not supported for embedding mode.')

    if config.USE_MIXED_PRECISION:
        raise RuntimeError('Mixed precision is not supported for embedding mode.')

    if config.BATCH_SIZE != 1:
        raise RuntimeError('BATCH_SIZE must be set to 1 for embedding mode.')


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


def _get_pretrained_weights(
    is_rank0: bool,
    config: _ConfigType,
) -> t.Union[None, t.OrderedDict[str, torch.Tensor], str]:
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
    return pretrained


def _compress_and_save(
    output: t.Dict[str, torch.Tensor],
    compressors: t.List[BaseCompressor],
    saver: t.Any,
    path: str,
):
    for compressor in compressors:
        output = compressor.compress(output)

    saver.save(
        embed=output,
        path=path,
    )


def extract_emb_main(config_path: str, continue_mode: bool, extract_mode='train') -> None:  # noqa: C901
    # Initialize and get the config
    worker_rank, num_workers, is_rank0 = _initialize_horovod()
    config_cls = TrainConfig if extract_mode == 'train' else TestConfig
    config = _get_config(config_path, is_rank0, config_cls)
    _validate_config(config)

    _set_gpu_memory_limit(worker_rank, config.GPU_MEMORY_LIMIT_GB)

    # Check the number of GPUs
    if config.GPUS is not None:
        if num_workers != config.GPUS:
            raise ValueError(f'{config.GPUS} GPUs are required, but we got {num_workers}.')

    # Initialize datasets
    if is_rank0:
        print('### Initializing datasets ###')
    embedding_dir = config.EMBED_MODE_CACHE_DIR if continue_mode else None
    if extract_mode == 'train':
        train_subdatasets = [
            Dataset(
                csv_path=dataset_config.TRAIN_CSV_PATH,
                slide_dir=dataset_config.SLIDE_DIR,
                slide_file_extension=dataset_config.SLIDE_FILE_EXTENSION,
                contour_dir=dataset_config.CONTOUR_DIR,
                format='trim',
                resize_ratio=config.RESIZE_RATIO,
                snapshot_path=(config.DEBUG_PATH if is_rank0 else None),
                embedding_dir=embedding_dir,
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
                embedding_dir=embedding_dir,
            )
            for dataset_config in config.TRAIN_DATASET_CONFIGS
            if dataset_config.VAL_CSV_PATH is not None
        ]
        val_dataset = ConcatDataset(datasets=val_subdatasets) if len(val_subdatasets) > 0 else None
        if is_rank0:
            if val_dataset is None:
                print('No validation dataset.')
            else:
                print(f'validation dataset contains {len(val_dataset)} slides.')
        datasets = [train_dataset, val_dataset]
        dataset_names = ['Train', 'Val']
    elif extract_mode == 'test':
        test_subdatasets = [
            Dataset(
                csv_path=dataset_config.TEST_CSV_PATH,
                slide_dir=dataset_config.SLIDE_DIR,
                slide_file_extension=dataset_config.SLIDE_FILE_EXTENSION,
                contour_dir=dataset_config.CONTOUR_DIR,
                format='trim',
                resize_ratio=config.RESIZE_RATIO,
                snapshot_path=(config.DEBUG_PATH if is_rank0 else None),
                embedding_dir=embedding_dir,
            )
            for dataset_config in config.TEST_DATASET_CONFIGS
        ]
        test_dataset = ConcatDataset(datasets=test_subdatasets)
        if is_rank0:
            if test_dataset is None:
                print('No test dataset.')
            else:
                print(f'Test dataset contains {len(test_dataset)} slides.')
        datasets = [test_dataset]
        dataset_names = ['Test']
    else:
        raise ValueError(f'Invalid extract_mode: {extract_mode}')

    # Initialize dataloaders.
    if is_rank0:
        print('### Initializing dataloaders ###')

    samplers, dataloaders = [], []
    for name, dataset in zip(dataset_names, datasets):
        sampler, dataloader = None, None
        if dataset:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=num_workers,
                rank=worker_rank,
                shuffle=False,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                sampler=sampler,
                num_workers=1,
                prefetch_factor=1,
                persistent_workers=False,
            )
            print(f'{name} dataloader contains {len(dataloader)} slides on rank {worker_rank}.')
        samplers.append(sampler)
        dataloaders.append(dataloader)

    # Initialize the model
    if is_rank0:
        print('### Initializing the model ###')

    pretrained = _get_pretrained_weights(is_rank0, config)

    model = Hms2ModelBuilder().build(
        n_classes=config.NUM_CLASSES,
        augmentation_list=[],
        backbone=config.MODEL,
        pretrained=pretrained,
        pre_pooling=config.EMBED_MODE_GPU_COMPRESSOR,
        pooling='no',
        custom_dense='no',
        use_hms2=config.USE_HMS2,
        use_cpu_for_dense=True,
    )
    if is_rank0 and isinstance(config.PRETRAINED, Hms2PretrainOption):
        pretrained = torch.load(
            config.PRETRAINED.path,
            map_location='cpu',
        )
        pretrained = remove_unmatched_from_state_dict(pretrained, model)
        model.load_state_dict(pretrained, strict=False)

    compressors: t.List[BaseCompressor] = [get_compressor(c) for c in config.EMBED_MODE_COMPRESSORS]
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    if is_rank0:
        print(model)

    saver = get_saver('numpy')
    os.makedirs(config.EMBED_MODE_CACHE_DIR, exist_ok=True)

    with scoping():
        model.eval()
        for name, loader, sampler in zip(dataset_names, dataloaders, samplers):
            if not loader:
                continue

            if is_rank0:
                print(f'### Start extracting {name} dataset ###')

            dataset = loader.dataset
            progress_bar = tqdm(
                loader,
                desc=f'Extract {name}',
                disable=(not is_rank0),
                leave=False,
                file=sys.stdout,
            )
            last_slide_idx, indices = -1, list(iter(sampler))
            with ThreadPoolExecutor() as executor:
                previous_task = None
                for data_idx, data in enumerate(progress_bar):
                    img_batch, _ = data
                    output = {}
                    slide_idx = indices[data_idx]
                    slide_name = dataset.get_slide_name(slide_idx)

                    if slide_idx < last_slide_idx:
                        """
                        DistrubutedSampler without shuffling will output the indices in round-robin order.
                        This means that if the current slide index is less than the last processed slide index,
                        it must be a re-processed slide.
                        """
                        continue

                    with torch.no_grad():
                        embed = model(img_batch)
                        output['embed'] = embed[0]  # Batch size is always 1 in embedding mode
                        path = os.path.join(config.EMBED_MODE_CACHE_DIR, f'{slide_name}')
                        if previous_task is not None:
                            previous_task.result()
                        executor.submit(
                            _compress_and_save,
                            output=output,
                            compressors=compressors,
                            saver=saver,
                            path=path,
                        )

                    last_slide_idx = slide_idx

                if previous_task is not None:
                    previous_task.result()

            gc.collect()


def train_main_embed_mode(  # noqa: C901
    config: TrainConfig,
    continue_mode: bool,
    worker_rank: int,
    num_workers: int,
    is_rank0: bool,
) -> None:
    # Initialize and get the config
    _validate_config(config)

    # Check the number of GPUs
    if config.GPUS is not None:
        if num_workers != config.GPUS:
            raise ValueError(f'{config.GPUS} GPUs are required, but we got {num_workers}.')

    # Initialize datasets
    if is_rank0:
        print('### Initializing datasets ###')

    saver = get_saver('numpy')
    train_subdatasets = [
        EmbeddingDataset(
            csv_path=dataset_config.TRAIN_CSV_PATH,
            slide_dir=dataset_config.SLIDE_DIR,
            slide_file_extension=dataset_config.SLIDE_FILE_EXTENSION,
            contour_dir=dataset_config.CONTOUR_DIR,
            embed_dir=config.EMBED_MODE_CACHE_DIR,
            saver=saver,
        )
        for dataset_config in config.TRAIN_DATASET_CONFIGS
    ]
    train_dataset = ConcatDataset(datasets=train_subdatasets)

    if is_rank0:
        print(f'Training dataset contains {len(train_dataset)} slides.')

    val_subdatasets = [
        EmbeddingDataset(
            csv_path=dataset_config.VAL_CSV_PATH,
            slide_dir=dataset_config.SLIDE_DIR,
            slide_file_extension=dataset_config.SLIDE_FILE_EXTENSION,
            contour_dir=dataset_config.CONTOUR_DIR,
            embed_dir=config.EMBED_MODE_CACHE_DIR,
            saver=saver,
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
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
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
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )
        print(f'Val dataloader contains {len(val_dataloader)} slides on rank {worker_rank}.')
    else:
        val_sampler = None
        val_dataloader = None

    # Initialize the model
    if is_rank0:
        print('### Initializing the model ###')

    pretrained = _get_pretrained_weights(is_rank0, config)
    model = Hms2ModelBuilder().build_embedding(
        n_classes=config.NUM_CLASSES,
        backbone=config.MODEL,
        pretrained=pretrained,
        pre_pooling=config.PRE_POOLING,
        pooling=config.POOL_USE,
        custom_dense=config.CUSTOM_DENSE,
        compressors=config.EMBED_MODE_COMPRESSORS,
    )

    if is_rank0 and isinstance(config.PRETRAINED, Hms2PretrainOption):
        pretrained = torch.load(
            config.PRETRAINED.path,
            map_location='cpu',
        )
        pretrained = remove_unmatched_from_state_dict(pretrained, model)
        model.load_state_dict(pretrained, strict=False)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    if is_rank0:
        print(model)

    # Initialize the loss and activation function
    loss_fn = get_loss_fn(config.LOSS)
    activation_fn = get_activation_fn(config.LOSS)

    # Initialize the optimizer
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
            model.train()

            losses = []
            progress_bar = tqdm(
                train_dataloader,
                desc='train',
                disable=(not is_rank0),
                leave=False,
                file=sys.stdout,
            )
            for batch_idx, (data, y_true) in enumerate(progress_bar):
                callback_list_object.on_train_batch_begin(batch_idx)

                step_success = False
                loss = None
                y_pred = None
                while not step_success:
                    optimizer.zero_grad()
                    with torch.autocast('cuda', enabled=False):
                        y_pred = model.forward_embedding(data)
                        y_true = y_true.cuda()
                        loss = loss_fn.cuda()(y_pred, y_true)

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
            for batch_idx, (data, y_true) in enumerate(progress_bar):
                callback_list_object.on_validation_batch_begin(batch_idx)

                with torch.no_grad():
                    y_pred = model.forward_embedding(data)

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


def test_emb_main(config_path: str) -> None:
    # Initialize and get the config
    worker_rank, num_workers, is_rank0 = _initialize_horovod()
    config = _get_config(config_path, is_rank0, TestConfig)
    _validate_config(config)

    # Create results folder
    os.makedirs(os.path.dirname(config.TEST_RESULT_PATH), exist_ok=True)

    # Initialize datasets
    saver = get_saver('numpy')
    if is_rank0:
        print('### Initializing datasets ###')
    test_subdatasets = [
        EmbeddingDataset(
            csv_path=dataset_config.TEST_CSV_PATH,
            slide_dir=dataset_config.SLIDE_DIR,
            slide_file_extension=dataset_config.SLIDE_FILE_EXTENSION,
            contour_dir=dataset_config.CONTOUR_DIR,
            embed_dir=config.EMBED_MODE_CACHE_DIR,
            saver=saver,
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

    model = Hms2ModelBuilder().build_embedding(
        n_classes=config.NUM_CLASSES,
        backbone=config.MODEL,
        pretrained=None,
        pre_pooling=config.PRE_POOLING,
        pooling=config.POOL_USE,
        custom_dense=config.CUSTOM_DENSE,
        compressors=config.EMBED_MODE_COMPRESSORS,
    )

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
        data, _ = test_dataset[dataset_idx]
        data: t.Dict[str, torch.Tensor] = data

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].unsqueeze(0)  # Add batch dimension
        y_pred = model.forward_embedding(data)[0]
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
