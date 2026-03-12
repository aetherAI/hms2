import time
import typing as t
from pathlib import Path

import horovod.torch as hvd
import numpy as np
import torch
from torch import nn

from .event_logging import (
    EventLogger, TrainBatchEvent, TrainEpochEvent, ValidationBatchEvent, ValidationEpochEvent,
)
from .metrics import (
    AccuracyMetric, AUCMetric, BinaryAccuracyMetric,
    ConcordanceIndexMetric, Metric, MultiLabelAUCMetric,
)


LogsDict = t.Dict[str, t.Any]
StateDict = t.Dict[str, t.Any]


class Callback:
    def __init__(
        self,
        name: str,
        model: t.Optional[nn.Module] = None,
        optimizer: t.Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.name = name
        self.model = model
        self.optimizer = optimizer

    def on_train_epoch_begin(self, epoch_idx: int) -> None:
        pass

    def on_train_epoch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        pass

    def on_train_batch_begin(self, batch_idx: int) -> None:
        pass

    def on_train_batch_end(self, batch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        pass

    def on_validation_epoch_begin(self, epoch_idx: int) -> None:
        pass

    def on_validation_epoch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        pass

    def on_validation_batch_begin(self, batch_idx: int) -> None:
        pass

    def on_validation_batch_end(self, batch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        pass

    def state_dict(self) -> StateDict:
        return {}

    def load_state_dict(self, state_dict: StateDict) -> None:
        pass


class RollbackOnLRDecay(Callback):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        filepath: str,
        optimizer_state_filepath: t.Optional[str] = None,
        verbose: bool = False,
        name: str = 'rollback_on_lr_decay',
    ):
        super().__init__(model=model, optimizer=optimizer, name=name)
        self.filepath = filepath
        self.optimizer_state_filepath = optimizer_state_filepath
        self.verbose = verbose

        self.last_lrs = None

    def on_train_epoch_begin(self, epoch_idx: int) -> None:
        lrs = []
        if self.optimizer is None:
            raise RuntimeError
        param_groups = self.optimizer.param_groups
        for param_group in param_groups:
            lrs.append(param_group['lr'])

        if self.last_lrs is None:
            self.last_lrs = lrs
        elif np.all(np.array(self.last_lrs) == np.array(lrs)):
            self.last_lrs = lrs
        else:
            if self.model is None:
                raise RuntimeError
            self.model.load_state_dict(torch.load(self.filepath, map_location='cpu'))
            if self.verbose:
                print(f'The weight is rollbacked by loading {self.filepath}.')

            if self.optimizer_state_filepath is not None:
                # Rollback optimizer states but remain the lr and etc.
                old_state_dict = torch.load(self.optimizer_state_filepath, map_location='cpu')
                new_state_dict = self.optimizer.state_dict()
                new_state_dict['state'] = old_state_dict

                self.optimizer.load_state_dict(
                    new_state_dict,
                )
                if self.verbose:
                    print(f'The optimizer state is rollbacked by loading {self.optimizer_state_filepath}.')
            self.last_lrs = lrs

    def state_dict(self) -> StateDict:
        return {
            'last_lrs': self.last_lrs,
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        self.last_lrs = state_dict['last_lrs']


class EarlyStoppingOnLRDecay(Callback):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        verbose: bool = False,
        name: str = 'early_stopping_on_lr_decay',
    ):
        super().__init__(optimizer=optimizer, name=name)
        self.verbose = verbose

        self.last_lrs = None

    def on_train_epoch_begin(self, epoch_idx: int) -> None:
        lrs = []
        if self.optimizer is None:
            raise RuntimeError
        param_groups = self.optimizer.param_groups
        for param_group in param_groups:
            lrs.append(param_group['lr'])

        if self.last_lrs is None:
            self.last_lrs = lrs
        elif np.all(np.array(self.last_lrs) == np.array(lrs)):
            self.last_lrs = lrs
        else:
            if self.verbose:
                print('Early stopping is triggered.')
            raise SystemExit(0)

    def state_dict(self) -> StateDict:
        return {
            'last_lrs': self.last_lrs,
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        self.last_lrs = state_dict['last_lrs']


class ModelCheckpoint(Callback):
    def __init__(
        self,
        model: nn.Module,
        filepath: str,
        monitor: str,
        mode: str = 'min',
        save_best_only: bool = True,
        optimizer: t.Optional[torch.optim.Optimizer] = None,
        optimizer_state_filepath: t.Optional[str] = None,
        verbose: bool = False,
        name: str = 'model_checkpoint',
        is_lora_model: bool = False,
        history_folder: t.Optional[str] = None,
        history_filename: str = 'model_{epoch_idx}.pt',
    ):
        super().__init__(model=model, optimizer=optimizer, name=name)
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.optimizer_state_filepath = optimizer_state_filepath
        self.verbose = verbose
        self.is_lora_model = is_lora_model
        self.history_folder = history_folder
        self.history_filename = history_filename

        if mode not in ['min', 'max']:
            raise ValueError('mode should be either "min" or "max"')
        if (optimizer is None) != (optimizer_state_filepath is None):
            raise ValueError('Both optimizer and optimizer_state_filepath should be given or not given.')

        if self.history_folder is not None:
            Path(self.history_folder).mkdir(exist_ok=True)
        self.best_record = None

    def on_train_epoch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        # Save the model in history
        if self.history_folder is not None:
            if self.model is None:
                raise RuntimeError
            history_path = Path(self.history_folder) / self.history_filename.format(epoch_idx=epoch_idx)
            torch.save(
                self.model.state_dict(),
                history_path,
            )
            if self.verbose:
                print(f'Saving the model into history as {history_path}.')

        # Save the model
        if self.save_best_only:
            return

        self._save()

        if self.verbose:
            print(f'Saving the model to {self.filepath}.')
            if self.optimizer is not None:
                print(f'Saving the optimizer state to {self.optimizer_state_filepath}.')

    def on_validation_epoch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        if not self.save_best_only:
            return

        skip = None
        if logs is None or self.monitor not in logs:
            raise ValueError(f'No such key {self.monitor} in logs')

        score = logs[self.monitor]

        if self.mode == 'min':
            if self.best_record is None or score < self.best_record:
                self.best_record = score
                skip = False
            else:
                skip = True
        elif self.mode == 'max':
            if self.best_record is None or score > self.best_record:
                self.best_record = score
                skip = False
            else:
                skip = True
        else:
            raise ValueError('mode should be either "min" or "max"')

        if skip:
            return

        self._save()

        if self.verbose:
            print(
                f'Saving the model to {self.filepath} since the {self.monitor} improves to {self.best_record}.',
            )
            if self.optimizer is not None:
                print(f'Saving the optimizer state to {self.optimizer_state_filepath}.')

    def state_dict(self) -> StateDict:
        return {
            'best_record': self.best_record,
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        self.best_record = state_dict['best_record']

    def _save(self) -> None:
        if self.model is None:
            raise RuntimeError
        torch.save(self.model.state_dict(), self.filepath)
        if self.is_lora_model:
            # Save lora-adapters and tunable parameters only
            self.model.save_pretrained(str(Path(self.filepath).parent))

        if self.optimizer is not None:
            if self.optimizer_state_filepath is None:
                raise RuntimeError
            torch.save(self.optimizer.state_dict()['state'], self.optimizer_state_filepath)


class ContinueModule(Callback):
    def __init__(
        self,
        model: nn.Module,
        filepath: str,
        optimizer: t.Optional[torch.optim.Optimizer] = None,
        lr_scheduler: t.Optional[t.Any] = None,
        callbacks: t.Optional[t.Sequence['Callback']] = None,
        load_states: bool = False,
        save_states: bool = False,
        verbose: bool = False,
        name: str = 'continue_module',
    ):
        super().__init__(model=model, optimizer=optimizer, name=name)
        self.filepath = filepath
        self.lr_scheduler = lr_scheduler
        self.callbacks = callbacks
        self.load_states = load_states
        self.save_states = save_states
        self.verbose = verbose

        self.load_states_done = False

    def on_train_epoch_begin(self, epoch_idx: int) -> None:
        if not self.load_states:
            return
        if self.load_states_done:
            return

        state_dict = torch.load(self.filepath, map_location='cpu')
        if self.model is None:
            raise RuntimeError
        self.model.load_state_dict(state_dict['model'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.load_state_dict(state_dict[callback.name])
        self.load_state_dict(state_dict[self.name])

        if self.verbose:
            print(f'The states are resumed from {self.filepath}.')

        self.load_states_done = True

    def on_validation_epoch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        if not self.save_states:
            return

        state_dict = {}
        if self.model is None:
            raise RuntimeError
        state_dict['model'] = self.model.state_dict()
        if self.optimizer is not None:
            state_dict['optimizer'] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()

        if self.callbacks is not None:
            for callback in self.callbacks:
                if callback.name in state_dict:
                    raise NotImplementedError('Duplicated callbacks are not supported yet.')

                state_dict[callback.name] = callback.state_dict()

        state_dict[self.name] = self.state_dict()
        state_dict['epoch_idx'] = epoch_idx

        torch.save(state_dict, self.filepath)
        if self.verbose:
            print(f'Saving the states to {self.filepath}.')


class MetricHandler(Callback):
    def __init__(
        self,
        metric_list: t.Sequence[str],
        use_horovod: bool,
        verbose: bool,
        activation_fn: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
        name: str = 'metric_handler',
    ):
        super().__init__(name=name)
        self.verbose = verbose
        self.activation_fn = activation_fn
        self.metric_list = self._initialize_metric_list(metric_list)
        self.use_horovod = use_horovod

        self.y_true_list = None
        self.y_pred_list = None

    def _initialize_metric_list(
        self,
        metric_name_list: t.Sequence[str],
    ) -> t.Sequence['Metric']:
        metric_list = []
        for metric_name in metric_name_list:
            if metric_name == AccuracyMetric.NAME:
                metric = AccuracyMetric()
            elif metric_name == BinaryAccuracyMetric.NAME:
                metric = BinaryAccuracyMetric()
            elif metric_name == AUCMetric.NAME:
                metric = AUCMetric(
                    activation_fn=self.activation_fn,
                    verbose=self.verbose,
                )
            elif metric_name == MultiLabelAUCMetric.NAME:
                metric = MultiLabelAUCMetric(
                    activation_fn=self.activation_fn,
                    verbose=self.verbose,
                )
            elif metric_name == ConcordanceIndexMetric.NAME:
                metric = ConcordanceIndexMetric()
            else:
                raise RuntimeError(f'{metric_name} is not a supported metric.')
            metric_list.append(metric)

        return metric_list

    def _on_epoch_begin(self) -> None:
        self.y_true_list = []
        self.y_pred_list = []

    def _on_epoch_end(self, print_prefix: str, logs: t.Optional[LogsDict]) -> None:
        # Collect all data
        y_true_full_list = self.y_true_list
        y_pred_full_list = self.y_pred_list
        if self.use_horovod:
            y_true_full_list = hvd.allgather_object(y_true_full_list)
            y_true_full_list = [obj for obj_list in y_true_full_list for obj in obj_list]
            y_pred_full_list = hvd.allgather_object(y_pred_full_list)
            y_pred_full_list = [obj for obj_list in y_pred_full_list for obj in obj_list]
        y_true_full_list = np.concatenate(y_true_full_list, axis=0)
        y_pred_full_list = np.concatenate(y_pred_full_list, axis=0)

        # Calculate metrics
        metric_results = {}
        for metric in self.metric_list:
            metric_result = metric(y_pred_full_list, y_true_full_list)
            metric_results[metric.NAME] = metric_result

        # Print the metrics if verbose == True
        if self.verbose:
            print_str = print_prefix
            for metric_name, metric_result in metric_results.items():
                print_str += f'{metric_name}: {metric_result}, '
            print(print_str)

        # Update logs
        if logs is not None:
            logs['metrics'] = metric_results

    def _on_batch_end(self, logs: LogsDict) -> None:
        if self.y_true_list is None:
            raise RuntimeError
        if self.y_pred_list is None:
            raise RuntimeError
        self.y_true_list.append(logs['y_true'])
        self.y_pred_list.append(logs['y_pred'])

    def on_train_epoch_begin(self, epoch_idx: int) -> None:
        self._on_epoch_begin()

    def on_train_epoch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        self._on_epoch_end('Training metrics: ', logs)

    def on_train_batch_end(self, batch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        if logs is None:
            raise ValueError('logs cannot be empty')
        self._on_batch_end(logs)

    def on_validation_epoch_begin(self, epoch_idx: int) -> None:
        self._on_epoch_begin()

    def on_validation_epoch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        self._on_epoch_end('Validation metrics: ', logs)

    def on_validation_batch_end(self, batch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        if logs is None:
            raise ValueError('logs cannot be empty')
        self._on_batch_end(logs)


class EventLoggingCallback(Callback):
    def __init__(
        self,
        log_path: Path,
        name: str = 'event_logging',
    ) -> None:
        super().__init__(name=name)
        self.event_logger = EventLogger(log_path)

    def on_train_epoch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        if logs is None:
            raise ValueError('logs cannot be empty')

        event = TrainEpochEvent(
            epoch=epoch_idx,
            loss=logs['train_loss'],
            metrics=logs['metrics'],
        )
        self.event_logger.append_and_write(event)

    def on_train_batch_end(self, batch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        if logs is None:
            raise ValueError('logs cannot be empty')

        event = TrainBatchEvent(
            epoch=logs['epoch_idx'],
            batch=batch_idx,
            total_batches=logs['total_batches'],
        )
        self.event_logger.append_and_write(event)

    def on_validation_epoch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        if logs is None:
            raise ValueError('logs cannot be empty')

        event = ValidationEpochEvent(
            epoch=epoch_idx,
            loss=logs['val_loss'],
            metrics=logs['metrics'],
        )
        self.event_logger.append_and_write(event)

    def on_validation_batch_end(self, batch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        if logs is None:
            raise ValueError('logs cannot be empty')

        event = ValidationBatchEvent(
            epoch=logs['epoch_idx'],
            batch=batch_idx,
            total_batches=logs['total_batches'],
        )
        self.event_logger.append_and_write(event)


class DataBoundWarningCallback(Callback):
    def __init__(
        self,
        tolerance: float = 1.0,
        name: str = 'data_bound_warning',
    ) -> None:
        super().__init__(name=name)
        self.tolerance = tolerance

        self.last_time = None
        self.last_data_time = None
        self.last_compute_time = None

    def on_train_epoch_begin(self, epoch_idx: int) -> None:
        self.last_time = time.time()

    def on_train_batch_begin(self, batch_idx: int) -> None:
        if self.last_time is None:
            raise ValueError

        self.last_data_time = time.time() - self.last_time
        self.last_time = time.time()

    def on_train_batch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        if self.last_time is None or self.last_data_time is None:
            raise ValueError

        self.last_compute_time = time.time() - self.last_time
        self.last_time = time.time()

        if self.last_data_time / self.last_compute_time > self.tolerance:
            print(
                f'Data loading time ({self.last_data_time:.2f}) is too long compared to '
                f'compute time ({self.last_compute_time:.2f})',
            )


class CallbackList(Callback):
    def __init__(self, callback_list: t.Sequence['Callback'], name: str = 'callback_list'):
        super().__init__(name=name)
        self.callback_list = callback_list

    def on_train_epoch_begin(self, epoch_idx: int) -> None:
        for callback in self.callback_list:
            callback.on_train_epoch_begin(epoch_idx)

    def on_train_epoch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        for callback in self.callback_list:
            callback.on_train_epoch_end(epoch_idx, logs=logs)

    def on_train_batch_begin(self, batch_idx: int) -> None:
        for callback in self.callback_list:
            callback.on_train_batch_begin(batch_idx)

    def on_train_batch_end(self, batch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        for callback in self.callback_list:
            callback.on_train_batch_end(batch_idx, logs=logs)

    def on_validation_epoch_begin(self, epoch_idx: int) -> None:
        for callback in self.callback_list:
            callback.on_validation_epoch_begin(epoch_idx)

    def on_validation_epoch_end(self, epoch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        for callback in self.callback_list:
            callback.on_validation_epoch_end(epoch_idx, logs=logs)

    def on_validation_batch_begin(self, batch_idx: int) -> None:
        for callback in self.callback_list:
            callback.on_validation_batch_begin(batch_idx)

    def on_validation_batch_end(self, batch_idx: int, logs: t.Optional[LogsDict] = None) -> None:
        for callback in self.callback_list:
            callback.on_validation_batch_end(batch_idx, logs=logs)

    def state_dict(self) -> StateDict:
        state_dict = {}
        for callback in self.callback_list:
            if callback.name in state_dict:
                raise NotImplementedError('Duplicated callbacks are not supported yet.')

            state_dict[callback.name] = callback.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: StateDict) -> None:
        for callback in self.callback_list:
            if callback.name not in state_dict:
                raise RuntimeError(f'{callback.name} in the CallbackList is not in the state_dict')

            callback.load_state_dict(state_dict[callback.name])
