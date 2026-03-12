from __future__ import annotations

import abc
import datetime
import typing as t
from pathlib import Path

import pydantic
from atomicwrites import atomic_write


class EventLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

        self.events = EventLog()

    def append_and_write(self, event: BaseEvent) -> None:
        self.events.__root__.append(event)
        with atomic_write(self.log_path, overwrite=True) as file:
            file.write(self.events.json(indent=4))


class BaseEvent(pydantic.BaseModel, metaclass=abc.ABCMeta):
    timestamp: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now,
    )


class TrainBatchEvent(BaseEvent):
    name: t.Literal['train_batch'] = 'train_batch'
    epoch: int
    batch: int
    total_batches: int


class TrainEpochEvent(BaseEvent):
    name: t.Literal['train_epoch'] = 'train_epoch'
    epoch: int
    loss: float
    metrics: t.Dict[str, t.Union[float, t.List[float]]]


class ValidationBatchEvent(BaseEvent):
    name: t.Literal['validation_batch'] = 'validation_batch'
    epoch: int
    batch: int
    total_batches: int


class ValidationEpochEvent(BaseEvent):
    name: t.Literal['validation_epoch'] = 'validation_epoch'
    epoch: int
    loss: float
    metrics: t.Dict[str, t.Union[float, t.List[float]]]


class TestBatchEvent(BaseEvent):
    name: t.Literal['test_batch'] = 'test_batch'
    batch: int
    total_batches: int


class VisualizeBatchEvent(BaseEvent):
    name: t.Literal['visualize_batch'] = 'visualize_batch'
    batch: int
    total_batches: int


class EventLog(pydantic.BaseModel):
    __root__: t.List[BaseEvent] = []
