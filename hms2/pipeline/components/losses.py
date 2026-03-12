from __future__ import annotations

import typing as t

import numpy as np
import numpy.typing as npt
import pydantic
import scipy.special
import torch
import torch.nn.functional
import torchvision.ops
from torch import nn


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class LossConfig(pydantic.BaseModel):
    name: str
    arguments: dict[str, t.Any] = pydantic.Field(default={})

    @classmethod
    def from_loss_name(cls: type[LossConfig], loss_name: str) -> LossConfig:
        # For backward compatibility
        if loss_name == 't_forward_stomach_lymphnode':
            loss_config = LossConfig(
                name='ce',
                arguments={
                    't_matrix': [
                        [0.99, 0.01],
                        [0.02, 0.98],
                    ],
                },
            )
        elif loss_name == 't_forward_gastric_biopsy_tumor':
            loss_config = LossConfig(
                name='ce',
                arguments={
                    't_matrix': [
                        [1.0, 0.0],
                        [0.02, 0.98],
                    ],
                },
            )
        else:
            loss_config = LossConfig(name=loss_name)

        return loss_config


def get_loss_fn(loss_config: LossConfig | str | None) -> nn.Module:
    if isinstance(loss_config, str):
        loss_config = LossConfig.from_loss_name(loss_config)

    # Get the loss function
    if loss_config is None:
        raise ValueError('A loss function should be set.')
    elif loss_config.name == 'ce':
        loss_fn = CustomizedCrossEntropyLoss(**loss_config.arguments)
    elif loss_config.name == 'bce':
        loss_fn = BCEWithLogitsLoss(**loss_config.arguments)
    elif loss_config.name == 'negative_partial_log_likelihood_for_cox_ph':
        loss_fn = NegativePartialLogLikelihoodForCoxPH(use_horovod=True)
    elif loss_config.name == 'ranknet_loss':
        loss_fn = RanknetLoss(use_horovod=True)
    else:
        raise NotImplementedError(f'{loss_config.name} is not a supported loss.')

    return loss_fn


def get_activation_fn(
    loss_config: LossConfig | str | None,
) -> t.Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]:
    if loss_config is None:
        activation_fn = lambda x: x
        return activation_fn

    if isinstance(loss_config, str):
        loss_config = LossConfig.from_loss_name(loss_config)

    if loss_config.name in [
        'ce',
    ]:

        def activation_fn(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            return scipy.special.softmax(x, axis=-1)

    elif loss_config.name in [
        'bce',
        'negative_partial_log_likelihood_for_cox_ph',
        'ranknet_loss',
    ]:
        activation_fn = scipy.special.expit
    else:
        raise NotImplementedError(f'{loss_config.name} is not a supported loss.')

    return activation_fn


def get_label_type(loss_config: LossConfig | str | None) -> str:
    if loss_config is None:
        raise ValueError('A loss function should be set.')

    if isinstance(loss_config, str):
        loss_config = LossConfig.from_loss_name(loss_config)

    if loss_config.name in [
        'ce',
    ]:
        label_type = 'multi_class'
    elif loss_config.name in [
        'bce',
        'negative_partial_log_likelihood_for_cox_ph',
        'ranknet_loss',
    ]:
        label_type = 'multi_label'
    else:
        raise NotImplementedError(f'{loss_config.name} is not a supported loss.')
    return label_type


class CustomizedCrossEntropyLoss(nn.Module):
    t_matrix: torch.Tensor | None
    balanced_softmax_term: torch.Tensor | None

    def __init__(
        self,
        t_matrix: list[list[float]] | None = None,
        balanced_softmax: list[float] | None = None,
    ):
        super().__init__()
        if t_matrix is None:
            self.t_matrix = None
        else:
            self.register_buffer(
                't_matrix',
                torch.tensor(t_matrix, dtype=torch.float),
            )

        if balanced_softmax is None:
            self.balanced_softmax_term = None
        else:
            class_samples = np.array(balanced_softmax)
            balanced_softmax_term = np.log(class_samples / class_samples.min())
            self.register_buffer(
                'balanced_softmax_term',
                torch.tensor(balanced_softmax_term, dtype=torch.float),
            )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.balanced_softmax_term is None:
            balanced_input = input
        else:
            balanced_input = input + self.balanced_softmax_term.to(input.device)

        input_norm = balanced_input - torch.max(balanced_input, dim=-1, keepdim=True)[0].detach()
        exp = torch.exp(input_norm)
        sumexp = torch.sum(exp, dim=-1, keepdim=True)

        if self.t_matrix is None:
            corrected = exp
        else:
            corrected = torch.matmul(exp, self.t_matrix.to(input.device))

        log_preds = torch.log(corrected / sumexp)
        loss = nn.functional.nll_loss(log_preds, target)
        return loss


class NegativePartialLogLikelihoodForCoxPH(nn.Module):
    def __init__(self, use_horovod: bool):
        super().__init__()
        self.use_horovod = use_horovod

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): The predicted risk scores of the shape [N, 1].
            target (torch.Tensor):
                A [N, 2] tensor with the survival statuses (0 for death, 1 for alive)
                and the last observed times.
        """
        # Shape checking
        if input.shape[1] != 1:
            raise RuntimeError(f'The input shape {input} is invalid.')
        if target.shape[1] != 2:
            raise RuntimeError(f'The target shape {target} is invalid.')

        # Aliasing
        predicted_risks = input[:, 0].contiguous()
        survival_statuses = target[:, 0].contiguous()
        last_observed_times = target[:, 1].contiguous()

        # If use_horovod, exchange data with the other workers.
        if self.use_horovod:
            predicted_risks = hvd.allgather(predicted_risks)
            survival_statuses = hvd.allgather(survival_statuses)
            last_observed_times = hvd.allgather(last_observed_times)

        # Calculate the loss
        batch_size = predicted_risks.shape[0]
        device = predicted_risks.device
        negative_partial_log_likelihoods = torch.where(
            survival_statuses == 0,
            (
                -predicted_risks
                + torch.logsumexp(
                    torch.where(
                        (last_observed_times[:, np.newaxis] <= last_observed_times[np.newaxis, :]),
                        predicted_risks[np.newaxis, :],
                        torch.full(
                            size=[batch_size, batch_size],
                            fill_value=(-np.inf),
                            device=device,
                        ),
                    ),
                    dim=-1,
                )
            ),
            torch.zeros(batch_size, device=device),
        )
        reduced_loss = torch.mean(negative_partial_log_likelihoods)

        return reduced_loss


class RanknetLoss(nn.Module):
    def __init__(self, use_horovod: bool):
        super().__init__()
        self.use_horovod = use_horovod

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): The predicted risk scores of the shape [N, 1].
            target (torch.Tensor):
                A [N, 2] tensor with the survival statuses (0 for death, 1 for alive)
                and the last observed times.
        """
        # Shape checking
        if input.shape[1] != 1:
            raise RuntimeError(f'The input shape {input} is invalid.')
        if target.shape[1] != 2:
            raise RuntimeError(f'The target shape {target} is invalid.')

        # Aliasing
        predicted_risks = input[:, 0].contiguous()
        survival_statuses = target[:, 0].contiguous()
        last_observed_times = target[:, 1].contiguous()

        # If use_horovod, exchange data with the other workers.
        if self.use_horovod:
            predicted_risks = hvd.allgather(predicted_risks)
            survival_statuses = hvd.allgather(survival_statuses)
            last_observed_times = hvd.allgather(last_observed_times)

        # Calculate the loss
        pairwise_higher_risk_than = torch.logical_and(
            (survival_statuses == 0)[:, np.newaxis],
            (last_observed_times[:, np.newaxis] <= last_observed_times[np.newaxis, :]),
        )  # pairwise_higher_risk_than[i, j]: i-th is more risky than j-th.
        pairwise_negative_log_likelihood = -torch.nn.functional.logsigmoid(
            (predicted_risks[:, np.newaxis] - predicted_risks[np.newaxis, :]),
        )
        pairwise_ranknet_loss = torch.where(
            pairwise_higher_risk_than,
            pairwise_negative_log_likelihood,
            torch.zeros_like(
                pairwise_negative_log_likelihood,
                device=pairwise_negative_log_likelihood.device,
            ),
        )
        ranknet_loss = torch.mean(pairwise_ranknet_loss)

        return ranknet_loss


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    ancestor_matrix: torch.Tensor | None
    balanced_sigmoid_term: torch.Tensor | None

    def __init__(
        self,
        tree_min_loss_parents: list[int | None] | None = None,
        balanced_sigmoid: list[float] | None = None,
        focal_loss_gamma: float | None = None,
    ):
        super().__init__()

        if tree_min_loss_parents is not None:
            self.register_buffer(
                'ancestor_matrix',
                self._calculate_ancestor_matrix(tree_min_loss_parents),
            )
        else:
            self.ancestor_matrix = None

        if balanced_sigmoid is None:
            self.balanced_sigmoid_term = None
        else:
            balanced_sigmoid_np = np.array(balanced_sigmoid)
            balanced_sigmoid_term = np.log(balanced_sigmoid_np) - np.log(1.0 - balanced_sigmoid_np)
            self.register_buffer(
                'balanced_sigmoid_term',
                torch.tensor(balanced_sigmoid_term, dtype=torch.float),
            )

        self.focal_loss_gamma = focal_loss_gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.balanced_sigmoid_term is not None:
            input = input + self.balanced_sigmoid_term.to(input.device)
        if self.ancestor_matrix is not None:
            input = self._calculate_tree_min_logits(input, target)

        dont_care_mask = torch.isnan(target)
        masked_target = torch.where(
            dont_care_mask,
            torch.zeros_like(target),
            target,
        )

        if self.focal_loss_gamma is not None:
            losses_flat = torchvision.ops.sigmoid_focal_loss(
                inputs=input,
                targets=masked_target,
                alpha=-1,
                gamma=self.focal_loss_gamma,
                reduction='none',
            )
        else:
            losses_flat = torch.nn.functional.binary_cross_entropy_with_logits(
                input=input,
                target=masked_target,
                reduction='none',
            )

        losses_masked = torch.where(
            dont_care_mask,
            torch.zeros_like(losses_flat),
            losses_flat,
        )

        if self.reduction == 'mean':
            loss = losses_masked.mean()
        elif self.reduction == 'sum':
            loss = losses_masked.sum()
        elif self.reduction == 'none':
            loss = losses_masked
        else:
            raise ValueError(f'Invalid reduction: {self.reduction}')

        return loss

    def _calculate_tree_min_logits(
        self,
        logits: torch.Tensor,  # [N, C]
        target: torch.Tensor,  # [N, C]
    ) -> torch.Tensor:
        batch_size, channels = logits.shape
        min_ancestors, _ = torch.min(
            torch.where(
                self.ancestor_matrix.to(logits.device).broadcast_to(batch_size, channels, channels),
                logits[:, np.newaxis, :],
                np.inf,
            ),
            axis=-1,
        )  # [N, C]
        descendant_matrix = self.ancestor_matrix.to(logits.device).transpose(0, 1)
        max_descendants, _ = torch.max(
            torch.where(
                descendant_matrix.broadcast_to(batch_size, channels, channels),
                logits[:, np.newaxis, :],
                -np.inf,
            ),
            axis=-1,
        )  # [N, C]
        tree_min_logits = torch.where(
            target == 1.0,
            min_ancestors,
            max_descendants,
        )
        return tree_min_logits

    def _calculate_ancestor_matrix(self, parents: list[int | None]) -> torch.Tensor:
        num_classes = len(parents)
        ancestor_matrix = []
        for class_index in range(num_classes):
            ancestor_list = [index == class_index for index in range(num_classes)]
            next_class = parents[class_index]
            while next_class is not None:
                ancestor_list[next_class] = True
                next_class = parents[next_class]

            ancestor_matrix.append(ancestor_list)

        ancestor_matrix = torch.tensor(ancestor_matrix)
        return ancestor_matrix
