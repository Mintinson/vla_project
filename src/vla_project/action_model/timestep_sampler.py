"""Timestep sampling strategies for diffusion model training.

This module provides various sampling strategies for selecting timesteps during
diffusion model training. Different sampling strategies can reduce variance in
the training objective and improve convergence.

The module includes uniform sampling, loss-aware sampling that adapts based on
per-timestep losses, and factory functions for creating samplers.

Modified from OpenAI's diffusion repositories:
    GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
    ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
    IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
"""

from abc import ABC, abstractmethod
from typing import Protocol, cast

import numpy as np
import torch
import torch.distributed as dist


class TimeStepDiffusion(Protocol):
    num_timesteps: int


def create_named_schedule_sampler(name: str, diffusion: TimeStepDiffusion) -> "ScheduleSampler":
    """Create a ScheduleSampler from a library of pre-defined samplers.

    Factory function that creates timestep samplers by name. Provides a convenient
    interface for instantiating different sampling strategies.

    Args:
        name (str): The name of the sampler to create. Supported options:
            - "uniform": Uniform sampling across all timesteps
            - "loss-second-moment": Loss-aware sampling based on second moments
        diffusion: The diffusion object to sample timesteps for.

    Returns:
        ScheduleSampler: The created sampler instance.

    Raises:
        NotImplementedError: If the specified sampler name is not recognized.

    Example:
        >>> sampler = create_named_schedule_sampler("uniform", diffusion)
        >>> timesteps, weights = sampler.sample(batch_size=32, device="cuda")

    """
    if name == "uniform":
        return UniformSampler(diffusion)
    if name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    msg = f"unknown schedule sampler: {name}"
    raise NotImplementedError(msg)


class ScheduleSampler(ABC):
    """A distribution over timesteps in the diffusion process, intended to reduce variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.

    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.

    This abstract base class defines the interface for timestep sampling strategies
    used during diffusion model training. Different strategies can help reduce
    variance in the training objective and improve convergence.
    """

    @abstractmethod
    def weights(self) -> np.ndarray:
        """Get a numpy array of weights, one per diffusion step.

        Returns:
            np.ndarray: Array of weights with shape (num_timesteps,).
                The weights needn't be normalized, but must be positive.

        """

    def sample(self, batch_size: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Importance-sample timesteps for a batch.

        Args:
            batch_size (int): The number of timesteps to sample.
            device (str | torch.device): The torch device to save tensors to.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - timesteps: A tensor of timestep indices with shape (batch_size,).
                - weights: A tensor of importance weights to scale the resulting
                  losses with shape (batch_size,).

        """
        w = self.weights()
        p = w / np.sum(w)
        rng = np.random.Generator(np.random.PCG64())
        indices_np = rng.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    """Uniform timestep sampler that assigns equal probability to all timesteps.

    This sampler provides uniform sampling across all diffusion timesteps,
    giving each timestep equal weight during training. This is the simplest
    sampling strategy and serves as a baseline.

    Attributes:
        diffusion: The diffusion process object.
        _weights (np.ndarray): Uniform weights array for all timesteps.

    """

    def __init__(self, diffusion: TimeStepDiffusion) -> None:
        """Initialize the uniform sampler.

        Args:
            diffusion: The diffusion process object containing num_timesteps.

        """
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self) -> np.ndarray:
        """Get uniform weights for all timesteps.

        Returns:
            np.ndarray: Array of ones with shape (num_timesteps,).

        """
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """Abstract base class for loss-aware timestep sampling strategies.

    This class provides the framework for samplers that adapt their timestep
    selection based on observed training losses. It includes distributed
    training support for synchronizing loss statistics across multiple processes.

    Subclasses should implement the update_with_all_losses method to define
    how loss observations update the sampling weights.
    """

    def update_with_local_losses(
        self,
        local_ts: torch.Tensor,
        local_losses: torch.Tensor,
    ) -> None:
        """Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        Args:
            local_ts (torch.Tensor): An integer tensor of timesteps.
            local_losses (torch.Tensor): A 1D tensor of losses corresponding to local_ts.

        Note:
            This method uses distributed communication to gather losses from all
            processes and maintain consistent reweighting across ranks.

        """
        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_ts.device) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = cast("int", max(batch_sizes))

        timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            cast("int", x.item()) for y, bs in zip(timestep_batches, batch_sizes, strict=False) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes, strict=False) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts: list[int], losses: list[float]) -> None:
        """Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        Args:
            ts (list[int]): A list of int timesteps.
            losses (list[float]): A list of float losses, one per timestep.

        """


class LossSecondMomentResampler(LossAwareSampler):
    """Loss-aware sampler that weights timesteps by the second moment of their losses.

    This sampler tracks the history of losses for each timestep and weights them
    according to their root mean squared (RMS) loss values. Timesteps with higher
    variance in losses are sampled more frequently, which can help reduce overall
    training variance.

    The sampler maintains a rolling history of losses for each timestep and uses
    a small uniform probability to ensure all timesteps are occasionally sampled.

    Attributes:
        diffusion: The diffusion process object.
        history_per_term (int): Number of loss values to maintain per timestep.
        uniform_prob (float): Probability of uniform sampling (vs. weighted sampling).
        _loss_history (np.ndarray): Array storing loss history for each timestep.
        _loss_counts (np.ndarray): Count of observed losses for each timestep.

    """

    def __init__(
        self,
        diffusion: TimeStepDiffusion,
        history_per_term: int = 10,
        uniform_prob: float = 0.001,
    ) -> None:
        """Initialize the loss second moment resampler.

        Args:
            diffusion: The diffusion process object containing num_timesteps.
            history_per_term (int, optional): Number of loss values to maintain
                for each timestep. Defaults to 10.
            uniform_prob (float, optional): Probability of uniform sampling to
                ensure all timesteps are occasionally visited. Defaults to 0.001.

        """
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term],
            dtype=np.float64,
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=int)

    def weights(self) -> np.ndarray:
        """Get weights based on the second moment of loss history.

        Returns uniform weights if not warmed up, otherwise returns weights
        proportional to the RMS of historical losses with uniform probability
        mixed in.

        Returns:
            np.ndarray: Array of sampling weights with shape (num_timesteps,).

        """
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts: list[int], losses: list[float]) -> None:
        """Update loss history with new observations.

        Updates the rolling history of losses for each timestep. When the history
        buffer is full, old losses are shifted out to make room for new ones.

        Args:
            ts (list[int]): List of timestep indices.
            losses (list[float]): List of corresponding loss values.

        """
        for t, loss in zip(ts, losses, strict=False):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self) -> bool:
        """Check if the sampler has collected enough loss history.

        Returns:
            bool: True if all timesteps have at least history_per_term loss observations.

        """
        return (self._loss_counts == self.history_per_term).all()
