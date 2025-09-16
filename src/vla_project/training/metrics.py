"""Training metrics tracking and logging utilities.

This module provides interfaces and implementations for tracking training metrics
across different logging backends including JSON Lines and Weights & Biases.
"""

import time
from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import jsonlines
import numpy as np
import torch
import wandb

from vla_project.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Define Tracker Interface ===
class Tracker(Protocol):
    """Protocol for metric tracking backends.

    Defines the interface that all metric trackers must implement for
    consistent logging across different backends.
    """

    def write_hyperparameters(self) -> None:
        """Write hyperparameters to the tracking backend."""
        ...

    def write(self, global_step: int, metrics: dict[str, int | float]) -> None:
        """Write metrics for a given global step.

        Args:
            global_step (int): Current training step.
            metrics (dict[str, int | float]): Dictionary of metric names to values.

        """
        ...

    def finalize(self) -> None:
        """Finalize and cleanup the tracker."""
        ...


# === Individual Tracker Definitions ===
class JSONLinesTracker:
    """JSON Lines file-based metric tracker.

    Logs hyperparameters and metrics to JSON Lines files for persistent storage
    and later analysis.

    Attributes:
        run_id (str): Unique identifier for the training run.
        run_dir (Path): Directory where logs will be stored.
        hparams (dict[str , Any]): Hyperparameters to log.

    """

    def __init__(self, run_id: str, run_dir: Path, hparams: dict[str, Any]) -> None:
        """Initialize the JSON Lines tracker.

        Args:
            run_id (str): Unique identifier for the training run.
            run_dir (Path): Directory where log files will be stored.
            hparams (dict[str , Any]): Hyperparameters to log.

        """
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        """Write hyperparameters to a JSON Lines file.

        Creates a run-metrics.jsonl file containing the run ID and hyperparameters.
        Only executes on rank 0 in distributed training.
        """
        with jsonlines.open(self.run_dir / "run-metrics.jsonl", mode="w", sort_keys=True) as js_tracker:
            js_tracker.write({"run_id": self.run_id, "hparams": self.hparams})

    @overwatch.rank_zero_only
    def write(self, _: int, metrics: dict[str, int | float]) -> None:
        """Write metrics to a JSON Lines file.

        Appends metrics to a run-specific JSON Lines file.
        Only executes on rank 0 in distributed training.

        Args:
            _ (int): Global step (unused in this implementation).
            metrics (dict[str, int | float]): Metrics to log.

        """
        with jsonlines.open(self.run_dir / f"{self.run_id}.jsonl", mode="a", sort_keys=True) as js_tracker:
            js_tracker.write(metrics)

    def finalize(self) -> None:
        """Finalize the JSON Lines tracker.

        No cleanup needed for file-based logging.
        """
        return


class WeightsBiasesTracker:
    """Weights & Biases metric tracker.

    Integrates with Weights & Biases for cloud-based experiment tracking
    with rich visualizations and collaboration features.

    Attributes:
        run_id (str): Unique identifier for the training run.
        run_dir (Path): Directory for local W&B files.
        hparams (dict[str, Any]): Hyperparameters to log.
        project (str): W&B project name.
        entity (str | None): W&B entity/team name.
        group (str): W&B run group for organization.
        wandb_dir (Path): Local directory for W&B files.

    """

    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        hparams: dict[str, Any],
        project: str = "prismatic",
        entity: str | None = None,
        group: str = "align",
    ) -> None:
        """Initialize the Weights & Biases tracker.

        Args:
            run_id (str): Unique identifier for the training run.
            run_dir (Path): Directory for local W&B files.
            hparams (dict[str, Any]): Hyperparameters to log.
            project (str, optional): W&B project name. Defaults to "prismatic".
            entity (str | None, optional): W&B entity/team name. Defaults to None.
            group (str, optional): W&B run group. Defaults to "align".

        """
        self.run_id = run_id
        self.run_dir = run_dir
        self.hparams = hparams

        # Get W&B-Specific Initialization Parameters
        self.project = project
        self.entity = entity
        self.group = group
        self.wandb_dir = run_dir

        # Call W&B.init()
        self.initialize()

    @overwatch.rank_zero_only
    def initialize(self) -> None:
        """Initialize the W&B run.

        Sets up the W&B session with the specified configuration.
        Only executes on rank 0 in distributed training.
        """
        wandb.init(
            name=self.run_id,
            dir=self.wandb_dir,
            config=self.hparams,
            project=self.project,
            entity=self.entity,
            group=self.group,
        )

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        """Write hyperparameters to W&B config.

        Updates the W&B config with hyperparameters.
        Only executes on rank 0 in distributed training.
        """
        wandb.config = self.hparams

    @overwatch.rank_zero_only
    def write(self, global_step: int, metrics: dict[str, int | float]) -> None:
        """Write metrics to W&B.

        Logs metrics to Weights & Biases with the specified global step.
        Only executes on rank 0 in distributed training.

        Args:
            global_step (int): Current training step for metric alignment.
            metrics (dict[str, int | float]): Metrics to log.

        """
        wandb.log(metrics, step=global_step)

    @staticmethod
    def finalize() -> None:
        """Finalize the W&B session.

        Properly closes the W&B run and allows time for final synchronization.
        Only executes on rank 0 in distributed training.
        """
        if overwatch.is_rank_zero():
            wandb.finish()

        # A job gets 210 seconds to get its affairs in order
        time.sleep(210)


# === Core Metrics Container :: Initializes Trackers => Compiles/Pushes Metrics ===
class Metrics:
    """Core metrics tracking container for standard training.

    Manages multiple tracking backends and provides a unified interface for
    logging training metrics including loss, learning rate, and timing information.

    Attributes:
        run_id (str): Unique identifier for the training run.
        run_dir (Path): Directory for storing logs.
        hparams (dict[str, Any]): Training hyperparameters.
        stage (str): Training stage (e.g., "align", "finetune").
        trackers (list[Tracker]): List of active metric trackers.
        global_step (int): Current global training step.
        start_time (float): Training start timestamp.
        step_start_time (float): Current step start timestamp.
        state (dict[str, deque | list]): Metric buffers for smoothing and accumulation.

    """

    def __init__(
        self,
        active_trackers: Sequence[str],
        run_id: str,
        run_dir: Path,
        hparams: dict[str, Any],
        stage: str,
        wandb_project: str = "prismatic",
        wandb_entity: str | None = None,
        grad_accumulation_steps: int = 1,
        window_size: int = 128,
    ) -> None:
        """Initialize the metrics tracking system.

        Args:
            active_trackers (Sequence[str]): List of tracker types to activate
                (e.g., ["jsonl", "wandb"]).
            run_id (str): Unique identifier for the training run.
            run_dir (Path): Directory for storing logs and outputs.
            hparams (dict[str, Any]): Training hyperparameters to log.
            stage (str): Training stage identifier (e.g., "align", "finetune").
            wandb_project (str, optional): W&B project name. Defaults to "prismatic".
            wandb_entity (str | None, optional): W&B entity name. Defaults to None.
            grad_accumulation_steps (int, optional): Number of gradient accumulation steps.
                Defaults to 1.
            window_size (int, optional): Size of moving average window for metrics.
                Defaults to 128.

        Raises:
            ValueError: If an unsupported tracker type is specified.

        """
        self.run_id = run_id
        self.run_dir = run_dir
        self.hparams = hparams
        self.stage = stage

        self.trackers: list[Tracker] = []
        for tracker_type in active_trackers:
            if tracker_type == "jsonl":
                tracker = JSONLinesTracker(run_id=run_id, run_dir=run_dir, hparams=hparams)
            elif tracker_type == "wandb":
                tracker = WeightsBiasesTracker(
                    run_id=run_id,
                    run_dir=run_dir,
                    hparams=hparams,
                    project=wandb_project,
                    entity=wandb_entity,
                    group=self.stage,
                )
            else:
                msg = f"Tracker with type `{tracker_type}` is not supported!"
                raise ValueError(msg)
            # Add Hyperparameters --> add to `self.trackers`
            tracker.write_hyperparameters()
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step = 0
        self.start_time = time.time()
        self.step_start_time = time.time()

        self.state: dict[str, deque | list] = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),
            "loss": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": [],
        }

    def log(self, global_step: int, metrics: dict[str, int | float]) -> None:
        """Log metrics to all active trackers.

        Args:
            global_step (int): Current global training step.
            metrics (dict[str, int | float]): Dictionary of metrics to log.

        """
        for tracker in self.trackers:
            tracker.write(global_step, metrics)

    def get_status(self, loss: float | None = None) -> str:
        """Generate a status string for the current training state.

        Args:
            loss (float | None, optional): Current loss value to include.
                Defaults to None.

        Returns:
            str: Formatted status string with step, learning rate, and optionally loss.

        """
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status report!
        return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f} -- Loss :: {loss:.4f}"

    def commit(
        self,
        *,
        global_step: int | None = None,
        lr: float | None = None,
        update_step_time: bool = False,
        **kwargs: torch.Tensor,
    ) -> None:
        """Update metric buffers with new values.

        Args:
            global_step (int | None, optional): Global step to update. Defaults to None.
            lr (float | None, optional): Learning rate to record. Defaults to None.
            update_step_time (bool, optional): Whether to update step timing.
                Defaults to False.
            **kwargs: Additional metrics to record in the state buffers.

        """
        if global_step is not None:
            self.global_step = global_step

        # For all other variables --> only track on rank zero!
        if not overwatch.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            else:
                self.state[key].append(value.detach())

    @overwatch.rank_zero_only
    def push(self) -> str:
        """Compute smoothed metrics and push to all trackers.

        Calculates moving averages and other aggregated metrics from the
        current state buffers and logs them to all active trackers.
        Only executes on rank 0 in distributed training.

        Returns:
            str: Status string with current metrics.

        """
        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        loss_raw = torch.stack(list(self.state["loss_raw"])).mean().item()
        loss = torch.stack(list(self.state["loss"])).mean().item()
        step_time = np.mean(list(self.state["step_time"]))
        lr = self.state["lr"][-1]
        status = self.get_status(loss)

        # Fire to Trackers
        prefix = self.stage.capitalize()
        self.log(
            self.global_step,
            metrics={
                f"{prefix}/Step": self.global_step,
                f"{prefix}/Loss": loss,
                f"{prefix}/Loss (Raw)": loss_raw,
                f"{prefix}/Learning Rate": lr,
                f"{prefix}/Step Time": step_time,  # pyright: ignore[reportArgumentType]
            },
        )
        return status

    def finalize(self) -> None:
        """Finalize all active trackers.

        Calls finalize on all tracker instances to ensure proper cleanup
        and final synchronization.
        """
        for tracker in self.trackers:
            tracker.finalize()


class VLAMetrics:
    """Specialized metrics tracking for Vision-Language-Action (VLA) training.

    Extends the basic metrics tracking with VLA-specific metrics including
    action accuracy, L1 loss, and per-dataset tracking capabilities.

    Attributes:
        run_id (str): Unique identifier for the training run.
        run_dir (Path): Directory for storing logs.
        hparams (dict[str, Any]): Training hyperparameters.
        trackers (list[Tracker]): List of active metric trackers.
        global_step (int): Current global training step.
        epoch (int): Current training epoch.
        start_time (float): Training start timestamp.
        step_start_time (float): Current step start timestamp.
        state (dict): Metric buffers for VLA-specific metrics.
        dataset_trackers (defaultdict): Per-dataset metric tracking.

    """

    def __init__(
        self,
        active_trackers: Sequence[str],
        run_id: str,
        run_dir: Path,
        hparams: dict[str, Any],
        wandb_project: str = "openvla",
        wandb_entity: str | None = "stanford-voltron",
        grad_accumulation_steps: int = 1,
        window_size: int = 1,
        resume_step: int | None = None,
        resume_epoch: int | None = None,
    ) -> None:
        """Initialize VLA metrics tracking.

        Args:
            active_trackers (Sequence[str]): List of tracker types to activate.
            run_id (str): Unique identifier for the training run.
            run_dir (Path): Directory for storing logs and outputs.
            hparams (dict[str, Any]): Training hyperparameters to log.
            wandb_project (str, optional): W&B project name. Defaults to "openvla".
            wandb_entity (str | None, optional): W&B entity name.
                Defaults to "stanford-voltron".
            grad_accumulation_steps (int, optional): Number of gradient accumulation steps.
                Defaults to 1.
            window_size (int, optional): Size of moving average window. Defaults to 1.
            resume_step (int | None, optional): Step to resume from. Defaults to None.
            resume_epoch (int | None, optional): Epoch to resume from. Defaults to None.

        Raises:
            ValueError: If an unsupported tracker type is specified.

        """
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

        # Initialize Trackers
        self.trackers: list[Tracker] = []
        for tracker_type in active_trackers:
            if tracker_type == "jsonl":
                tracker = JSONLinesTracker(run_id, run_dir, hparams)
            elif tracker_type == "wandb":
                tracker = WeightsBiasesTracker(
                    run_id,
                    run_dir,
                    hparams,
                    project=wandb_project,
                    entity=wandb_entity,
                    group="vla-train",
                )
            else:
                msg = f"Tracker with type `{tracker_type} is not supported!"
                raise ValueError(msg)

            # Add Hyperparameters --> add to `self.trackers`
            tracker.write_hyperparameters()
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step = 0 if resume_step is None else resume_step
        self.epoch = 0 if resume_epoch is None else resume_epoch
        self.start_time, self.step_start_time = time.time(), time.time()
        self.state = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),
            "loss": deque(maxlen=window_size),
            "l1_loss": deque(maxlen=window_size),
            "action_accuracy": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": [],
        }

        # Created metrics buffers for individual tracked datasets
        self.dataset_trackers = defaultdict(lambda: VLAMetrics([], "", Path(), {}))

    def log(self, global_step: int, metrics: dict[str, int | float]) -> None:
        """Log metrics to all active trackers.

        Args:
            global_step (int): Current global training step.
            metrics (dict[str, int | float]): Dictionary of metrics to log.

        """
        for tracker in self.trackers:
            tracker.write(global_step, metrics)

    def get_status(self, loss: float | None = None) -> str:
        """Generate a status string for the current VLA training state.

        Args:
            loss (float | None, optional): Current loss value to include.
                Defaults to None.

        Returns:
            str: Formatted status string with epoch, step, learning rate, and optionally loss.

        """
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status report!
        return (
            f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f} - Loss :: {loss:.4f}"
        )

    def commit(
        self,
        *,
        global_step: int | None = None,
        epoch: int | None = None,
        lr: float | None = None,
        update_step_time: bool = False,
        **kwargs: torch.Tensor,
    ) -> None:
        """Update VLA metric buffers with new values.

        Args:
            global_step (int | None, optional): Global step to update. Defaults to None.
            epoch (int | None, optional): Current epoch to update. Defaults to None.
            lr (float | None, optional): Learning rate to record. Defaults to None.
            update_step_time (bool, optional): Whether to update step timing.
                Defaults to False.
            **kwargs: Additional VLA-specific metrics to record.

        """
        if global_step is not None:
            self.global_step = global_step

        if epoch is not None:
            self.epoch = epoch

        # For all other variables --> only track on rank zero!
        if not overwatch.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            else:
                self.state[key].append(value.detach())

    def commit_for_dataset(self, dataset_name: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Update metrics for a specific dataset.

        Args:
            dataset_name (str): Name of the dataset to track metrics for.
            **kwargs: Dataset-specific metrics to record.

        """
        self.dataset_trackers[dataset_name].commit(**kwargs)

    @overwatch.rank_zero_only
    def push(self) -> str:
        """Compute and push VLA-specific metrics to all trackers.

        Calculates VLA-specific aggregated metrics including action accuracy,
        L1 loss, and per-dataset metrics, then logs them to all active trackers.
        Only executes on rank 0 in distributed training.

        Returns:
            str: Status string with current VLA metrics.

        """
        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        loss_raw = torch.stack(list(self.state["loss_raw"])).mean().item()
        loss = torch.stack(list(self.state["loss"])).mean().item()
        l1_loss = torch.stack(list(self.state["l1_loss"])).mean().item()
        action_accuracy = torch.stack(list(self.state["action_accuracy"])).mean().item()
        step_time = np.mean(list(self.state["step_time"]))
        lr = self.state["lr"][-1]
        status = self.get_status(loss)

        # Get metrics per dataset
        dataset_metrics = {}
        for ds, tracker in self.dataset_trackers.items():
            dataset_metrics.update(
                {
                    f"{ds}/L1 Loss": torch.stack(list(tracker.state["l1_loss"])).mean().item(),
                    f"{ds}/Action Token Accuracy": torch.stack(list(tracker.state["action_accuracy"])).mean().item(),
                },
            )

        # Fire to Trackers
        prefix = "VLA Train"
        self.log(
            self.global_step,
            metrics={
                f"{prefix}/Step": self.global_step,
                f"{prefix}/Epoch": self.epoch,
                f"{prefix}/Loss": loss,
                f"{prefix}/L1 Loss": l1_loss,
                f"{prefix}/Action Token Accuracy": action_accuracy,
                f"{prefix}/Loss (Raw)": loss_raw,
                f"{prefix}/Learning Rate": lr,
                f"{prefix}/Step Time": step_time,  # pyright: ignore[reportArgumentType]
                **dataset_metrics,
            },
        )
        return status

    def finalize(self) -> None:
        """Finalize all active VLA metric trackers.

        Calls finalize on all tracker instances to ensure proper cleanup
        and final synchronization.
        """
        for tracker in self.trackers:
            tracker.finalize()
