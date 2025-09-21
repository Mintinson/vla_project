"""Adaptive ensemble module for action prediction smoothing and temporal consistency.

This module implements an adaptive ensemble mechanism that maintains temporal consistency
in action predictions by combining multiple action predictions over time using cosine
similarity-based weighting. The ensembler helps reduce jitter and improves smoothness
in robotic control by leveraging the temporal correlation between consecutive predictions.

The adaptive ensemble approach uses exponential weighting based on cosine similarity
between the current prediction and historical predictions, allowing the system to
give more weight to predictions that are consistent with the current trajectory.

Example:
    Basic usage of the adaptive ensembler:

    ```python
    from vla_project.action_model.adaptive_ensemble import AdaptiveEnsembler
    import numpy as np

    # Create ensembler for 5-step action horizon
    ensembler = AdaptiveEnsembler(pred_action_horizon=5, adaptive_ensemble_alpha=1.0)

    # Process action predictions
    action1 = np.array([0.1, 0.2, 0.3])
    action2 = np.array([0.15, 0.25, 0.35])

    smoothed_action1 = ensembler.ensemble_action(action1)
    smoothed_action2 = ensembler.ensemble_action(action2)

    # Reset for new episode
    ensembler.reset()
    ```

"""

from collections import deque

import numpy as np


class AdaptiveEnsembler:
    """Adaptive ensemble mechanism for smoothing action predictions over time.

    This class implements an adaptive ensemble approach that maintains temporal consistency
    in action predictions by combining multiple action predictions using cosine similarity-based
    weighting. The ensembler helps reduce jitter and improves smoothness in robotic control
    by leveraging temporal correlation between consecutive predictions.

    The ensemble mechanism:
        1. Maintains a history of recent action predictions (up to pred_action_horizon)
        2. Computes cosine similarity between current and historical predictions
        3. Uses exponential weighting based on similarity scores
        4. Returns a weighted average of all predictions

    Attributes:
        pred_action_horizon: Maximum number of action predictions to keep in history.
        action_history: Deque storing recent action predictions for ensemble computation.
        adaptive_ensemble_alpha: Exponential weighting factor for cosine similarity.
            Higher values give more weight to similar predictions.

    Example:
        Creating and using an adaptive ensembler:

        ```python
        ensembler = AdaptiveEnsembler(pred_action_horizon=5, adaptive_ensemble_alpha=1.0)

        # Process a sequence of action predictions
        for action in action_sequence:
            smoothed_action = ensembler.ensemble_action(action)
            # Use smoothed_action for robot control
        ```

    """

    def __init__(self, pred_action_horizon: int, adaptive_ensemble_alpha: float = 0.0) -> None:
        """Initialize the adaptive ensembler.

        Args:
            pred_action_horizon: Maximum number of action predictions to maintain
                in the history buffer. Must be positive.
            adaptive_ensemble_alpha: Exponential weighting factor for cosine similarity.
                Higher values (e.g., 1.0-2.0) give more weight to predictions that are
                similar to the current prediction. Zero means uniform weighting.
                Defaults to 0.0.

        """
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self) -> None:
        """Clear the action history buffer.

        This method resets the ensembler to its initial state by clearing all
        stored action predictions from the history buffer. This should be called
        at the beginning of each new episode or when starting a new sequence
        of action predictions.
        """
        self.action_history.clear()

    def ensemble_action(self, cur_action: np.ndarray) -> np.ndarray:
        """Compute ensemble-weighted action prediction based on temporal consistency.

        This method processes the current action prediction by:
        1. Adding it to the action history buffer
        2. Computing cosine similarity between the current prediction and all historical predictions
        3. Calculating exponential weights based on similarity scores
        4. Returning a weighted average of all predictions in the history

        The method handles both 1D action vectors and multi-dimensional action sequences
        by appropriately indexing into the history buffer to align temporal dimensions.

        Args:
            cur_action: Current action prediction as a numpy array. Can be either:
                - 1D array of shape (action_dim,) for single-step actions
                - 2D array of shape (horizon, action_dim) for multi-step predictions

        Returns:
            Ensemble-weighted action prediction as numpy array with same shape as input.
            The returned action incorporates information from the temporal history to
            provide smoother, more consistent predictions.

        Note:
            - For the first prediction in a sequence, returns the input unchanged
            - Cosine similarity is computed with a small epsilon (1e-7) for numerical stability
            - Weights are normalized to sum to 1.0 before applying to predictions
            - The ensemble size grows up to pred_action_horizon, then uses a sliding window

        """
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history, strict=False)
                ],
            )

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions - 1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()

        # compute the weighted average across all predictions for this timestep
        return np.sum(weights[:, None] * curr_act_preds, axis=0)
