"""action_tokenizer.py.

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    """Tokenizer for converting continuous actions to discrete tokens and vice versa.

    This class provides functionality to discretize continuous action values into
    uniform bins and convert them to tokens that can be processed by language models.
    It also supports decoding token IDs back to continuous action values.

    Attributes:
        tokenizer (PreTrainedTokenizerBase): Underlying text tokenizer.
        n_bins (int): Number of discrete bins for action discretization.
        min_action (int): Minimum action value for clipping and binning.
        max_action (int): Maximum action value for clipping and binning.
        bins (np.ndarray): Array of bin boundaries for discretization.
        bin_center (np.ndarray): Array of bin center values for decoding.
        action_token_begin_idx (int): Starting index for action tokens in vocabulary.

    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
    ) -> None:
        """Initialize the ActionTokenizer.

        Sets up discretization bins and computes the starting index for action tokens
        in the tokenizer's vocabulary. Action tokens are allocated at the end of the
        vocabulary space.

        Args:
            tokenizer (PreTrainedTokenizerBase): Pre-trained tokenizer for text processing.
            bins (int, optional): Number of discrete bins for action space. Defaults to 256.
            min_action (int, optional): Minimum action value for normalization. Defaults to -1.
            max_action (int, optional): Maximum action value for normalization. Defaults to 1.

        """
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
        self.bin_center = 0.5 * (self.bins[1:] + self.bins[:-1])

        self.action_token_begin_idx: int = int(len(self.tokenizer.get_vocab()) - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> str | list[str]:
        """Convert continuous actions to tokenized string representation.

        Clips actions to the valid range, discretizes them into bins, and converts
        to token strings using the underlying tokenizer. Supports both single
        actions and batch processing.

        Args:
            action (np.ndarray): Continuous action values to tokenize. Can be 1D
                for single action or 2D for batch of actions.

        Returns:
            str | list[str]: Tokenized string representation of actions. Returns
                single string for 1D input or list of strings for 2D input.

        """
        action = np.clip(action, float(self.min_action), float(self.max_action))

        discrete_action = np.digitize(action, self.bins)  # Bin Indices

        if len(discrete_action.shape) == 1:
            return self.tokenizer.decode(list(len(self.tokenizer.get_vocab()) - discrete_action))

        return self.tokenizer.batch_decode((len(self.tokenizer.get_vocab()) - discrete_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """Convert discrete action token IDs back to continuous actions.

        Converts token IDs back to bin indices and maps them to continuous action
        values using bin centers. Handles edge cases where digitization might
        produce out-of-bounds indices.

        Args:
            action_token_ids (np.ndarray): Array of token IDs representing discrete actions.

        Returns:
            np.ndarray: Continuous action values corresponding to the input token IDs.

        Note:
            Due to the discretization process, bin indices are between [1, n_bins] inclusive,
            but bin centers only have (n_bins - 1) values. This method handles the mapping
            carefully to avoid out-of-bounds errors by clipping the final index.

        Example:
            With 256 bins, digitization returns indices [1, 256], but bin_centers has
            255 values. Indices are shifted to [0, 255] and clipped to [0, 254] to
            ensure valid indexing into bin_centers.

        """
        discretized_actions = len(self.tokenizer.get_vocab()) - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_center.shape[0] - 1)

        return self.bin_center[discretized_actions]

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size for action tokens.

        Returns:
            int: Number of action tokens (equal to number of bins).

        """
        return self.n_bins


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    original_actions = np.array([0.1, -0.5, 0.8, 0.0, -0.9])
    action_tokenizer = ActionTokenizer(tokenizer, bins=256, min_action=-1, max_action=1)

    # Convert to tokens
    token_string = action_tokenizer(original_actions)

    # Extract token IDs (skip special tokens from string parsing)
    tokens = tokenizer.tokenize(token_string)
    token_ids = np.array([tokenizer.convert_tokens_to_ids(token) for token in tokens])

    # Filter to only action tokens (those at end of vocab)
    vocab_size = len(tokenizer.get_vocab())
    action_token_ids = token_ids[token_ids >= action_tokenizer.action_token_begin_idx]

    # Convert back to actions
    decoded_actions = action_tokenizer.decode_token_ids_to_actions(action_token_ids)

    # Check that we get reasonable approximations (within bin precision)
    assert len(decoded_actions) == len(original_actions)
    np.testing.assert_allclose(decoded_actions, original_actions, atol=0.01)
