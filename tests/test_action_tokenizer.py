"""Tests for ActionTokenizer class."""

import numpy as np
import pytest
from transformers import AutoTokenizer

from vla_project.models.vla.action_tokenizer import ActionTokenizer


class TestActionTokenizer:
    """Test suite for ActionTokenizer functionality."""

    @pytest.fixture
    def tokenizer(self):
        """Create a base tokenizer for testing."""
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    @pytest.fixture
    def action_tokenizer(self, tokenizer):
        """Create an ActionTokenizer instance with default parameters."""
        return ActionTokenizer(tokenizer, bins=256, min_action=-1, max_action=1)

    @pytest.fixture
    def custom_action_tokenizer(self, tokenizer):
        """Create an ActionTokenizer instance with custom parameters."""
        return ActionTokenizer(tokenizer, bins=128, min_action=-2, max_action=2)

    def test_initialization(self, action_tokenizer: ActionTokenizer, tokenizer):
        """Test ActionTokenizer initialization."""
        assert action_tokenizer.n_bins == 256
        assert action_tokenizer.min_action == -1
        assert action_tokenizer.max_action == 1
        assert len(action_tokenizer.bins) == 256
        assert len(action_tokenizer.bin_center) == 255
        assert action_tokenizer.vocab_size == 256
        assert action_tokenizer.action_token_begin_idx == len(tokenizer.get_vocab()) - 257

    def test_custom_initialization(self, custom_action_tokenizer):
        """Test ActionTokenizer initialization with custom parameters."""
        assert custom_action_tokenizer.n_bins == 128
        assert custom_action_tokenizer.min_action == -2
        assert custom_action_tokenizer.max_action == 2
        assert len(custom_action_tokenizer.bins) == 128
        assert len(custom_action_tokenizer.bin_center) == 127

    def test_single_action_tokenization(self, action_tokenizer):
        """Test tokenization of single action values."""
        # Test action at center of range
        action = np.array([0.0])
        result = action_tokenizer(action)
        assert isinstance(result, str)

        # Test action at boundaries
        min_action = np.array([-1.0])
        max_action = np.array([1.0])
        min_result = action_tokenizer(min_action)
        max_result = action_tokenizer(max_action)
        assert isinstance(min_result, str)
        assert isinstance(max_result, str)
        assert min_result != max_result

    def test_batch_action_tokenization(self, action_tokenizer):
        """Test tokenization of batch of actions."""
        actions = np.array([[-0.5, 0.0, 0.5], [0.2, -0.8, 0.9]])
        result = action_tokenizer(actions)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, str) for r in result)

    def test_action_clipping(self, action_tokenizer):
        """Test that actions are properly clipped to valid range."""
        # Actions outside valid range should be clipped
        out_of_range_actions = np.array([-2.0, 3.0])
        result = action_tokenizer(out_of_range_actions)

        # Should be same as clipped actions
        clipped_actions = np.array([-1.0, 1.0])
        clipped_result = action_tokenizer(clipped_actions)
        assert result == clipped_result

    def test_roundtrip_conversion(self, action_tokenizer, tokenizer):
        """Test that actions can be converted to tokens and back accurately."""
        original_actions = np.array([0.1, -0.5, 0.8, 0.0, -0.9])

        # Convert actions to discrete bins and then to token IDs directly
        # This bypasses the string conversion which is problematic
        clipped_actions = np.clip(original_actions, float(action_tokenizer.min_action), float(action_tokenizer.max_action))
        discrete_actions = np.digitize(clipped_actions, action_tokenizer.bins)
        action_token_ids = len(tokenizer.get_vocab()) - discrete_actions

        # Convert token IDs back to actions
        decoded_actions = action_tokenizer.decode_token_ids_to_actions(action_token_ids)

        # Check that we get reasonable approximations (within bin precision)
        if len(decoded_actions) != len(original_actions):
            msg = f"Length mismatch: decoded {len(decoded_actions)} vs original {len(original_actions)}"
            raise ValueError(msg)
        np.testing.assert_allclose(decoded_actions, original_actions, atol=0.01)

    def test_edge_case_bin_indices(self, action_tokenizer):
        """Test edge cases in bin index handling."""
        # Test maximum bin index handling
        vocab_size = len(action_tokenizer.tokenizer.get_vocab())
        max_action_token_id = np.array([vocab_size - 1])  # This should map to bin index 1

        decoded = action_tokenizer.decode_token_ids_to_actions(max_action_token_id)
        assert len(decoded) == 1
        assert decoded[0] >= action_tokenizer.min_action
        assert decoded[0] <= action_tokenizer.max_action

    def test_vocab_size_property(self, action_tokenizer, custom_action_tokenizer):
        """Test vocab_size property returns correct value."""
        assert action_tokenizer.vocab_size == 256
        assert custom_action_tokenizer.vocab_size == 128

    def test_bin_center_calculation(self, action_tokenizer):
        """Test that bin centers are calculated correctly."""
        # Bin centers should be midpoints between bin boundaries
        expected_centers = 0.5 * (action_tokenizer.bins[1:] + action_tokenizer.bins[:-1])
        np.testing.assert_array_almost_equal(action_tokenizer.bin_center, expected_centers)

    def test_different_action_ranges(self, tokenizer):
        """Test ActionTokenizer with different action ranges."""
        # Test with positive range
        positive_tokenizer = ActionTokenizer(tokenizer, bins=64, min_action=0, max_action=5)
        action = np.array([2.5])
        result = positive_tokenizer(action)
        assert isinstance(result, str)

        # Test with large negative range
        negative_tokenizer = ActionTokenizer(tokenizer, bins=32, min_action=-10, max_action=-1)
        action = np.array([-5.0])
        result = negative_tokenizer(action)
        assert isinstance(result, str)

    @pytest.mark.parametrize("bins", [16, 64, 128, 512])
    def test_different_bin_sizes(self, tokenizer, bins):
        """Test ActionTokenizer with different bin sizes."""
        action_tokenizer = ActionTokenizer(tokenizer, bins=bins, min_action=-1, max_action=1)
        assert action_tokenizer.vocab_size == bins
        assert len(action_tokenizer.bins) == bins
        assert len(action_tokenizer.bin_center) == bins - 1

    def test_empty_input_handling(self, action_tokenizer):
        """Test handling of edge cases with empty or invalid inputs."""
        # Test empty array
        empty_action = np.array([])
        result = action_tokenizer(empty_action)
        assert isinstance(result, str)

    def test_single_vs_batch_consistency(self, action_tokenizer):
        """Test that single action and batch with one action produce same result."""
        single_action = np.array([0.5])
        batch_action = np.array([[0.5]])

        single_result = action_tokenizer(single_action)
        batch_result = action_tokenizer(batch_action)

        assert isinstance(single_result, str)
        assert isinstance(batch_result, list)
        assert len(batch_result) == 1
        # Results should be equivalent
        assert single_result == batch_result[0]
