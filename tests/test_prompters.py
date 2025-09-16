"""Tests for prompt builders including PhiPromptBuilder and other available prompters."""

import pytest

from vla_project.models.backbones.llm.prompting.base_prompter import PromptBuilder
from vla_project.models.backbones.llm.prompting.phi_prompter import PhiPromptBuilder


class TestPhiPromptBuilder:
    """Tests for PhiPromptBuilder class."""

    def test_init_without_system_prompt(self):
        """Test initialization without system prompt."""
        builder = PhiPromptBuilder("phi-2")
        assert builder.model_family == "phi-2"
        assert builder.prompt == ""
        assert builder.turn_count == 0
        assert builder.bos == "<|endoftext|>"
        assert builder.eos == "<|endoftext|>"

    def test_init_with_system_prompt(self):
        """Test initialization with system prompt."""
        system_prompt = "You are a coding assistant."
        builder = PhiPromptBuilder("phi-2", system_prompt)
        assert builder.model_family == "phi-2"
        assert system_prompt in builder.system_prompt # pyright: ignore[reportOperatorIssue]
        assert builder.turn_count == 0

    def test_first_human_turn_has_bos(self):
        """Test that first human turn includes BOS token."""
        builder = PhiPromptBuilder("phi-2")
        message = "Hello, world!"

        result = builder.add_turn("human", message)

        assert builder.bos in result
        assert "Input:" in result
        assert "Output:" in result
        assert message in result
        assert builder.turn_count == 1

    def test_subsequent_human_turns_no_bos(self):
        """Test that subsequent human turns don't include BOS token."""
        builder = PhiPromptBuilder("phi-2")
        builder.add_turn("human", "First message")
        builder.add_turn("gpt", "Response")

        result = builder.add_turn("human", "Second message")

        # BOS should not be in this turn (only in the first)
        assert builder.bos not in result
        assert "Input:" in result
        assert "Output:" in result

    def test_gpt_turn_has_eos(self):
        """Test that GPT turns include EOS token."""
        builder = PhiPromptBuilder("phi-2")
        builder.add_turn("human", "Hello")

        gpt_message = "Hi there!"
        result = builder.add_turn("gpt", gpt_message)

        assert builder.eos in result
        assert gpt_message in result

    def test_empty_gpt_message_handling(self):
        """Test handling of empty GPT messages."""
        builder = PhiPromptBuilder("phi-2")
        builder.add_turn("human", "Hello")

        result = builder.add_turn("gpt", "")

        assert " " in result  # Should contain a space for empty message
        assert builder.eos in result

    def test_image_tag_removal(self):
        """Test that image tags are removed from messages."""
        builder = PhiPromptBuilder("phi-2")
        message_with_image = "Look at this <image> and describe it."

        result = builder.add_turn("human", message_with_image)

        assert "<image>" not in result
        assert "<image>" not in builder.prompt
        assert "Look at this  and describe it." in result

    def test_alternating_roles_enforcement(self):
        """Test that roles must alternate correctly."""
        builder = PhiPromptBuilder("phi-2")

        # First turn should be human
        builder.add_turn("human", "First")

        # Second turn should be gpt
        builder.add_turn("gpt", "Second")

        # Third turn should be human
        builder.add_turn("human", "Third")

        assert builder.turn_count == 3

    def test_wrong_role_order_raises_error(self):
        """Test that wrong role order raises assertion error."""
        builder = PhiPromptBuilder("phi-2")

        # Trying to start with gpt should fail
        with pytest.raises(AssertionError):
            builder.add_turn("gpt", "This should fail")

        # After human turn, another human turn should fail
        builder.add_turn("human", "First")
        with pytest.raises(AssertionError):
            builder.add_turn("human", "This should also fail")

    def test_get_potential_prompt(self):
        """Test getting potential prompt without modifying state."""
        builder = PhiPromptBuilder("phi-2")
        builder.add_turn("human", "First message")
        original_prompt = builder.prompt
        original_count = builder.turn_count

        potential = builder.get_potential_prompt("What if I said this?")

        # State should not change
        assert builder.prompt == original_prompt
        assert builder.turn_count == original_count
        # Potential message should be formatted correctly
        assert "Input:" in potential
        assert "What if I said this?" in potential

    def test_get_prompt_strips_whitespace(self):
        """Test that get_prompt strips trailing whitespace."""
        builder = PhiPromptBuilder("phi-2")
        builder.add_turn("human", "Hello")

        # Add some trailing whitespace to prompt
        builder.prompt += "   \n\t  "

        prompt = builder.get_prompt()
        assert not prompt.endswith(" ")
        assert not prompt.endswith("\n")
        assert not prompt.endswith("\t")

    def test_phi_specific_formatting(self):
        """Test Phi-specific prompt formatting."""
        builder = PhiPromptBuilder("phi-2")

        # Add conversation
        builder.add_turn("human", "Write a Python function")
        builder.add_turn("gpt", "def hello(): print('Hello')")

        prompt = builder.get_prompt()

        # Check for Phi-specific elements
        assert "<|endoftext|>" in prompt  # BOS at start
        assert "Input:" in prompt
        assert "Output:" in prompt
        assert "<|endoftext|>" in prompt  # EOS after gpt response

    def test_conversation_flow_with_formatting(self):
        """Test a complete conversation with proper Phi formatting."""
        builder = PhiPromptBuilder("phi-2")

        # First exchange
        builder.add_turn("human", "What is 2+2?")
        builder.add_turn("gpt", "2+2 equals 4")

        # Second exchange
        builder.add_turn("human", "What about 3+3?")
        builder.add_turn("gpt", "3+3 equals 6")

        prompt = builder.get_prompt()

        assert builder.turn_count == 4
        assert prompt.count("Input:") == 2  # Two human inputs
        assert prompt.count("Output:") == 2  # Two output prompts
        assert prompt.count("<|endoftext|>") == 3  # One BOS + two EOS

    def test_wrap_functions(self):
        """Test the wrap functions are set correctly."""
        builder = PhiPromptBuilder("phi-2")

        # Test human wrap function
        human_wrapped = builder.wrap_human("test message")
        assert human_wrapped == "Input: test message\nOutput: "

        # Test gpt wrap function with regular message
        gpt_wrapped = builder.wrap_gpt("test response")
        assert gpt_wrapped == f"test response\n{builder.eos}"

        # Test gpt wrap function with empty message
        gpt_empty = builder.wrap_gpt("")
        assert gpt_empty == f" \n{builder.eos}"

    def test_inherits_from_prompt_builder(self):
        """Test that PhiPromptBuilder properly inherits from PromptBuilder."""
        builder = PhiPromptBuilder("phi-2")
        assert isinstance(builder, PromptBuilder)

        # Should have base class attributes
        assert hasattr(builder, "model_family")
        assert hasattr(builder, "prompt")
        assert hasattr(builder, "turn_count")

        # Should have base class methods
        assert hasattr(builder, "add_turn")
        assert hasattr(builder, "get_prompt")
        assert hasattr(builder, "get_potential_prompt")


class TestPromptBuilderBase:
    """Tests for the base PromptBuilder class (abstract)."""

    def test_cannot_instantiate_base_class(self):
        """Test that PromptBuilder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # Should fail because PromptBuilder is abstract
            PromptBuilder("test_model")  # type: ignore[abstract]


class TestPhiPromptBuilderEdgeCases:
    """Edge case tests for PhiPromptBuilder."""

    def test_very_long_conversation(self):
        """Test handling of very long conversations."""
        builder = PhiPromptBuilder("phi-2")

        # Add many turns
        for i in range(20):
            builder.add_turn("human", f"Message {i}")
            builder.add_turn("gpt", f"Response {i}")

        prompt = builder.get_prompt()
        assert builder.turn_count == 40
        assert prompt.count("Input:") == 20
        assert prompt.count("Output:") == 20
        # Should have 1 BOS + 20 EOS tokens
        assert prompt.count("<|endoftext|>") == 21

    def test_special_characters_in_messages(self):
        """Test handling of special characters and formatting."""
        builder = PhiPromptBuilder("phi-2")

        special_message = "Hello\nworld\ttest!@#$%^&*()"
        builder.add_turn("human", special_message)

        prompt = builder.get_prompt()
        assert special_message in prompt
        assert "Input:" in prompt

    def test_empty_and_whitespace_messages(self):
        """Test handling of empty and whitespace-only messages."""
        builder = PhiPromptBuilder("phi-2")

        # Empty human message
        builder.add_turn("human", "")
        assert builder.turn_count == 1

        # Whitespace-only human message
        builder.add_turn("gpt", "response")
        builder.add_turn("human", "   \n\t  ")
        assert builder.turn_count == 3

        prompt = builder.get_prompt()
        assert "Input:" in prompt
        assert "Output:" in prompt

    def test_multiple_image_tags(self):
        """Test removal of multiple image tags."""
        builder = PhiPromptBuilder("phi-2")

        message = "Look at <image> and <image> and describe <image> them."
        result = builder.add_turn("human", message)

        assert "<image>" not in result
        assert result.count("and  and") == 1  # Should have spaces where images were

    def test_get_potential_prompt_with_empty_current_prompt(self):
        """Test get_potential_prompt when current prompt is empty."""
        builder = PhiPromptBuilder("phi-2")

        # No turns added yet
        potential = builder.get_potential_prompt("First message")

        assert "Input: First message" in potential
        assert "Output:" in potential
        assert builder.turn_count == 0  # Should not change state
