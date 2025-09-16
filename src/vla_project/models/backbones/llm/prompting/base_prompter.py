"""base_prompter.py.

Abstract class definition of a multi-turn prompt builder for ensuring consistent formatting
for chat-based LLMs.
"""

from abc import ABC, abstractmethod


class PromptBuilder(ABC):
    """Abstract base class for building multi-turn prompts for chat-based language models.

    This class provides a framework for creating consistent prompt formatting across
    different model families and conversation styles.

    Attributes:
        model_family (str): The family/type of the language model.
        system_prompt (str | None): Optional system prompt to initialize conversations.

    """

    def __init__(self, model_family: str, system_prompt: str | None = None) -> None:
        """Initialize the PromptBuilder.

        Args:
            model_family (str): The family/type of the language model.
            system_prompt (str | None, optional): System prompt for conversation initialization.
                Defaults to None.

        """
        self.model_family = model_family

        # Only some models define a system prompt => let subclasses handle this logic!
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str:
        """Add a conversational turn to the prompt.

        Args:
            role (str): The role of the speaker (e.g., "human", "gpt").
            message (str): The message content for this turn.

        Returns:
            str: The formatted message that was added to the prompt.

        """
        ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> None | str:
        """Get a preview of what the prompt would look like with an additional user message.

        Args:
            user_msg (str): The potential user message to add.

        Returns:
            None | str: The complete prompt including the potential message, or None if invalid.

        """
        ...

    @abstractmethod
    def get_prompt(self) -> str:
        """Get the current complete prompt.

        Returns:
            str: The formatted prompt string ready for model input.

        """
        ...


class PurePromptBuilder(PromptBuilder):
    """Concrete implementation of PromptBuilder for simple "In:/Out:" style prompts.

    This builder creates prompts with a simple format where human messages are prefixed
    with "In: " and model responses are suffixed with end-of-sequence tokens.

    Attributes:
        bos (str): Beginning of sequence token.
        eos (str): End of sequence token.
        wrap_human (callable): Function to wrap human messages.
        wrap_gpt (callable): Function to wrap model responses.
        prompt (str): The accumulated prompt string.
        turn_count (int): Counter for conversation turns.

    """

    def __init__(self, model_family: str, system_prompt: str | None = None) -> None:
        """Initialize the PurePromptBuilder.

        Args:
            model_family (str): The family/type of the language model.
            system_prompt (str | None, optional): System prompt for conversation initialization.
                Defaults to None.

        """
        super().__init__(model_family, system_prompt)

        # TODO (siddk) =>> Can't always assume LlamaTokenizer --> FIX ME!
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"In: {msg}\nOut: "
        # self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # Delete blank
        self.wrap_gpt = lambda msg: f"{msg}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        """Add a conversational turn to the prompt.

        Alternates between human and model turns, formatting each appropriately.
        Human messages get "In: " prefix, model responses get end-of-sequence suffix.

        Args:
            role (str): The role of the speaker ("human" or "gpt").
            message (str): The message content. Image tags are automatically removed.

        Returns:
            str: The formatted message that was added to the prompt.

        Raises:
            AssertionError: If the role doesn't match the expected turn order.

        """
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")  # noqa: S101
        message = message.replace("<image>", "").strip()

        if (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, user_msg: str) -> str:
        """Get a preview of the prompt with an additional user message.

        Creates a copy of the current prompt and appends the potential user message
        to show what the complete prompt would look like.

        Args:
            user_msg (str): The potential user message to preview.

        Returns:
            str: The complete prompt including the potential message, with BOS token
                removed and trailing whitespace stripped.

        """
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(user_msg)
        prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        """Get the current complete prompt.

        Returns:
            str: The formatted prompt string with BOS token removed and trailing
                whitespace stripped, ready for model input.

        """
        # Remove prefix <bos> (if exists) because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()
