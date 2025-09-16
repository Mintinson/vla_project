from .base_prompter import PromptBuilder


class PhiPromptBuilder(PromptBuilder):
    """Prompt builder specifically designed for Phi-series language models.

    This class handles the specific formatting requirements for Phi models, including
    proper BOS/EOS token placement and Input/Output formatting style. Phi models use
    CodeGenTokenizer which doesn't automatically append BOS/EOS tokens, so this
    builder handles that explicitly.

    Attributes:
        bos (str): Beginning of sequence token ("<|endoftext|>").
        eos (str): End of sequence token ("<|endoftext|>").
        wrap_human (callable): Function to wrap human messages with "Input: " prefix.
        wrap_gpt (callable): Function to wrap model responses with EOS suffix.
        prompt (str): The accumulated prompt string.
        turn_count (int): Counter for conversation turns.

    """

    def __init__(self, model_family: str, system_prompt: str | None = None) -> None:
        """Initialize the PhiPromptBuilder.

        Sets up Phi-specific tokenization and formatting rules. Phi models use
        "<|endoftext|>" for both BOS and EOS tokens and require special handling
        for the first input to include a BOS token.

        Args:
            model_family (str): The family/type of the language model.
            system_prompt (str | None, optional): System prompt for conversation initialization.
                Defaults to None.

        """
        super().__init__(model_family, system_prompt)

        # Note =>> Phi Tokenizer is an instance of `CodeGenTokenizer(Fast)`
        #      =>> By default, does *not* append <BOS> / <EOS> tokens --> we handle that here (IMPORTANT)!
        self.bos, self.eos = "<|endoftext|>", "<|endoftext|>"

        # Get role-specific "wrap" functions
        #   =>> Note that placement of <bos>/<eos> were based on experiments generating from Phi-2 in Input/Output mode
        self.wrap_human = lambda msg: f"Input: {msg}\nOutput: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}\n{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        """Add a conversational turn to the prompt with Phi-specific formatting.

        Handles alternating human and model turns with proper Phi formatting.
        The first human message gets a BOS token prepended. Human messages use
        "Input: " prefix and model responses get EOS token suffix.

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

        # Special Handling for "first" input --> prepend a <BOS> token (expected by Prismatic)
        if self.turn_count == 0:
            bos_human_message = f"{self.bos}{self.wrap_human(message)}"
            wrapped_message = bos_human_message
        elif (self.turn_count % 2) == 0:
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
        formatted for Phi models to show what the complete prompt would look like.

        Args:
            user_msg (str): The potential user message to preview.

        Returns:
            str: The complete prompt including the potential message with trailing
                whitespace stripped.

        """
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(user_msg)
        prompt_copy += human_message

        return prompt_copy.rstrip()

    def get_prompt(self) -> str:
        """Get the current complete prompt formatted for Phi models.

        Returns the accumulated prompt string with trailing whitespace removed,
        ready for model input. Unlike some other prompt builders, this doesn't
        remove BOS tokens since Phi models handle them explicitly.

        Returns:
            str: The formatted prompt string ready for Phi model input.

        """
        return self.prompt.rstrip()
