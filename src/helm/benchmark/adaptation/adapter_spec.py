from dataclasses import dataclass, field
from typing import List, Optional

from helm.common.image_generation_parameters import ImageGenerationParameters


# Adaptation methods
ADAPT_GENERATION: str = "generation"
ADAPT_LANGUAGE_MODELING: str = "language_modeling"
ADAPT_MULTIPLE_CHOICE_JOINT: str = "multiple_choice_joint"
ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL: str = "multiple_choice_separate_original"
ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED: str = "multiple_choice_separate_calibrated"
ADAPT_RANKING_BINARY: str = "ranking_binary"

ADAPT_MULTIPLE_CHOICE_SEPARATE_METHODS: List[str] = [
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
]

# Multimodal adaptation methods
ADAPT_GENERATION_MULTIMODAL: str = "generation_multimodal"
ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL: str = "multiple_choice_joint_multimodal"


@dataclass(frozen=True)
class Substitution:
    """Represents a regular expression search/replace."""

    source: str
    target: str


@dataclass(frozen=True)
class AdapterSpec:
    """
    Specifies how to take a `Scenario` (a list of `Instance`s) and produce a
    `ScenarioState` (set of `Request`s ). Instead of having free-form prompt
    hacking, we try to make the process more declarative and systematic.
    Note that an `Instance` could produce many `Request`s (e.g., one for each `Reference`).
    """

    # Method of adaptation
    method: str = ""

    # Prepend all prompts with this string.
    # For example, it is recommended to prefix all prompts with [NLG] for UL2.
    global_prefix: str = ""

    # Append all prompts with this string.
    global_suffix: str = ""

    # Prompt starts with instructions
    instructions: str = ""

    # What goes before the input
    input_prefix: str = "Input: "

    # What goes after the input
    input_suffix: str = "\n"

    # What goes before the input (for multiple choice)
    reference_prefix: str = "A. "

    # What goes before the input (for multiple choice)
    reference_suffix: str = "\n"

    # What goes before the output
    output_prefix: str = "Output: "

    # What goes after the output
    output_suffix: str = "\n"

    # What goes between instruction and in-context example blocks in the constructed prompt
    instance_prefix: str = "\n"

    # List of regular expression substitutions that we perform
    substitutions: List[Substitution] = field(default_factory=list, hash=False)

    # Maximum number of (in-context) training instances to put into the prompt
    max_train_instances: int = 5

    # Maximum number of evaluation instances. For getting valid numbers, this
    # should be the entire dataset; only reduce this for piloting.
    max_eval_instances: Optional[int] = None

    # Generate this many outputs (which could be realized by `num_completions`
    # or `top_k_per_token`).
    num_outputs: int = 5

    # Number of trials, where in each trial we choose an independent, random
    # set of training instances.  Used to compute error bars.
    num_train_trials: int = 1

    # Number of trials, where we query the model with the same requests, but different random seeds
    num_trials: int = 1

    # If true, randomly sample N training examples; if false, select N consecutive training examples
    sample_train: bool = True

    # Decoding parameters (inherited by `Request`)

    # Model deployment to make the request to (need to fill in)
    model_deployment: str = ""

    # DEPRECATED: old model field, kept for backward compatibility
    # TODO: Remove this once we do not wish to support backward compatibility anymore.
    model: str = ""

    # Temperature to use
    temperature: float = 1

    # Maximum number of tokens to generate
    max_tokens: int = 100

    # When to stop (set hash=False to make `AdapterSpec` hashable)
    stop_sequences: List[str] = field(default_factory=list, hash=False)

    # Random string (used concretely to bypass cache / see diverse results)
    random: Optional[str] = None

    # If true, for instances with multiple correct reference, the gold answer should be considered
    # to be all the correct references rather than any of the correct references.
    multi_label: bool = False

    # Parameters for image generation
    image_generation_parameters: Optional[ImageGenerationParameters] = None
