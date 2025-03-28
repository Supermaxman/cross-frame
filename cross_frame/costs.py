import dataclasses
from openai.types.chat import ChatCompletion


def usage_to_text(usage):
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    lines = [
        f"Total Tokens: {usage.total_tokens}",
        f"Prompt Tokens: {usage.prompt_tokens}",
    ]
    if prompt_details is not None:
        if (
            prompt_details.cached_tokens is not None
            and prompt_details.cached_tokens > 0
        ):
            lines.append(f"  Cached Tokens: {prompt_details.cached_tokens}")
        if prompt_details.audio_tokens is not None and prompt_details.audio_tokens > 0:
            lines.append(f"  Audio Tokens: {prompt_details.audio_tokens}")

    lines.append(f"Completion Tokens: {usage.completion_tokens}")
    if completion_details is not None:
        if (
            completion_details.reasoning_tokens is not None
            and completion_details.reasoning_tokens > 0
        ):
            lines.append(f"  Reasoning Tokens: {completion_details.reasoning_tokens}")
        if (
            completion_details.audio_tokens is not None
            and completion_details.audio_tokens > 0
        ):
            lines.append(f"  Audio Tokens: {completion_details.audio_tokens}")
    return "\n".join(lines)


@dataclasses.dataclass
class CompletionUsageEstimate:
    completion_tokens: int
    prompt_tokens: int
    completion_cost: float
    prompt_cost: float
    total_cost: float


class CompletionUsageEstimator:
    def __init__(
        self,
        completion_cost: float = 0.01,
        completion_cost_tokens: int = 1000,
        prompt_cost: float = 0.0025,
        prompt_cost_tokens: int = 1000,
    ):
        self.running_completion_tokens = 0
        self.running_prompt_tokens = 0
        self.prompts = 0
        self.remaining_prompts = 0
        self.completion_cost = completion_cost / completion_cost_tokens
        self.prompt_cost = prompt_cost / prompt_cost_tokens

    def init(self, remaining_prompts: int):
        self.running_completion_tokens = 0
        self.running_prompt_tokens = 0
        self.prompts = 0
        self.remaining_prompts = remaining_prompts

    def update(self, completion: ChatCompletion):
        usage = completion.usage
        if usage is not None:
            self.running_completion_tokens += usage.completion_tokens
            self.running_prompt_tokens += usage.prompt_tokens
            self.prompts += 1
            self.remaining_prompts -= 1

    def estimate(self):
        # linearly interpolate by computing average tokens per prompt and multiplying by remaining prompts
        completion_tokens_per_prompt = self.running_completion_tokens / self.prompts
        prompt_tokens_per_prompt = self.running_prompt_tokens / self.prompts
        total_completion_tokens = (
            self.running_completion_tokens
            + self.remaining_prompts * completion_tokens_per_prompt
        )
        total_prompt_tokens = (
            self.running_prompt_tokens
            + self.remaining_prompts * prompt_tokens_per_prompt
        )
        completion_cost = total_completion_tokens * self.completion_cost
        prompt_cost = total_prompt_tokens * self.prompt_cost
        total_cost = completion_cost + prompt_cost
        return CompletionUsageEstimate(
            completion_tokens=total_completion_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_cost=completion_cost,
            prompt_cost=prompt_cost,
            total_cost=total_cost,
        )
