from typing import Callable, Literal
from jaxtyping import Float, Int, Bool

from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    response_start = []
    response_end = []
    full_sequences = []

    for i, prompt in enumerate(prompt_strs):
        response = output_strs[i]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        full_sequences.append(torch.tensor(prompt_ids + response_ids))
        response_start.append(len(prompt_ids) - 1) # -1 for switching into the labels coordinate system
        response_end.append(len(prompt_ids) + len(response_ids) - 1)

    out = {}
    padded = pad_sequence(full_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
    out['input_ids'] = padded[:, :-1]
    out['labels'] = padded[:, 1:]
    positions = torch.arange(out['labels'].shape[-1])
    positions = rearrange(positions, 'seq -> 1 seq')
    response_start = torch.tensor(response_start)
    response_start = rearrange(response_start, 'batch -> batch 1')
    response_end = torch.tensor(response_end)
    response_end = rearrange(response_end, "batch -> batch 1")
    response_mask = (positions >= response_start) & (positions < response_end)
    out['response_mask'] = response_mask

    return out

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    model_output: CausalLMOutputWithCrossAttentions = model(input_ids)
    result: dict[str, torch.Tensor] = {}

    log_prob = torch.log_softmax(model_output.logits, dim=-1)
    result["log_probs"] = log_prob.gather(-1, labels[:, :, None]).squeeze(-1)

    if return_token_entropy:
        # calculate token entropy
        entropy = torch.distributions.Categorical(logits=model_output.logits).entropy()
        result["token_entropy"] = entropy

    return result

def compute_rollout_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    reward_dicts = [reward_fn(resp, truth) for resp, truth in zip(rollout_responses, repeated_ground_truths)]
    rewards = [reward['reward'] for reward in reward_dicts]
    meta = {}
    meta['avg_format_rewards'] = sum([reward['format_reward'] for reward in reward_dicts]) / len(reward_dicts)
    meta['avg_answer_rewards'] = sum([reward['answer_reward'] for reward in reward_dicts]) / len(reward_dicts)
    meta['avg_rewards'] = sum(rewards) / len(rewards)
    rewards = torch.tensor(rewards)
    return rewards, meta

def compute_group_normalized_rewards(
    raw_rewards: Float[Tensor, "rollout_batch_size"],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[Float[Tensor, "rollout_batch_size"], dict[str, float]]:

    rewards = rearrange(raw_rewards, '(prompt_num group_size) -> prompt_num group_size', group_size=group_size)
    
    if baseline == "mean":
        base = torch.mean(rewards, dim=1)
    else:
        raise NotImplementedError

    if advantage_normalizer == "std":
        norm = torch.std(rewards, dim=1) + advantage_eps
    else:
        raise NotImplementedError

    advantages = (rewards - base) / norm
    advantages = rearrange(advantages, 'prompt_num group_size -> (prompt_num group_size)')

    metadata = {
        "mean": torch.mean(raw_rewards).item(),
        "std": torch.std(raw_rewards).item(),
    }

    return advantages, metadata

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Float[Tensor, "batch_size ..."],
    policy_log_probs: Float[Tensor, "batch_size seq_len"],
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Float[Tensor, "batch_size seq_len"] | None = None,
    cliprange: float | None = None,
    response_mask: Float[Tensor, "batch_size seq_len"] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    metadata = {}
    per_token_policy_gradient_loss = None

    raw_rewards_or_advantages = rearrange(raw_rewards_or_advantages, "batch_size ... -> batch_size (...)")

    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = - raw_rewards_or_advantages * policy_log_probs
    else:
        raise NotImplementedError

    return per_token_policy_gradient_loss, metadata

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Float[Tensor, "batch_size seq_len"],
    mask: Bool[Tensor, "batch_size seq_len"],
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    loss = None

    if loss_normalization == "sequence":
        masked = per_token_policy_gradient_loss.masked_fill(~mask, 0)
        seq_avg_loss = masked.sum(dim=1) / mask.sum(dim=1)
        loss = seq_avg_loss.mean()
    else:
        raise NotImplementedError

    return loss