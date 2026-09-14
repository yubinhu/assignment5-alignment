from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
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
