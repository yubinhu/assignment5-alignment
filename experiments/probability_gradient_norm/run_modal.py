"""Standalone score-gradient experiment. No training or assignment-code imports.

Run from repository root:
  .venv/bin/modal run experiments/probability_gradient_norm/run_modal.py
One A100-80GB, one invocation, no retries, 50 minute hard timeout.
"""
from pathlib import Path
import json
import modal

HERE = Path(__file__).resolve().parent
app = modal.App("cs336-score-gradient-pilot")
volume = modal.Volume.from_name("cs336-score-gradient-pilot", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "transformers==5.7.0", "accelerate==1.6.0")
    .env({"HF_HOME": "/artifacts/hf", "TOKENIZERS_PARALLELISM": "false"})
)


@app.function(image=image, gpu="A100-80GB", timeout=3000, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": volume})
def experiment(prompts, astra, template, provenance):
    import hashlib
    import importlib.metadata
    import os
    import random
    import time
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    deadline = started + 2850
    random.seed(336)
    torch.manual_seed(336)
    torch.cuda.manual_seed_all(336)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_id = "allenai/OLMo-2-0425-1B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda").eval()
    revision = model.config._commit_hash
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    params = list(model.parameters())
    assert all(p.dtype == torch.float32 and p.requires_grad for p in params)
    max_tokens = 256
    outdir = Path("/artifacts/run_20261122")
    outdir.mkdir(exist_ok=True)
    payload = {
        "metadata": {
            "model": model_id, "revision": revision, "seed": 336,
            "n_prompts": len(prompts), "max_new_tokens": max_tokens,
            "matched_prefix_cap": 128, "gpu": torch.cuda.get_device_name(),
            "gpu_count": 1, "timeout_seconds": 3000,
            "score_dtype": "float32", "tf32": False,
            "generation_dtype": "bfloat16 autocast with float32 parameters",
            "score_temperature": 1.0, "weights_updated": False,
            "prompt_template": template, "astra_provenance": provenance,
            "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "accelerate"]},
            "input_sha256": hashlib.sha256(json.dumps([prompts, astra, template], sort_keys=True).encode()).hexdigest(),
            "gradient_definition": "L2 over all parameter gradients of sum response-token log probabilities; no reward, KL, clipping, or optimizer",
            "token_boundary": "Prompt and continuation encoded separately; original generated token IDs retained. EOS excluded for all sources. Closing answer tag retained, following text discarded.",
            "primary_comparison": "Within-prompt same-token-count prefix: min(128, shortest response among all five conditions)",
        }, "records": [],
    }

    def checkpoint():
        payload["metadata"]["gpu_function_seconds"] = time.monotonic() - started
        payload["metadata"]["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        tmp = outdir / "results.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, outdir / "results.json")
        volume.commit()

    def trim(ids):
        ids = list(ids)
        for i, token in enumerate(ids):
            if token == tok.eos_token_id:
                return ids[:i], "eos"
            if "</answer>" in tok.decode(ids[:i + 1], skip_special_tokens=False):
                return ids[:i + 1], "answer_tag"
        return ids, "length" if len(ids) == max_tokens else "external_end"

    prompt_ids = {p["id"]: tok.encode(template.format(question=p["question"]), add_special_tokens=False) for p in prompts}
    generated = {p["id"]: {} for p in prompts}
    conditions = [("greedy", None, 336), ("sample_t1_a", 1.0, 337),
                  ("sample_t1_b", 1.0, 338), ("sample_t15", 1.5, 339)]
    for condition, temperature, seed in conditions:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for offset in range(0, len(prompts), 8):
            if time.monotonic() > deadline:
                raise TimeoutError("Stopping generation within GPU budget")
            batch = prompts[offset:offset + 8]
            encoded = tok.pad({"input_ids": [prompt_ids[p["id"]] for p in batch]}, return_tensors="pt").to("cuda")
            kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature is not None,
                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                          use_cache=True)
            if temperature is not None:
                kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.generate(**encoded, **kwargs)
            for p, row in zip(batch, output):
                ids, reason = trim(row[encoded.input_ids.shape[1]:].tolist())
                generated[p["id"]][condition] = {"token_ids": ids, "finish_reason": reason,
                    "response": tok.decode(ids, skip_special_tokens=False)}
            print(f"Generated {condition} {offset + len(batch)}/{len(prompts)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        (outdir / "generations.json").write_text(json.dumps(generated))
        volume.commit()

    for row in astra:
        raw_ids = tok.encode(row["response"], add_special_tokens=False)
        ids, reason = trim(raw_ids[:max_tokens])
        generated[row["id"]]["astra"] = {"token_ids": ids, "finish_reason": reason,
            "response": tok.decode(ids, skip_special_tokens=False), "original_response": row["response"]}

    def measure(prefix, continuation):
        assert prefix and continuation
        model.zero_grad(set_to_none=True)
        ids = torch.tensor([prefix + continuation], device="cuda")
        outputs = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        logits = outputs.logits[:, len(prefix) - 1:-1, :]
        target = ids[:, len(prefix):]
        lp = -F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
        with torch.no_grad():
            prob = logits.softmax(-1)
            chosen = prob.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            logit_norm_sq = 1 - 2 * chosen + prob.square().sum(-1)
            diagnostic = {"token_nll": (-lp).tolist(), "token_logit_grad_norm": logit_norm_sq.clamp_min(0).sqrt()[0].tolist()}
        score = lp.sum()
        score.backward()
        squared = torch.stack([p.grad.detach().norm().double().square() for p in params if p.grad is not None]).sum()
        norm = squared.sqrt().item()
        assert norm > 0 and torch.isfinite(squared).item() and torch.isfinite(score).item()
        result = {"n_tokens": len(continuation), "sum_logprob": score.item(),
                  "mean_logprob": score.item() / len(continuation), "grad_norm_sum": norm,
                  "grad_norm_mean": norm / len(continuation), **diagnostic}
        model.zero_grad(set_to_none=True)
        return result

    for p in prompts:
        pid = p["id"]
        responses = generated[pid]
        assert len(responses) == 5
        length = min(128, *(len(r["token_ids"]) for r in responses.values()))
        assert length > 0, f"Empty response for {pid}"
        for condition, response in responses.items():
            if time.monotonic() > deadline:
                payload["metadata"]["stopped_early"] = True
                checkpoint()
                return payload
            full = measure(prompt_ids[pid], response["token_ids"])
            matched = full if len(response["token_ids"]) == length else measure(prompt_ids[pid], response["token_ids"][:length])
            payload["records"].append({"id": pid, "question": p["question"],
                "condition": condition, **response, "prompt_token_ids": prompt_ids[pid],
                "n_tokens": len(response["token_ids"]), "truncated": response["finish_reason"] == "length",
                "full": full, "matched": matched})
        checkpoint()
        print(f"Scored prompt {pid}: {len(payload['records'])} rows; matched={length}; elapsed={time.monotonic()-started:.1f}s", flush=True)

    # Repeat a real example to detect accidental gradient accumulation/dropout.
    first = payload["records"][0]
    repeat = measure(first["prompt_token_ids"], first["token_ids"][:first["matched"]["n_tokens"]])
    relerr = abs(repeat["grad_norm_sum"] / first["matched"]["grad_norm_sum"] - 1)
    assert relerr < 1e-5, relerr
    payload["metadata"]["repeat_gradient_relative_error"] = relerr
    payload["metadata"]["stopped_early"] = False
    checkpoint()
    return payload


@app.function(image=image, gpu="A100-80GB", timeout=600, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": volume})
def whitespace_sensitivity(data):
    """Post-hoc formatting check; reuse identical fixed checkpoint and matched lengths."""
    import time
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = data["metadata"]
    model = AutoModelForCausalLM.from_pretrained(meta["model"], revision=meta["revision"],
        dtype=torch.float32, attn_implementation="sdpa").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(meta["model"], revision=meta["revision"])
    groups = {}
    for r in data["records"]:
        groups.setdefault(r["id"], {})[r["condition"]] = r

    def measure(prefix, continuation):
        model.zero_grad(set_to_none=True)
        ids = torch.tensor([prefix + continuation], device="cuda")
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, len(prefix)-1:-1, :]
        lp = -F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, len(prefix):].reshape(-1), reduction="sum")
        lp.backward()
        norm = torch.stack([p.grad.detach().norm().double().square() for p in model.parameters() if p.grad is not None]).sum().sqrt().item()
        result = {"n_tokens": len(continuation), "sum_logprob": lp.item(),
                  "mean_logprob": lp.item()/len(continuation), "grad_norm_sum": norm,
                  "grad_norm_mean": norm/len(continuation)}
        model.zero_grad(set_to_none=True)
        return result

    records = []
    for pid, group in groups.items():
        a, g = group["astra"], group["greedy"]
        text = " " + a["response"].lstrip()
        tokens = tok.encode(text, add_special_tokens=False)
        length = min(a["matched"]["n_tokens"], len(tokens))
        greedy = g["matched"] if length == g["matched"]["n_tokens"] else measure(g["prompt_token_ids"], g["token_ids"][:length])
        new = measure(a["prompt_token_ids"], tokens[:length])
        records.append({"id": pid, "n_tokens": length, "astra_leading_space": new,
                        "greedy": greedy, "original_astra": a["matched"], "token_ids": tokens[:length]})
    result = {"description": "Post-hoc control adding one leading space to Astra, retaining original common prefix lengths wherever possible.",
              "gpu_function_seconds": time.monotonic()-started, "gpu": torch.cuda.get_device_name(),
              "model_revision": meta["revision"], "records": records}
    Path("/artifacts/run_20261122/whitespace_sensitivity.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(whitespace: bool = False):
    if whitespace:
        data = json.loads((HERE / "results.json").read_text())
        result = whitespace_sensitivity.remote(data)
        (HERE / "whitespace_sensitivity.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Whitespace sensitivity saved; GPU function seconds: {result['gpu_function_seconds']:.1f}")
        return
    root = HERE.parents[1]
    prompts = json.loads((HERE / "prompts.json").read_text())
    astra = json.loads((HERE / "astra_responses.json").read_text())
    assert {p["id"] for p in prompts} == {a["id"] for a in astra}
    assert len(prompts) == len(astra) == 32
    template = (root / "cs336_alignment/prompts/r1_zero.prompt").read_text()
    provenance = json.loads((HERE / "astra_provenance.json").read_text())
    result = experiment.remote(prompts, astra, template, provenance)
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {len(result['records'])} records. GPU function seconds: {result['metadata']['gpu_function_seconds']:.1f}")
