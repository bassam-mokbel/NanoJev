#!/usr/bin/env python3
"""Full fine-tuning of a NanoJev bundle on custom decision data (for example, survey responses).

This follows the recipe of the released checkpoint: FP32 parameter and AdamW storage, BF16
autocast, complete-question cross entropy, separate backbone/head learning rates, gradient
checkpointing and dev-only checkpoint selection with the starting weights as a candidate. It adds
per-row sampling weights, an optional ranked probability score for ordered Score questions, a
train-marginal baseline, and one temperature fitted on the calibration split only.

Requests are encoded by predict_toy_decisions.prepare_examples, so training paths are the exact
token inputs DecisionPredictor and serve_decisions.py use. The output directory is itself a
NanoJev bundle (config.json, best.safetensors, tokenizer/, backbone_config/).

Commands:
  train        fine-tune all parameters and select a checkpoint on dev
  init-bundle  wrap an untuned Hugging Face backbone with fresh decision heads (control arm)
"""
import argparse
from array import array
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from predict_toy_decisions import (
    check_rotary, load_decision_model_class, local_checkpoint_files, prepare_examples, read_json,
    reject_nonfinite, unique_object,
)
from train_pipeline_decisions import pack_complete_questions

SPLITS = ("train", "dev", "calibration", "test", "ood")
ROW_KEYS = {"id", "split", "state", "questions", "gold", "gold_probs", "weight", "group"}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def new_output_dir(path):
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Use a new empty output directory; {out} already contains files")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Data: training rows are prediction requests plus explicit targets.

def target_vector(question, candidate_ids, gold, gold_probs):
    """Full target distribution in encoded candidate order; soft targets take precedence."""
    typ = question["type"]
    if gold_probs is not None:
        if typ == "boolean" and isinstance(gold_probs, (int, float)) and not isinstance(gold_probs, bool):
            gold_probs = {"false": 1.0 - gold_probs, "true": gold_probs}
        if typ == "score" and isinstance(gold_probs, list):
            gold_probs = {str(i): p for i, p in enumerate(gold_probs)}
        if not isinstance(gold_probs, dict) or not gold_probs or set(gold_probs) - set(candidate_ids):
            raise ValueError(f"gold_probs keys must be a subset of candidates {candidate_ids}")
        vector = [gold_probs.get(key, 0.0) for key in candidate_ids]
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in vector):
            raise ValueError("gold_probs must be finite nonnegative numbers")
        total = math.fsum(vector)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"gold_probs must sum to 1 (got {total})")
        return [float(p) / total for p in vector]
    if typ == "boolean":
        if not isinstance(gold, bool):
            raise ValueError("Boolean gold must be true or false")
        index = int(gold)
    elif typ == "choice":
        if gold not in candidate_ids:
            raise ValueError(f"Choice gold {gold!r} is not one of {candidate_ids}")
        index = candidate_ids.index(gold)
    else:
        if type(gold) is not int or not 0 <= gold < len(candidate_ids):
            raise ValueError("Score gold must be a level index")
        index = gold
    return [float(i == index) for i in range(len(candidate_ids))]


def load_examples(path, tokenizer, max_length):
    examples, row_ids, group_split = [], set(), {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        row = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
        if not isinstance(row, dict) or not {"id", "split", "state", "questions"} <= set(row) or set(row) - ROW_KEYS:
            raise ValueError(f"{where}: rows need id/split/state/questions; optional gold/gold_probs/weight/group")
        if row["split"] not in SPLITS:
            raise ValueError(f"{where}: split must be one of {SPLITS}")
        if row["id"] in row_ids:
            raise ValueError(f"{where}: duplicate row id {row['id']!r}")
        row_ids.add(row["id"])
        weight = row.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{where}: weight must be a positive finite number")
        # A respondent (or segment) must never appear in two splits.
        group = str(row.get("group", row["id"]))
        if group_split.setdefault(group, row["split"]) != row["split"]:
            raise ValueError(f"{where}: group {group!r} appears in both {group_split[group]} and {row['split']}")
        gold, gold_probs = row.get("gold", {}), row.get("gold_probs", {})
        if not isinstance(gold, dict) or not isinstance(gold_probs, dict):
            raise ValueError(f"{where}: gold and gold_probs must be objects keyed by question id")
        questions = row["questions"] if isinstance(row["questions"], dict) else {}
        unlabeled = [qid for qid in questions if qid not in gold and qid not in gold_probs]
        unknown = (set(gold) | set(gold_probs)) - set(questions)
        if unlabeled or unknown:
            raise ValueError(f"{where}: questions without targets {unlabeled}, targets without questions {sorted(unknown)}")
        request = {"states": [{"id": row["id"], "state": row["state"], "questions": row["questions"]}]}
        for ex in prepare_examples(request, tokenizer, max_length):
            question = questions[ex["qid"]]
            try:
                target = target_vector(question, ex["candidate_ids"], gold.get(ex["qid"]), gold_probs.get(ex["qid"]))
            except ValueError as exc:
                raise ValueError(f"{where}:{ex['qid']}: {exc}") from None
            # Each path repeats the state; 4-byte arrays instead of Python int lists keep large surveys in RAM.
            ex["leaf_tokens"] = [array("i", path) for path in ex["leaf_tokens"]]
            ex.update(split=row["split"], weight=float(weight), group=group, target=target,
                      prior_key=json.dumps(question, ensure_ascii=False, sort_keys=True))
            examples.append(ex)
    if not examples:
        raise ValueError(f"{path} contains no questions")
    return examples


def fit_prior(train):
    """Weighted train marginal per exact question: the 'answer share' baseline to beat."""
    sums, mass = {}, {}
    for ex in train:
        acc = sums.setdefault(ex["prior_key"], [0.0] * len(ex["target"]))
        for i, p in enumerate(ex["target"]):
            acc[i] += ex["weight"] * p
        mass[ex["prior_key"]] = mass.get(ex["prior_key"], 0.0) + ex["weight"]
    return {key: [v / mass[key] for v in acc] for key, acc in sums.items()}


def prior_probs(prior, ex, floor=1e-4):
    k = len(ex["candidate_ids"])
    smoothed = [max(p, floor) for p in prior.get(ex["prior_key"], [1.0 / k] * k)]
    total = sum(smoothed)
    return [p / total for p in smoothed]


def softmax(logits, temperature=1.0):
    scaled = [z / temperature for z in logits]
    top = max(scaled)
    exp = [math.exp(z - top) for z in scaled]
    total = math.fsum(exp)
    return [e / total for e in exp]


def summarize(examples, probabilities, bins=15):
    """Weighted CE against the target, top-label accuracy/ECE, Brier, and Score expectation error."""
    if not examples:
        return None
    totals, by_type, calibration = {}, {}, [[0.0, 0.0, 0.0] for _ in range(bins)]
    for ex, p in zip(examples, probabilities):
        t, w = ex["target"], ex["weight"]
        top = max(range(len(p)), key=p.__getitem__)
        values = {"ce": -math.fsum(ti * math.log(max(pi, 1e-12)) for ti, pi in zip(t, p) if ti),
                  "brier": math.fsum((pi - ti) ** 2 for pi, ti in zip(p, t)),
                  "accuracy": float(top == max(range(len(t)), key=t.__getitem__))}
        if ex["type"] == "score":
            values["score_abs_error"] = abs(math.fsum(i * pi for i, pi in enumerate(p)) -
                                            math.fsum(i * ti for i, ti in enumerate(t)))
        for bucket in (totals, by_type.setdefault(ex["type"], {})):
            bucket["weight"] = bucket.get("weight", 0.0) + w
            for key, value in values.items():
                bucket[key] = bucket.get(key, 0.0) + w * value
        cell = calibration[min(int(p[top] * bins), bins - 1)]
        cell[0] += w
        cell[1] += w * p[top]
        cell[2] += w * t[top]  # Expected correctness of the top label under the target distribution.

    def finish(bucket):
        # score_abs_error only accumulates over Score questions, so it has its own denominator.
        score_weight = sum(ex["weight"] for ex in examples if ex["type"] == "score")
        result = {key: value / (score_weight if key == "score_abs_error" else bucket["weight"])
                  for key, value in bucket.items() if key != "weight"}
        return {**result, "weight": bucket["weight"]}

    ece = math.fsum(abs(cell[1] - cell[2]) for cell in calibration if cell[0]) / totals["weight"]
    return {**finish(totals), "top_label_ece": ece, "questions": len(examples),
            "by_type": {typ: finish(bucket) for typ, bucket in sorted(by_type.items())}}


def by_question(examples, probabilities):
    groups = {}
    for ex, p in zip(examples, probabilities):
        pair = groups.setdefault(ex["qid"], ([], []))
        pair[0].append(ex)
        pair[1].append(p)
    return {qid: summarize(*pair) for qid, pair in sorted(groups.items())}


def fit_temperature(examples, logits):
    """One shared temperature by grid search over [0.25, 4] on the calibration split only."""
    grid = [math.exp(math.log(0.25) + i * (math.log(4.0) - math.log(0.25)) / 80) for i in range(81)]
    return min((summarize(examples, [softmax(z, t) for z in logits])["ce"], t) for t in grid)[1]


class EpochSampler:
    """Deterministic shuffled epochs; step k always sees the same questions, so runs can resume."""

    def __init__(self, items, batch_questions, seed):
        self.items, self.batch, self.seed, self.cached = items, batch_questions, seed, (None, None)

    def order(self, epoch):
        if self.cached[0] != epoch:
            indices = list(range(len(self.items)))
            random.Random(f"{self.seed}:{epoch}").shuffle(indices)
            self.cached = (epoch, indices)
        return self.cached[1]

    def batch_at(self, step):
        chosen = []
        for position in range(step * self.batch, (step + 1) * self.batch):
            epoch, offset = divmod(position, len(self.items))
            chosen.append(self.items[self.order(epoch)[offset]])
        return chosen

    def epoch_of(self, step):
        return step * self.batch / len(self.items)


def padded_tokens(group):
    return sum(len(ex["leaf_tokens"]) for ex in group) * max(len(p) for ex in group for p in ex["leaf_tokens"])


def size_sorted_groups(examples, max_questions, max_tokens):
    # Sorting by path length before packing keeps padding small within each microbatch.
    ordered = sorted(examples, key=lambda ex: max(map(len, ex["leaf_tokens"])))
    return pack_complete_questions(ordered, max_questions, max_tokens)


# ---------------------------------------------------------------------------
# Model.

def stream_weights(model, weights_path):
    """Copy a safetensors file into an existing model tensor by tensor, with strict key/shape checks."""
    import torch
    from safetensors import safe_open
    expected = model.state_dict()
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != set(expected):
            raise ValueError(f"Weight/model key mismatch: missing {sorted(set(expected) - keys)[:5]}, "
                             f"unexpected {sorted(keys - set(expected))[:5]}")
        with torch.no_grad():
            for key in sorted(keys):
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != tuple(expected[key].shape):
                    raise ValueError(f"Shape mismatch for {key}")
                expected[key].copy_(tensor)


def load_bundle_model(checkpoint_dir, device):
    """FP32 model built on the target device; weights streamed without a second full CPU copy."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    root, paths = local_checkpoint_files(checkpoint_dir)
    run_config = read_json(paths["run_config"])
    if not isinstance(run_config, dict) or run_config.get("set_head") not in {"none", "attention"}:
        raise ValueError("checkpoint config.json lacks a valid set_head")
    tokenizer = AutoTokenizer.from_pretrained(str(paths["tokenizer"]), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    body_config = AutoConfig.from_pretrained(str(paths["body_config"]), local_files_only=True, trust_remote_code=False)
    body_config.use_cache = False
    with torch.device(device):
        body = AutoModel.from_config(body_config, attn_implementation="sdpa", trust_remote_code=False).float()
        model = load_decision_model_class()(body, run_config["set_head"])
    check_rotary(body, paths["body_config"])
    stream_weights(model, paths["weights"])
    return model, tokenizer, run_config, root, paths


def autocast_for(device, precision):
    import torch
    return torch.autocast(device.type, dtype=torch.bfloat16) if precision == "bf16" else contextlib.nullcontext()


def question_losses(logits, group, rps_weight):
    """Complete-question soft cross entropy, plus an optional ranked probability score for Score."""
    import torch
    losses = []
    for z, ex in zip(logits, group):
        k = len(ex["candidate_ids"])
        target = torch.tensor(ex["target"], dtype=torch.float32, device=z.device)
        log_p = z[:k].float().log_softmax(-1)
        loss = -(target * log_p).sum()
        if rps_weight and ex["type"] == "score":
            gap = log_p.exp().cumsum(0)[:-1] - target.cumsum(0)[:-1]
            loss = loss + rps_weight * gap.pow(2).sum() / (k - 1)
        losses.append(loss)
    return torch.stack(losses)


def predict_logits(model, examples, args, device, pad_token):
    import torch
    was_training = model.training
    model.eval()
    by_id = {}
    with torch.inference_mode():
        for group in size_sorted_groups(examples, args.microbatch_questions, args.max_microbatch_tokens):
            with autocast_for(device, args.precision):
                logits, _ = model(group, pad_token)
            for ex, z in zip(group, logits):
                values = z[:len(ex["candidate_ids"])].float().cpu().tolist()
                if not all(math.isfinite(v) for v in values):
                    raise RuntimeError(f"Nonfinite logits for {ex['id']}")
                by_id[ex["id"]] = values
    model.train(was_training)
    return [by_id[ex["id"]] for ex in examples]


def evaluate(model, examples, args, device, pad_token):
    return summarize(examples, [softmax(z) for z in predict_logits(model, examples, args, device, pad_token)])


def memory_snapshot(device):
    import torch
    if device.type != "cuda":
        return {}
    return {"cuda_max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            "cuda_max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9}


def lr_factor(step, args):
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    if args.schedule == "constant":
        return 1.0
    progress = min(1.0, (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def save_weights(model, path):
    from safetensors.torch import save_file
    temporary = Path(f"{path}.partial")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, temporary)
    os.replace(temporary, path)


def train(args):
    import torch
    out = Path(args.output_dir)
    state_path = out / "resume_state.pt"
    if args.resume and not state_path.is_file():
        raise ValueError(f"--resume requires {state_path} (written by --save-resume-state)")
    if not args.resume and not args.probe_only:
        new_output_dir(out)
    device = torch.device(args.device)
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support BF16; use --precision fp32")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    model, tokenizer, run_config, root, paths = load_bundle_model(args.checkpoint_dir, device)
    max_length = args.max_length or run_config.get("max_length", 512)
    examples = load_examples(args.data, tokenizer, max_length)
    splits = {split: [ex for ex in examples if ex["split"] == split] for split in SPLITS}
    if not splits["train"] or not splits["dev"]:
        raise ValueError("Training needs nonempty train and dev splits")
    # Fail before any training if a complete question cannot fit the explicit microbatch budget.
    for group in splits.values():
        pack_complete_questions(group, args.microbatch_questions, args.max_microbatch_tokens)
    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    body = list(model.backbone.parameters())
    heads = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW([{"params": body, "lr": args.backbone_lr},
                                   {"params": heads, "lr": args.head_lr}], weight_decay=args.weight_decay)
    base_lrs = [args.backbone_lr, args.head_lr]
    sampler = EpochSampler(splits["train"], args.batch_questions, args.seed)
    prior = fit_prior(splits["train"])
    pad = tokenizer.pad_token_id
    receipt = {"base_checkpoint_dir": str(root), "base_weights_sha256": sha256_file(paths["weights"]),
               "data_sha256": sha256_file(args.data), "max_length": max_length, "precision": args.precision,
               "batch_questions": args.batch_questions, "seed": args.seed}
    counts = {s: {"questions": len(g), **{t: sum(ex["type"] == t for ex in g) for t in ("boolean", "choice", "score")}}
              for s, g in splits.items() if g}
    dev_prior = summarize(splits["dev"], [prior_probs(prior, ex) for ex in splits["dev"]])
    print(json.dumps({"loaded_seconds": time.perf_counter() - started, "splits": counts,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "dev_train_prior_ce": dev_prior["ce"], **memory_snapshot(device)}), flush=True)

    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["receipt"] != receipt:
            raise ValueError("--resume data, base weights or batch settings differ from the saved run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, best, best_step, logs = state["step"], state["best"], state["best_step"], state["logs"]
        stale = state["stale_evals"]
        del state
    else:
        # The largest microbatch runs first, so an out-of-memory failure happens in seconds, not hours.
        largest = max(size_sorted_groups(splits["train"], args.microbatch_questions, args.max_microbatch_tokens),
                      key=padded_tokens)
        model.train()
        probe_started = time.perf_counter()
        with autocast_for(device, args.precision):
            logits, _ = model(largest, pad)
        question_losses(logits, largest, args.rps_weight).sum().backward()
        # Allocate AdamW moments now too, without changing any weight.
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        probe = {"padded_tokens": padded_tokens(largest), "questions": len(largest),
                 "seconds": time.perf_counter() - probe_started, **memory_snapshot(device)}
        print(json.dumps({"memory_probe": probe}), flush=True)
        if args.probe_only:
            return
        dump(out / "memory_probe.json", probe)
        dump(out / "run_args.json", {**vars(args), "receipt": receipt})
        shutil.copytree(paths["tokenizer"], out / "tokenizer")
        shutil.copytree(paths["body_config"], out / "backbone_config")
        initial = evaluate(model, splits["dev"], args, device, pad)
        best, best_step, start_step, stale = initial["ce"], 0, 0, 0
        logs = [{"step": 0, "dev": initial}]
        print(json.dumps(logs[0]), flush=True)

    model.train()
    window, completed = time.perf_counter(), start_step
    for step in range(start_step, args.steps):
        warm = step < args.head_steps  # Optional head-only phase, e.g. for freshly initialized heads.
        for param in body:
            param.requires_grad_(not warm)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * lr_factor(step, args)
        batch = sampler.batch_at(step)
        total_weight = math.fsum(ex["weight"] for ex in batch)
        optimizer.zero_grad(set_to_none=True)
        loss_value, tokens = 0.0, 0
        # One weighted mean per optimizer step; microbatches never divide by their own size.
        for group in size_sorted_groups(batch, args.microbatch_questions, args.max_microbatch_tokens):
            with autocast_for(device, args.precision):
                logits, _ = model(group, pad)
            weights = torch.tensor([ex["weight"] for ex in group], dtype=torch.float32, device=logits.device)
            loss = (question_losses(logits, group, args.rps_weight) * weights).sum() / total_weight
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step + 1}")
            loss.backward()
            loss_value += float(loss.detach())
            tokens += padded_tokens(group)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        completed = step + 1
        item = {"step": step + 1, "phase": "head" if warm else "full", "epoch": round(sampler.epoch_of(step + 1), 3),
                "loss": loss_value, "grad_norm": float(grad_norm), "padded_tokens": tokens}
        if (step + 1) % args.log_every == 0:
            item.update(seconds_per_step=(time.perf_counter() - window) / args.log_every, **memory_snapshot(device))
            window = time.perf_counter()
        evaluate_now = not warm and ((step + 1 - args.head_steps) % args.eval_every == 0 or step + 1 == args.steps)
        if evaluate_now:
            dev = evaluate(model, splits["dev"], args, device, pad)
            item["dev"] = dev
            if dev["ce"] < best - args.min_improvement:
                best, best_step, stale = dev["ce"], step + 1, 0
                save_weights(model, out / "best.safetensors")
            else:
                stale += 1
            if args.save_resume_state:
                temporary = Path(f"{state_path}.partial")
                torch.save({"step": step + 1, "best": best, "best_step": best_step, "stale_evals": stale,
                            "logs": logs + [item], "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "receipt": receipt}, temporary)
                os.replace(temporary, state_path)
            window = time.perf_counter()
        if evaluate_now or "seconds_per_step" in item:
            logs.append(item)
            print(json.dumps(item), flush=True)
        if args.patience and stale >= args.patience:
            print(json.dumps({"early_stop": step + 1, "evals_without_improvement": stale}), flush=True)
            break

    # Step 0 is a selection candidate: if fine-tuning never beat it on dev, ship the starting weights.
    if best_step == 0:
        stream_weights(model, paths["weights"])
        shutil.copyfile(paths["weights"], out / "best.safetensors")
    else:
        stream_weights(model, out / "best.safetensors")
    logits = {s: predict_logits(model, g, args, device, pad) for s, g in splits.items() if g and s != "train"}
    temperature = fit_temperature(splits["calibration"], logits["calibration"]) if splits["calibration"] else 1.0
    final = {split: {"model": summarize(splits[split], [softmax(z) for z in values]),
                     "model_calibrated": summarize(splits[split], [softmax(z, temperature) for z in values]),
                     "train_prior": summarize(splits[split], [prior_probs(prior, ex) for ex in splits[split]])}
             for split, values in logits.items()}
    report = {"best_step": best_step, "selected_on": "dev weighted cross entropy (step 0 included)",
              "temperature": {"value": temperature, "fitted_on": "calibration" if splits["calibration"] else None},
              "splits": counts, "final": final, "logs": logs, "memory": memory_snapshot(device),
              "total_seconds": time.perf_counter() - started}
    for split in ("test", "ood"):
        if split in logits:
            report[f"{split}_by_question"] = by_question(splits[split], [softmax(z, temperature) for z in logits[split]])
    dump(out / "report.json", report)
    config = {**run_config, "max_length": max_length,
              "post_training": {"method": "full_fine_tune", "best_step": best_step, "steps_run": completed,
                                "previous": run_config.get("post_training"), **receipt, "recommended_temperature": temperature,
                                "temperature_note": "Pass explicitly to DecisionPredictor.predict(temperature=...); "
                                                    "the runtime default stays 1.",
                                "weights_sha256": sha256_file(out / "best.safetensors")}}
    dump(out / "config.json", config)
    print(json.dumps({"done": str(out), "best_step": best_step, "temperature": temperature,
                      **{f"{s}_ce": {k: v["ce"] for k, v in m.items()} for s, m in final.items()}}), flush=True)


def init_bundle(args):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer
    out = new_output_dir(args.output_dir)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    body = AutoModel.from_pretrained(args.model, revision=args.revision, attn_implementation="sdpa").float()
    model = load_decision_model_class()(body, args.set_head)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, out / "best.safetensors")
    tokenizer.save_pretrained(out / "tokenizer")
    body.config.save_pretrained(out / "backbone_config")
    dump(out / "config.json", {"model": args.model, "revision": args.revision,
                               "resolved_model_revision": getattr(body.config, "_commit_hash", None),
                               "set_head": args.set_head, "max_length": args.max_length, "seed": args.seed,
                               "note": "Untuned backbone with freshly initialized decision heads."})
    print(json.dumps({"bundle": str(out), "parameters": sum(t.numel() for t in model.parameters())}), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    t = commands.add_parser("train", help="full fine-tuning with dev-only selection")
    t.add_argument("--checkpoint-dir", required=True, help="bundle: config.json, best.safetensors, tokenizer/, backbone_config/")
    t.add_argument("--data", required=True, help="JSONL rows: id, split, state, questions, gold and/or gold_probs")
    t.add_argument("--output-dir", required=True, help="new empty directory; becomes a NanoJev bundle")
    t.add_argument("--device", default="cuda")
    t.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                   help="bf16: FP32 weights with BF16 autocast (release recipe); fp32 disables autocast")
    t.add_argument("--max-length", type=int, help="per candidate path; default from bundle config; never truncated")
    t.add_argument("--steps", type=int, default=600)
    t.add_argument("--batch-questions", type=int, default=24, help="complete questions per optimizer step")
    t.add_argument("--microbatch-questions", type=int, default=8)
    t.add_argument("--max-microbatch-tokens", type=int, default=16384,
                   help="candidate paths x longest path per forward; the main activation-memory knob")
    t.add_argument("--backbone-lr", type=float, default=1e-5)
    t.add_argument("--head-lr", type=float, default=1e-4)
    t.add_argument("--head-steps", type=int, default=0, help="initial head-only updates (use with init-bundle)")
    t.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    t.add_argument("--warmup-steps", type=int, default=0)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--rps-weight", type=float, default=0.0,
                   help="weight of the ranked probability score added to CE for ordered Score questions")
    t.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    t.add_argument("--eval-every", type=int, default=50)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--patience", type=int, default=0, help="stop after this many evals without dev improvement; 0 = off")
    t.add_argument("--min-improvement", type=float, default=0.0)
    t.add_argument("--save-resume-state", action="store_true",
                   help="write model+optimizer (~7.2 GB) at every eval so --resume can continue")
    t.add_argument("--resume", action="store_true", help="continue from <output-dir>/resume_state.pt")
    t.add_argument("--seed", type=int, default=17)
    t.add_argument("--probe-only", action="store_true",
                   help="run the largest microbatch and one optimizer allocation, report peak memory, write nothing")

    i = commands.add_parser("init-bundle", help="untuned backbone + fresh heads, for a control arm")
    i.add_argument("--model", default="Qwen/Qwen3-0.6B")
    i.add_argument("--revision", default="main")
    i.add_argument("--set-head", choices=["none", "attention"], default="attention")
    i.add_argument("--max-length", type=int, default=2048)
    i.add_argument("--seed", type=int, default=17)
    i.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        positive = [args.steps, args.batch_questions, args.microbatch_questions, args.eval_every, args.log_every]
        if min(positive) <= 0 or min(args.max_microbatch_tokens, args.warmup_steps, args.head_steps, args.patience) < 0:
            parser.error("steps, batch sizes and intervals must be positive; budgets and counts nonnegative")
        if args.head_steps >= args.steps:
            parser.error("--head-steps must be smaller than --steps")
        train(args)
    else:
        init_bundle(args)


if __name__ == "__main__":
    main()
