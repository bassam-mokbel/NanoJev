# Post-training NanoJev on market research surveys

This guide fully fine-tunes the released NanoJev checkpoint on your own survey data, using the
same recipe as the released model. Two scripts do the work:

- [`prepare_survey_data.py`](../scripts/prepare_survey_data.py) converts a respondent CSV and a
  question spec into training rows.
- [`train_survey_decisions.py`](../scripts/train_survey_decisions.py) trains every parameter and
  writes an ordinary NanoJev bundle.

A synthetic example lives in [`configs/survey_example/`](../configs/survey_example/).

## What the model learns

Every survey question becomes a NanoJev primitive, asked about one respondent (or segment):

| Survey item | Primitive | What the model returns |
|---|---|---|
| Single-select (brand used most) | `choice` | A distribution over the offered options |
| Likert, satisfaction, NPS | `score` | A distribution over 2–10 ordered levels, plus the expected level |
| Yes/no, awareness, one option of a multi-select | `boolean` | P(true) |

The **state** is the text the model conditions on: demographics, behaviours, earlier answers, or
an open-ended verbatim. Training minimizes cross entropy between the model's distribution and the
observed answer (respondent mode), or the segment's weighted answer shares (aggregate mode).

Every run also scores a **train answer-share baseline**: the weighted marginal distribution of
each question in the training split. If the model does not beat this baseline on test, it has
learned nothing beyond the crosstab.

## 1. Hardware

The recipe keeps FP32 weights and AdamW state and runs the forward pass in BF16:

| Item | Memory |
|---|---:|
| FP32 weights, 596M parameters | 2.4 GB |
| Gradients | 2.4 GB |
| AdamW moments | 4.8 GB |
| BF16 weight casts, activations with checkpointing (16k-token microbatch), CUDA context | about 3–4 GB |
| **Total** | **about 13–14 GB** |

- **Recommended:** any NVIDIA GPU with BF16 and at least 24 GB, for example L4, A10, RTX 3090/4090, A5000, A100 or H100.
- **16 GB GPUs** work with `--max-microbatch-tokens 8192`.
- **Pre-Ampere GPUs** (V100, T4) lack BF16. They need `--precision fp32`, which is slower and needs more activation memory.
- **Other resources:** about 15 GB of disk per run, and system RAM roughly twice the size of your tokenized dataset.

The script measures real peak memory before training starts (step 7), so you don't have to trust these estimates.

## 2. Environment

```bash
git clone <your fork> NanoJev && cd NanoJev
python3 -m venv .venv && source .venv/bin/activate
```

Install a CUDA build of PyTorch for your driver ([pytorch.org selector](https://pytorch.org/get-started/locally/)), then:

```bash
pip install "transformers>=5.17" "safetensors>=0.8" tokenizers huggingface_hub
```

Alternatively, `pip install -e ".[dev]"` installs the same dependencies from `pyproject.toml` and adds the `nanojev` command.
Each `python scripts/prepare_survey_data.py …` below can then be run as `nanojev survey prepare …`, and each
`python scripts/train_survey_decisions.py train|init-bundle …` as `nanojev survey train|init-bundle …`.

**transformers 5 is required for the released checkpoint.** Its `backbone_config/config.json` stores
the RoPE base (1,000,000) under the transformers 5 `rope_parameters` field. transformers 4.x
ignores that field and silently builds a base of 10,000, which scrambles every position encoding.
The trainer recomputes the rotary frequencies the config file declares, compares them with the
ones the model actually built, and stops with an error on a mismatch.

The release was trained with Python 3.14, torch 2.14 and transformers 5.17
([requirements-toy.txt](../requirements-toy.txt)). These scripts were also tested on CPU with
Python 3.10, torch 2.7 and transformers 5.18. `--disable-native-triton` is not needed here.

Run the offline tests:

```bash
python -m unittest discover -s scripts -p test_survey_training.py -v
```

## 3. Download the starting checkpoint

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="C-Tianyu/NanoJev", revision="unified-games-v1",
                  local_dir="checkpoints/NanoJev-unified",
                  allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"])
```

## 4. Map your questionnaire onto primitives

Five rules come from how NanoJev encodes inputs. The converter enforces the first three.

1. **Choice option keys are model input.** Each candidate is encoded as `key: description`. A key like `"3"` carries no meaning, so use `"corner_roast"`.
2. **Score levels are judged one at a time.** Each level description is encoded alone, without its number or its neighbours. Write self-contained levels like "Somewhat satisfied with their usual brand", not "4".
3. **At most 10 Score levels.** Bucket an 11-point NPS scale into, for example, detractor, passive and promoter levels, each with its own description.
4. **Multi-select questions become one Boolean per option.** Questions are independent: none sees another's answer.
5. **Never put the answer, or anything derived from it, in the state.** The converter rejects a question column used as a state field. Derived variables, such as a loyalty segment computed from the brand question, are your responsibility.

Questions are answered independently. If you want "given their earlier answers" prediction, put
those earlier answers in the state.

**Open-ended coding** fits the same format:
- **State:** the verbatim (plus profile fields if useful).
- **Mutually exclusive codes:** one Choice question over the codeframe.
- **Multi-label codes:** one Boolean per code.

## 5. Write a spec and convert

Export your survey to CSV with one row per respondent and raw codes. From SPSS, `pandas.read_spss(path, convert_categoricals=False)` keeps the codes.

Then copy [`configs/survey_example/spec.json`](../configs/survey_example/spec.json) and edit it:

| Spec key | Meaning |
|---|---|
| `id_column` | Respondent id |
| `weight_column` | Optional survey weight. Use it to learn population rather than sample distributions. |
| `group_column` | Optional. Keeps households or panels together in one split. Defaults to the respondent id. |
| `missing_codes` | Codes treated as unanswered, for example refused or skipped. Those questions are left out for that respondent. |
| `state.fields` | Profile columns, each with a label and optional `values` mapping codes to text |
| `questions` | Each entry has `id`, `column`, `type` and `instructions`, plus `options` (choice), `levels` (score) or `true_codes`/`false_codes` (boolean) |
| `splits` | Fractions for `train`, `dev`, `calibration`, `test`, and optionally `ood` |

```bash
python scripts/prepare_survey_data.py --csv my_survey.csv --spec my_spec.json --output data/survey/rows.jsonl
```

Splits are assigned by a hash of the respondent (or group) id, so they are stable across reruns.

For a real generalization test, hold out a **whole wave, market or brand** yourself: set those rows' `split` to `ood` in the JSONL. Random respondent splits only test new people from the same survey.

`--aggregate` merges respondents whose rendered state is identical and writes the weighted answer shares as soft targets. This only helps when the state has a few coarse fields. With many fields, nearly every segment holds one respondent.

The training JSONL can also be written by your own code. Each row looks like this:

```json
{"id": "R0001", "split": "train", "weight": 1.3, "group": "household-17",
 "state": "Market research survey respondent ...\n- Age: 25-34\n- Region: South",
 "questions": {"brand_pref": {"type": "choice", "instructions": "Which coffee brand ...?",
                              "criteria": {"brightbean": "BrightBean, a premium ...", "store_brand": "The supermarket's own ..."}}},
 "gold": {"brand_pref": "store_brand"}}
```

`gold` holds a choice key, a score level index or a Boolean. `gold_probs` holds a full distribution
and takes precedence over `gold`. Every question in a row needs one or the other.

## 6. Optional: dry run on the synthetic example

The example is a fictional coffee survey of 240 respondents. It runs in a few minutes:

```bash
python scripts/prepare_survey_data.py --csv configs/survey_example/responses.csv \
  --spec configs/survey_example/spec.json --output data/survey_example/rows.jsonl
python scripts/train_survey_decisions.py train --checkpoint-dir checkpoints/NanoJev-unified \
  --data data/survey_example/rows.jsonl --output-dir runs/survey_example --steps 100 --eval-every 20
```

## 7. Probe memory

```bash
python scripts/train_survey_decisions.py train --checkpoint-dir checkpoints/NanoJev-unified \
  --data data/survey/rows.jsonl --output-dir runs/probe --probe-only
```

This runs the largest training microbatch forward and backward, allocates the optimizer state,
prints peak CUDA memory, and writes nothing. If it runs out of memory, halve
`--max-microbatch-tokens` (the budget is candidate paths × longest path).

A single question that exceeds the budget is an error. A question's candidates are never split
across microbatches, because the softmax must cover the complete question.

## 8. Train

```bash
python scripts/train_survey_decisions.py train \
  --checkpoint-dir checkpoints/NanoJev-unified \
  --data data/survey/rows.jsonl \
  --output-dir runs/survey_v1 \
  --steps 1500 --batch-questions 24 --eval-every 50 --patience 6 \
  --backbone-lr 1e-5 --head-lr 1e-4 --rps-weight 0.5 --save-resume-state
```

| Flag | What it controls |
|---|---|
| `--steps` | Optimizer updates. Each update sees `--batch-questions` questions, so steps ≈ epochs × train questions / 24. Start with 2–3 epochs and let dev selection and `--patience` stop early. |
| `--backbone-lr` / `--head-lr` | Learning rates. These defaults are the released `hard_lr1e5` recipe. On a few thousand questions, try 5e-6 if dev CE rises early. |
| `--rps-weight` | Adds a ranked probability score for Score questions, so predicting "4" when the answer is "5" costs less than predicting "1". 0 gives plain cross entropy, like the release. |
| `--schedule cosine --warmup-steps 50` | Optional decay schedule. The release used a constant rate. |
| `--save-resume-state` / `--resume` | Writes about 7 GB of model and optimizer state at each eval. After an interruption, rerun the same command with `--resume`; it continues from that eval with the same data order. (Bit-exact on CPU; BF16 GPU kernels are not bitwise deterministic.) |
| `--max-length` | Per-path token limit. Inputs are never truncated: an over-long path is an error. Raise it if your states are long. |

**Monitoring:**
- **Training output:** each eval prints one JSON line with dev metrics.
- **GPU:** watch it with `nvidia-smi`.
- **Speed:** `seconds_per_step` gives the real throughput. Cost grows with candidates × state length, because each candidate re-encodes the full state, so keep states compact.

**What gets saved:**
- `best.safetensors` is overwritten only when weighted dev cross entropy improves.
- Step 0 (the starting weights) is always a candidate. If fine-tuning never helps, the run ships the original weights and reports `best_step: 0`.

### Control arm: start from untuned Qwen3

The released checkpoint's most recent updates were on game data. It is worth testing whether that
helps or hurts on surveys. The same arm also gives a lineage free of Jev-derived training targets
(the released checkpoint's Maze and Snake targets came from the Jev API).

```bash
python scripts/train_survey_decisions.py init-bundle --model Qwen/Qwen3-0.6B --output-dir checkpoints/qwen3-fresh-heads
python scripts/train_survey_decisions.py train --checkpoint-dir checkpoints/qwen3-fresh-heads \
  --data data/survey/rows.jsonl --output-dir runs/survey_v1_qwen --head-steps 100 \
  --steps 1600 --batch-questions 24 --eval-every 50 --patience 6 --rps-weight 0.5
```

`--head-steps` trains only the freshly initialized heads first, so random heads don't push large
gradients into the backbone. Choose between arms on **dev**, then look at test once.

## 9. Read the report

`runs/<name>/report.json` contains:

- **`final.<split>.model`:** weighted cross entropy (`ce`), Brier score, top-label accuracy and ECE, and `score_abs_error` (expected level against the true level). Each is also broken down by question type.
- **`final.<split>.model_calibrated`:** the same metrics after the temperature fitted on the `calibration` split. The value is in `temperature`.
- **`final.<split>.train_prior`:** the answer-share baseline. **The model must beat it to be useful.**
- **`test_by_question`:** per-question metrics. Expect some questions to be predictable from the profile and others not.

Accuracy is a weak metric for survey prediction. Many answers are genuinely uncertain given a
profile, so cross entropy and Brier against the observed answers are the main measures.

For segment-level share estimates, aggregate the predicted distributions over the respondents in
a held-out segment and compare them with the observed shares.

The temperature is fitted on calibration data from the same distribution. Check that it improves
`test` before using it, and don't assume it transfers to a new market or wave.

## 10. Use the model

The output directory is a standard bundle (`config.json`, `best.safetensors`, `tokenizer/`, `backbone_config/`):

```python
import sys; sys.path.insert(0, "scripts")
from predict_toy_decisions import DecisionPredictor

engine = DecisionPredictor("runs/survey_v1", precision="bf16")
result = engine.predict({"states": [{"id": "p1", "state": "Market research survey respondent ...\n- Age: 18-24",
                                     "questions": {"brand_pref": {"type": "choice", "instructions": "...",
                                                                  "criteria": {"brightbean": "...", "store_brand": "..."}}}}]},
                        temperature=1.0)  # or config.json post_training.recommended_temperature
```

Use exactly the instruction and option text from training; the model has only learned those wordings.

`python scripts/serve_decisions.py --checkpoint-dir runs/survey_v1` serves the same model over HTTP
at `POST /api/evaluate`. It always uses temperature 1 and caps each request at 32 states, 96
questions and 256 candidate paths, so use `DecisionPredictor` directly for batch scoring.

### Inference on a Jetson Orin Nano Super (8 GB)

Training doesn't fit on the device, but inference does: the fine-tuned bundle is 2.4 GB of FP32
weights. These steps come from NVIDIA's documentation as of October 2026 and have **not** been run
end to end with this model.

1. **Flash JetPack 7.2.1** (Jetson Linux 39.2.1: Ubuntu 24.04, CUDA 13.2, Python 3.12). Source: [JetPack downloads](https://developer.nvidia.com/embedded/jetpack/downloads).
2. **Install the upstream PyTorch aarch64 build:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   ```
   Then install transformers ≥ 5.17 as above. Avoid NVIDIA's older `2.5.0a0` Jetson wheels: transformers 5 requires torch ≥ 2.5 and treats `2.5.0a0` as older, so it disables PyTorch. A "compute capability 8.7" warning at startup is harmless.
3. **Use maximum performance and free memory:**
   ```bash
   sudo nvpmodel -m 2
   sudo jetson_clocks
   sudo systemctl set-default multi-user.target
   ```
   Mode 2 is MAXN_SUPER. Switching to headless frees about 0.8 GB; swap does not help here, because GPU allocations cannot be swapped.
4. **Load the model.** `DecisionPredictor` builds the model on the CPU and then copies it to the GPU. Both copies live in the same 8 GB, so loading briefly peaks at about 5 GB. Measure it with `sudo tegrastats` or `jtop`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `requires N padded tokens, over budget` | One question has too many or too long candidates. Raise `--max-microbatch-tokens`, or shorten the state and descriptions. |
| `exceeds max_length` | Raise `--max-length`. Paths are never truncated. |
| `group ... appears in both` | A respondent or household is in two splits. Fix the split assignment. |
| `Rotary embedding does not match rope_theta` | transformers is older than 5.17. Upgrade it. |
| CUDA out of memory | Lower `--max-microbatch-tokens` and rerun `--probe-only`. |
| Dev CE never beats step 0 | Lower the learning rates, check the data for leakage or label errors, and compare against the Qwen control arm. |
| Model CE ≈ `train_prior` CE | The state carries little information about the answer. Add informative profile fields or earlier answers. |

## Privacy

Fine-tuned models can memorize training text. Before conversion:
- **Remove direct identifiers** (names, emails, phone numbers, free text containing them) from state fields.
- **Check your data permissions:** make sure respondent consent and your data agreements allow training models on the responses.
