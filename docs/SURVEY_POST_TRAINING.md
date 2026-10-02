# Post-training NanoJev on market research surveys

This guide fully fine-tunes the released NanoJev checkpoint on your own survey data, using the
same recipe as the released model. The inputs are [qupa-datatypes](https://bitbucket.org/interrogaregmbh/qupa-datatypes)
models: a `Survey` definition and its `SurveyResponse` records, each a list of `SurveyAnswer`s.

- [`prepare_survey_data.py`](../scripts/prepare_survey_data.py) turns surveys and responses into
  NanoJev training rows, as described by a **training spec** (JSON).
- [`train_survey_decisions.py`](../scripts/train_survey_decisions.py) trains every parameter and
  writes an ordinary NanoJev bundle. It takes the spec directly (`--spec`) or prepared rows (`--data`).

A synthetic example lives in [`configs/survey_example/`](../configs/survey_example/): a fictional
coffee survey with 240 generated respondents, built with qupa models by `generate_example.py`.

## What the model learns

Each training row asks how one respondent answers one survey question, given that respondent's
other answers. Targets come from the full question definition, so options the respondent did not
choose are training signal too.

| qupa question | NanoJev primitive | Target |
|---|---|---|
| `SingleQuestion` or grid row, ordered `Scale` with 2–10 substantive points | `score` | Selected point in `Scale.analytical_points()` order (low to high) |
| …the same, but the scale also has non-substantive points ("don't know") | extra `boolean` | Whether the respondent chose one of them; the score is trained only on substantive answers |
| Any other `SingleQuestion` or grid row: `ChoiceSet`, nominal or unspecified scales, more than 10 points | `choice` | Selected option, over every value |
| `MultiQuestion`, `MultiGridQuestion` row | one `boolean` per option | Whether that option was selected |
| `RankingQuestion` | sequential `choice`s | Rank 1 among all items, rank 2 among the remaining ones, and so on |
| `MaxDiffQuestion` | two `choice`s per task | Best among the presented items, then worst among the rest |
| Text, numeric, text-list and numeric-list questions | none | Used as context only |

The **state** is the respondent's other answers, verbalized through qupa's resolution layer: one
line per question with its participant-facing wording, rows indented beneath grids, and loop items
in brackets. Loop-scoped answers become separate targets whose question names the loop item.

Every run also scores a **train answer-share baseline**: the weighted marginal distribution of
each target in the training split. If the model does not beat this baseline on test, it has
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

The script measures real peak memory before training starts (step 6), so you don't have to trust these estimates.

## 2. Environment

The project uses [uv](https://docs.astral.sh/uv/). `qupa-datatypes` is pinned in `pyproject.toml` to
a tag in a private Bitbucket repository, so the machine needs SSH access to it (for example a
deploy key).

```bash
git clone <your fork> NanoJev && cd NanoJev
uv sync --extra dev
uv run pytest scripts/test_survey_training.py
```

On Linux, the PyPI torch wheel includes CUDA. If your driver needs a different CUDA build, point
uv at the matching PyTorch index (see the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)).

The commands below use `uv run nanojev …`. Each one is equivalent to `python scripts/<module>.py`:
`survey prepare` is `prepare_survey_data.py`, and `survey train|init-bundle` is
`train_survey_decisions.py train|init-bundle`.

**transformers 5 is required for the released checkpoint.** Its `backbone_config/config.json` stores
the RoPE base (1,000,000) under the transformers 5 `rope_parameters` field. transformers 4.x
ignores that field and silently builds a base of 10,000, which scrambles every position encoding.
The loaders recompute the rotary frequencies the config file declares, compare them with the
ones the model actually built, and stop with an error on a mismatch.

## 3. Download the starting checkpoint

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="C-Tianyu/NanoJev", revision="unified-games-v1",
                  local_dir="checkpoints/NanoJev-unified",
                  allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"])
```

## 4. Write a training spec

A spec lists the data sources and decides which answers are targets and which are context.
Relative paths resolve from the spec file. Start from [`configs/survey_example/training_spec.json`](../configs/survey_example/training_spec.json):

```json
{
  "sources": [
    {"name": "coffee_w1", "survey": "w1/survey.json", "responses": "w1/responses.jsonl"},
    {"name": "coffee_w2", "survey": "w2/survey.json", "responses_csv": "w2/rawdata.csv",
     "response_map": "w2/response-map.json", "split": "ood"}
  ],
  "context": "preceding",
  "weight_key": "weight",
  "context_exclude_kinds": ["text"]
}
```

**Sources.** Each source is one survey revision:
- `responses`: `SurveyResponse` records as JSONL, a JSON list, or one object.
- `responses_csv` with `response_map`: a raw export imported through qupa's `import_responses_csv`. Unmapped columns are counted in the conversion report.
- `split` (optional): puts the whole source into one split. Use `"ood"` to hold out a wave, market or survey, which is a stronger generalization test than new respondents from the same survey.

Every response is validated with `SurveyResponse.validate_against(survey)` before use.

**Fields:**

| Field | Default | Meaning |
|---|---|---|
| `targets` | `"all"` | `"all"` visible closed questions, or a list of question ids |
| `exclude` | `[]` | Question ids never used as targets |
| `context` | `"preceding"` | `"preceding"`: only earlier survey elements. `"all_other"`: every other answer. `"listed"`: exactly `context_questions` (one row per respondent, and those questions are never targets) |
| `context_exclude`, `context_exclude_kinds` | `[]` | Question ids or kinds (for example `"text"`) never shown in a state |
| `max_state_chars` | 4000 | Context budget per state. The nearest questions are kept first |
| `max_question_chars` | 40000 | Every candidate repeats the state, so the state budget is also capped at this ÷ the row's largest candidate count. Keep it near 3× `--max-microbatch-tokens` |
| `max_context_answers`, `max_line_chars` | 40, 300 | Further caps on context size |
| `include_hidden` | `false` | Use `VariableType.HIDDEN` questions as targets and context. They are often recodes of other answers, so they leak |
| `include_synthetic` | `false` | Use answers with `is_synthetic: true`. Training a twin on twins' answers is circular |
| `non_substantive` | `"boolean"` | `"boolean"` adds the don't-know target described above; `"skip"` drops those answers |
| `weight_key`, `group_key` | none | `SurveyResponse.custom_meta` keys for the survey weight and a split group (for example a household). With CSV imports, map them via the response map's `metadata_columns` |
| `splits`, `split_seed` | 70/10/10/10 | Fractions over `train`, `dev`, `calibration`, `test` and optionally `ood`. Assigned by a hash of the group, so stable across reruns |
| `scale_verbalizations`, `verbalization_language` | none | A JSON/JSONL list of qupa `ScaleVerbalization` records and the language (`DE` or `EN`) to use |
| `skip_invalid_responses` | `false` | Skip responses that fail validation, listing them in the report, instead of stopping |

### Why "preceding" is the default

Routing only flows forward. A follow-up asked only of brand users reveals that a respondent uses
the brand, so with `all_other` a later filtered question can leak the target's answer. `preceding`
context cannot contain a question routed on the target. The same caution applies to `listed`
questions that come after a target in the survey.

### Score levels must read on their own

NanoJev judges every Score level separately, without its position or its neighbours. Many
questionnaires label only the endpoints (`1 = Not at all`, `2`, … `7 = Completely`), and a bare "4"
means nothing alone. The converter therefore uses, in order:

1. a stored `ScaleVerbalization` label for the point, matched by `scale_content_fingerprint`, when the spec provides one;
2. otherwise, on an ordered scale with bare-number points, the point plus the labelled endpoints: `4 [1 = Not at all … 7 = Completely]`.

Choice candidates are encoded as `option_id: text`, so qupa ids like `code_3` appear in model inputs.
They are harmless next to the text, but descriptive ids are better.

## 5. Convert and inspect

```bash
uv run nanojev survey prepare --spec my_spec.json --output data/survey/rows.jsonl
```

This writes `rows.jsonl` and `rows.report.json`. The report records:
- **source hashes**: the SHA256 of every input file, plus qupa's survey fingerprint;
- **counts**: rows per split and targets per kind and role;
- **what was left out and why**: for example `not_a_target:text:visible`, `synthetic_answers`, or `context_answers_over_budget`.

Read a few rows before training. Each row carries `meta.targets`, which maps every NanoJev question
back to its survey question, row, option and loop scope. It also carries `meta.context_questions`,
the questions shown in its state.

The rows format is generic, so other tools can write it too:

```json
{"id": "coffee_w1:R0001:q_brand", "split": "train", "weight": 1.2, "group": "coffee_w1:R0001",
 "state": "Survey: …\nAnswers this respondent gave:\n- Where do you live?: South",
 "questions": {"q_brand": {"type": "choice", "instructions": "Survey question: …",
                           "criteria": {"brightbean": "BrightBean, …", "store_brand": "The supermarket's own …"}}},
 "gold": {"q_brand": "store_brand"}, "meta": {"targets": {"q_brand": {"question_id": "q_brand"}}}}
```

`gold` holds a choice key, a score level index or a Boolean. `gold_probs` holds a full distribution
and takes precedence over `gold`.

## 6. Probe memory

```bash
uv run nanojev survey train --checkpoint-dir checkpoints/NanoJev-unified \
  --spec my_spec.json --output-dir runs/probe --probe-only
```

This runs the largest training microbatch forward and backward, allocates the optimizer state,
prints peak CUDA memory, and writes nothing. If it runs out of memory, halve
`--max-microbatch-tokens` and lower the spec's `max_question_chars` to match.

A single question that exceeds the budget is an error. A question's candidates are never split
across microbatches, because the softmax must cover the complete question.

To try the pipeline first, run the same commands on `configs/survey_example/training_spec.json`
with `--steps 100 --eval-every 20`.

## 7. Train

```bash
uv run nanojev survey train \
  --checkpoint-dir checkpoints/NanoJev-unified \
  --spec my_spec.json \
  --output-dir runs/survey_v1 \
  --steps 1500 --batch-questions 24 --eval-every 50 --patience 6 \
  --backbone-lr 1e-5 --head-lr 1e-4 --rps-weight 0.5 --save-resume-state
```

With `--spec`, the converted rows and their report are saved in the run directory (`rows.jsonl`,
`conversion_report.json`), and the rows' SHA256 is recorded in the bundle's `config.json`.

| Flag | What it controls |
|---|---|
| `--steps` | Optimizer updates. Each update sees `--batch-questions` questions, so steps ≈ epochs × train questions / 24. Start with 2–3 epochs and let dev selection and `--patience` stop early. |
| `--backbone-lr` / `--head-lr` | Learning rates. These defaults are the released `hard_lr1e5` recipe. On a few thousand questions, try 5e-6 if dev CE rises early. |
| `--rps-weight` | Adds a ranked probability score for Score questions, so predicting "4" when the answer is "5" costs less than predicting "1". 0 gives plain cross entropy, like the release. |
| `--schedule cosine --warmup-steps 50` | Optional decay schedule. The release used a constant rate. |
| `--save-resume-state` / `--resume` | Writes about 7 GB of model and optimizer state at each eval. After an interruption, rerun the same command with `--resume`. It reconverts the spec and refuses to continue if the rows changed. It then continues from that eval with the same data order. (Bit-exact on CPU; BF16 GPU kernels are not bitwise deterministic.) |
| `--max-length` | Per-path token limit. Inputs are never truncated: an over-long path is an error. |

**Monitoring:**
- **Training output:** each eval prints one JSON line with dev metrics.
- **GPU:** watch it with `nvidia-smi`.
- **Speed:** `seconds_per_step` gives the real throughput. Cost grows with candidates × state length, because each candidate re-encodes the full state.

**What gets saved:**
- `best.safetensors` is overwritten only when weighted dev cross entropy improves.
- Step 0 (the starting weights) is always a candidate. If fine-tuning never helps, the run ships the original weights and reports `best_step: 0`.

### Control arm: start from untuned Qwen3

The released checkpoint's most recent updates were on game data. It is worth testing whether that
helps or hurts on surveys. The same arm also gives a lineage free of Jev-derived training targets
(the released checkpoint's Maze and Snake targets came from the Jev API).

```bash
uv run nanojev survey init-bundle --model Qwen/Qwen3-0.6B --output-dir checkpoints/qwen3-fresh-heads
uv run nanojev survey train --checkpoint-dir checkpoints/qwen3-fresh-heads \
  --spec my_spec.json --output-dir runs/survey_v1_qwen --head-steps 100 \
  --steps 1600 --batch-questions 24 --eval-every 50 --patience 6 --rps-weight 0.5
```

`--head-steps` trains only the freshly initialized heads first, so random heads don't push large
gradients into the backbone. Choose between arms on **dev**, then look at test once.

## 8. Read the report

`runs/<name>/report.json` contains:

- **`final.<split>.model`:** weighted cross entropy (`ce`), Brier score, top-label accuracy and ECE, and `score_abs_error` (expected level against the true level). Each is also broken down by NanoJev question type.
- **`final.<split>.model_calibrated`:** the same metrics after the temperature fitted on the `calibration` split. The value is in `temperature`.
- **`final.<split>.train_prior`:** the answer-share baseline. **The model must beat it to be useful.**
- **`test_by_question`** (and `ood_by_question`): metrics per **survey question** id, aggregated over its rows, options, ranks and loop items.

Accuracy is a weak metric for survey prediction. Many answers are genuinely uncertain given a
profile, so cross entropy and Brier against the observed answers are the main measures.

The temperature is fitted on calibration data from the same distribution. Check that it improves
`test` before using it, and don't assume it transfers to a new market or wave.

## 9. Use the model

The output directory is a standard bundle (`config.json`, `best.safetensors`, `tokenizer/`, `backbone_config/`).
Requests must use exactly the wording the converter produces. To score respondents whose answers
you already have, for example a new wave held out for evaluation, build rows from a spec and send
each row's `state` and `questions`:

```python
from nanojev import DecisionPredictor
from nanojev.survey import build_rows, load_spec

rows, _ = build_rows(*load_spec("holdout_spec.json"))
engine = DecisionPredictor("runs/survey_v1", precision="bf16")
result = engine.predict({"states": [{"id": r["id"], "state": r["state"], "questions": r["questions"]} for r in rows[:32]]},
                        temperature=1.0)  # or config.json post_training.recommended_temperature
```

Each answer's probabilities map back to qupa identifiers through the row's `meta.targets`:
- choice keys are option, point or item ids;
- score levels follow `analytical_points()`.

`build_rows` creates targets only for answered questions, because each needs its observed answer.
Predicting questions a respondent has **not** answered needs a request builder that renders the
same state and questions without labels. Turning the predictions into synthetic `SurveyAnswer`s
(`is_synthetic=True`) is a further step. Neither is part of this pipeline yet.

`uv run nanojev serve --checkpoint-dir runs/survey_v1` serves the same model over HTTP at
`POST /api/evaluate`. It always uses temperature 1 and caps each request at 32 states, 96
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
   Then install transformers ≥ 5.17. Avoid NVIDIA's older `2.5.0a0` Jetson wheels: transformers 5 requires torch ≥ 2.5 and treats `2.5.0a0` as older, so it disables PyTorch. A "compute capability 8.7" warning at startup is harmless.
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
| `source …, respondent …: …` during conversion | A response does not validate against its survey. Fix the data, or set `skip_invalid_responses` and read `invalid_responses` in the report. |
| `requires N padded tokens, over budget` | One question has too many or too long candidates. Raise `--max-microbatch-tokens`, or lower `max_question_chars` in the spec. |
| `exceeds max_length` | Raise `--max-length`, or lower `max_state_chars`. Paths are never truncated. |
| `group ... appears in both` | A respondent or group is in two splits. With `group_key`, check that a group belongs to one source. |
| `Rotary embedding does not match rope_theta` | transformers is older than 5.17. Upgrade it. |
| CUDA out of memory | Lower `--max-microbatch-tokens` and rerun `--probe-only`. |
| Dev CE never beats step 0 | Lower the learning rates, check for leakage or label errors, and compare against the Qwen control arm. |
| Model CE ≈ `train_prior` CE | The context carries little information about the answer. Try `all_other` context (mind routing leakage) or a larger `max_state_chars`. |

## Privacy

Fine-tuned models can memorize training text, and open answers flow into states as context.
- **Keep identifying verbatims out:** list them in `context_exclude`, or exclude whole kinds with `context_exclude_kinds: ["text", "text_list"]`.
- **Check your data permissions:** make sure respondent consent and your data agreements allow training models on the responses.
