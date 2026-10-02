#!/usr/bin/env python3
"""Offline checks for survey conversion and full fine-tuning.

Run: python3 -m unittest discover -s scripts -p test_survey_training.py -v
Data checks need only the standard library. Training checks build a two-layer random Qwen3
with a character tokenizer on CPU; they download nothing and skip without torch/transformers.
"""
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import prepare_survey_data as survey
from predict_toy_decisions import prepare_examples
import train_survey_decisions as trainer

EXAMPLE = Path(__file__).resolve().parent.parent / "configs" / "survey_example"
HAS_TORCH = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors", "tokenizers"))


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]


def example_spec():
    return json.loads((EXAMPLE / "spec.json").read_text(encoding="utf-8"))


def write_rows(directory, rows):
    path = Path(directory) / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def choice_row(identifier="r1", split="train", **extra):
    return {"id": identifier, "split": split, "state": "Respondent profile: age 25-34.",
            "questions": {"brand": {"type": "choice", "instructions": "Which brand do they buy?",
                                    "criteria": {"acme": "Acme coffee", "zenith": "Zenith coffee"}}},
            "gold": {"brand": "zenith"}, **extra}


class SurveyConversionTest(unittest.TestCase):
    def test_respondent_rows_map_codes_and_skip_missing_answers(self):
        rows = survey.convert(EXAMPLE / "responses.csv", example_spec())
        first = next(r for r in rows if r["id"] == "R0001")
        self.assertIn("- Age: 25-34", first["state"])
        self.assertEqual(first["gold"]["brand_pref"], "everbrew")
        self.assertEqual(first["gold"]["recommend"], 0)  # Code 6 falls in the first NPS bucket.
        self.assertIsInstance(first["gold"]["aware_everbrew"], bool)
        with open(EXAMPLE / "responses.csv", newline="") as handle:
            refused = {r["respondent_id"] for r in csv.DictReader(handle) if r["Q5"] == "99"}
        self.assertTrue(refused)
        for row in rows:
            self.assertEqual(set(row["questions"]), set(row["gold"]))
            if row["id"] in refused:
                self.assertNotIn("satisfaction", row["questions"])

    def test_splits_are_deterministic_and_cover_requested_names(self):
        first = survey.convert(EXAMPLE / "responses.csv", example_spec())
        second = survey.convert(EXAMPLE / "responses.csv", example_spec())
        self.assertEqual([r["split"] for r in first], [r["split"] for r in second])
        self.assertEqual({r["split"] for r in first}, {"train", "dev", "calibration", "test"})

    def test_aggregate_rows_are_weighted_distributions(self):
        rows = survey.convert(EXAMPLE / "responses.csv", example_spec(), aggregate=True)
        groups = {}
        for row in rows:
            (qid, probs), = row["gold_probs"].items()
            self.assertAlmostEqual(math.fsum(probs.values()), 1.0)
            self.assertEqual(groups.setdefault(row["group"], row["split"]), row["split"])
        self.assertTrue(all(len(r["questions"]) == 1 and "gold" not in r for r in rows))

    def test_answer_columns_cannot_leak_into_the_state(self):
        spec = example_spec()
        spec["state"]["fields"].append({"column": "Q3", "label": "Brand"})
        with self.assertRaisesRegex(ValueError, "leak"):
            survey.convert(EXAMPLE / "responses.csv", spec)

    def test_numeric_option_keys_are_rejected(self):
        spec = example_spec()
        spec["questions"][0]["options"]["1"]["key"] = "1"
        with self.assertRaisesRegex(ValueError, "model input"):
            survey.validate_spec(spec)

    def test_score_level_limit(self):
        spec = example_spec()
        spec["questions"][3]["levels"] = [{"codes": [str(i)], "description": f"Rating {i} of 10"} for i in range(11)]
        with self.assertRaisesRegex(ValueError, "2-10 levels"):
            survey.validate_spec(spec)


class TrainingDataTest(unittest.TestCase):
    def test_paths_match_inference_encoding(self):
        row = survey.convert(EXAMPLE / "responses.csv", example_spec())[0]
        with tempfile.TemporaryDirectory() as directory:
            examples = trainer.load_examples(write_rows(directory, [row]), CharacterTokenizer(), 100000)
        request = {"states": [{"id": row["id"], "state": row["state"], "questions": row["questions"]}]}
        reference = prepare_examples(request, CharacterTokenizer(), 100000)
        self.assertEqual([[list(p) for p in ex["leaf_tokens"]] for ex in examples],
                         [ex["leaf_tokens"] for ex in reference])
        for ex in examples:
            self.assertEqual(sum(ex["target"]), 1.0)

    def test_soft_targets_and_shorthands(self):
        q = {"type": "score", "instructions": "x", "criteria": ["low", "mid", "high"]}
        self.assertEqual(trainer.target_vector(q, ["0", "1", "2"], None, [0.2, 0.3, 0.5]), [0.2, 0.3, 0.5])
        b = {"type": "boolean", "instructions": "x"}
        self.assertEqual(trainer.target_vector(b, ["false", "true"], None, 0.25), [0.75, 0.25])
        self.assertEqual(trainer.target_vector(b, ["false", "true"], True, None), [0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            trainer.target_vector(q, ["0", "1", "2"], None, [0.2, 0.3, 0.6])

    def test_rejects_missing_targets_and_split_crossing(self):
        with tempfile.TemporaryDirectory() as directory:
            unlabeled = choice_row()
            unlabeled["gold"] = {}
            with self.assertRaisesRegex(ValueError, "without targets"):
                trainer.load_examples(write_rows(directory, [unlabeled]), CharacterTokenizer(), 100000)
            crossing = [choice_row("a", group="person"), choice_row("b", "dev", group="person")]
            with self.assertRaisesRegex(ValueError, "appears in both"):
                trainer.load_examples(write_rows(directory, crossing), CharacterTokenizer(), 100000)

    def test_summary_and_prior(self):
        rows = [choice_row("a", weight=3.0), choice_row("b", gold={"brand": "acme"})]
        with tempfile.TemporaryDirectory() as directory:
            examples = trainer.load_examples(write_rows(directory, rows), CharacterTokenizer(), 100000)
        prior = trainer.fit_prior(examples)
        self.assertEqual(trainer.prior_probs(prior, examples[0]), [0.25, 0.75])
        exact = trainer.summarize(examples, [ex["target"] for ex in examples])
        self.assertAlmostEqual(exact["ce"], 0.0)
        self.assertEqual(exact["accuracy"], 1.0)
        self.assertAlmostEqual(trainer.summarize(examples, [[0.25, 0.75]] * 2)["ce"],
                               -(3 * math.log(0.75) + math.log(0.25)) / 4)

    def test_sampler_epochs_are_permutations(self):
        sampler = trainer.EpochSampler(list(range(10)), 4, seed=3)
        seen = sum((sampler.batch_at(step) for step in range(5)), [])
        self.assertEqual(sorted(seen[:10]), list(range(10)))
        self.assertEqual(sorted(seen[10:20]), list(range(10)))
        self.assertEqual(seen, sum((trainer.EpochSampler(list(range(10)), 4, 3).batch_at(s) for s in range(5)), []))


def build_tiny_bundle(root, seed=0):
    import string
    import torch
    from safetensors.torch import save_file
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import AutoModel, PreTrainedTokenizerFast, Qwen3Config
    from predict_toy_decisions import load_decision_model_class
    root = Path(root)
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{c: i + 3 for i, c in enumerate(sorted(set(string.printable)))}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>", pad_token="<pad>",
                            unk_token="<unk>").save_pretrained(root / "tokenizer")
    config = Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=4096)
    torch.manual_seed(seed)
    model = load_decision_model_class()(AutoModel.from_config(config), "attention")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, root / "best.safetensors")
    config.save_pretrained(root / "backbone_config")
    (root / "config.json").write_text(json.dumps({"model": "tiny-test", "set_head": "attention", "max_length": 4096}))
    return root


@unittest.skipUnless(HAS_TORCH, "torch/transformers not installed; training checks require project dependencies")
class TinyTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.bundle = build_tiny_bundle(root / "bundle")
        rows = survey.convert(EXAMPLE / "responses.csv", example_spec())
        cls.data = root / "rows.jsonl"
        cls.data.write_text("".join(json.dumps(r) + "\n" for r in rows[:80]), encoding="utf-8")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_train(self, name, *extra):
        out = self.root / name
        trainer.main(["train", "--checkpoint-dir", str(self.bundle), "--data", str(self.data), "--output-dir", str(out),
                      "--device", "cpu", "--precision", "fp32", "--batch-questions", "6", "--eval-every", "4",
                      "--log-every", "4", "--max-microbatch-tokens", "8192", *extra])
        return out

    def test_output_is_a_loadable_bundle_matching_the_report(self):
        import torch
        out = self.run_train("trained", "--steps", "8", "--backbone-lr", "1e-3", "--head-lr", "1e-3")
        report = json.loads((out / "report.json").read_text())
        model, tokenizer, config, _, _ = trainer.load_bundle_model(out, torch.device("cpu"))
        self.assertEqual(config["post_training"]["best_step"], report["best_step"])
        dev = [ex for ex in trainer.load_examples(self.data, tokenizer, 4096) if ex["split"] == "dev"]
        args = trainer.build_parser().parse_args(["train", "--checkpoint-dir", "x", "--data", "x", "--output-dir", "x",
                                                  "--precision", "fp32", "--max-microbatch-tokens", "8192"])
        reloaded = trainer.evaluate(model, dev, args, torch.device("cpu"), tokenizer.pad_token_id)
        self.assertAlmostEqual(reloaded["ce"], report["final"]["dev"]["model"]["ce"], places=6)

    def test_no_dev_improvement_ships_the_starting_weights(self):
        out = self.run_train("frozen", "--steps", "4", "--backbone-lr", "0", "--head-lr", "0")
        self.assertEqual(json.loads((out / "report.json").read_text())["best_step"], 0)
        self.assertEqual((out / "best.safetensors").read_bytes(), (self.bundle / "best.safetensors").read_bytes())

    def test_microbatching_does_not_change_the_gradient(self):
        import torch
        model, tokenizer, _, _, _ = trainer.load_bundle_model(self.bundle, torch.device("cpu"))
        batch = [ex for ex in trainer.load_examples(self.data, tokenizer, 4096) if ex["split"] == "train"][:6]
        for i, ex in enumerate(batch):
            ex["weight"] = 1.0 + i
        total = sum(ex["weight"] for ex in batch)

        def gradient(groups):
            model.zero_grad(set_to_none=True)
            for group in groups:
                logits, _ = model(group, tokenizer.pad_token_id)
                weights = torch.tensor([ex["weight"] for ex in group])
                ((trainer.question_losses(logits, group, 0.5) * weights).sum() / total).backward()
            return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

        whole = gradient([batch])
        split = gradient([[ex] for ex in batch])
        self.assertLess(float((whole - split).abs().max()), 1e-5)

    def test_rotary_guard_rejects_a_misread_theta(self):
        from transformers import AutoConfig, AutoModel
        body = AutoModel.from_config(AutoConfig.from_pretrained(self.bundle / "backbone_config"))
        trainer.check_rotary(body, self.bundle / "backbone_config")
        claimed = self.root / "claimed_config"
        claimed.mkdir(exist_ok=True)
        raw = json.loads((self.bundle / "backbone_config" / "config.json").read_text())
        raw.pop("rope_theta", None)
        raw["rope_parameters"] = {"rope_theta": 1000000, "rope_type": "default"}
        (claimed / "config.json").write_text(json.dumps(raw))
        with self.assertRaisesRegex(RuntimeError, "transformers>=5.17"):
            trainer.check_rotary(body, claimed)  # body was built with theta 10000

    def test_init_bundle_from_local_backbone(self):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        source = self.root / "hf_source"
        body = AutoModel.from_config(AutoConfig.from_pretrained(self.bundle / "backbone_config"))
        body.save_pretrained(source)
        AutoTokenizer.from_pretrained(self.bundle / "tokenizer").save_pretrained(source)
        out = self.root / "fresh"
        trainer.main(["init-bundle", "--model", str(source), "--output-dir", str(out), "--max-length", "4096"])
        model, _, config, _, _ = trainer.load_bundle_model(out, torch.device("cpu"))
        self.assertEqual(config["set_head"], "attention")
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
