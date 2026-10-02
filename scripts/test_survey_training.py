#!/usr/bin/env python3
"""Offline checks for survey conversion and full fine-tuning.

Run: python3 -m unittest discover -s scripts -p test_survey_training.py -v
Conversion checks need qupa-datatypes, not torch. Training checks build a two-layer random Qwen3
with a character tokenizer on CPU; they download nothing and skip without torch/transformers.
"""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError
from qupa_datatypes import Survey, scale_content_fingerprint

import prepare_survey_data as survey
from predict_toy_decisions import prepare_examples
import train_survey_decisions as trainer

EXAMPLE = Path(__file__).resolve().parent.parent / "configs" / "survey_example"
HAS_TORCH = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors", "tokenizers"))
SURVEY = Survey.model_validate_json((EXAMPLE / "survey.json").read_text(encoding="utf-8"))
POSITION = {element.question_id: index for index, element in enumerate(SURVEY.elements)}


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]


def example_responses():
    return [json.loads(line) for line in (EXAMPLE / "responses.jsonl").read_text(encoding="utf-8").splitlines()]


def convert(directory, responses=None, **spec):
    """Rows and report for the example survey, optionally with edited responses or spec fields."""
    directory = Path(directory)
    path = directory / "responses.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in (responses or example_responses())), encoding="utf-8")
    source = {"name": "coffee", "survey": str(EXAMPLE / "survey.json"), "responses": str(path)}
    (directory / "spec.json").write_text(json.dumps({"sources": [source], "weight_key": "weight", **spec}))
    return survey.build_rows(*survey.load_spec(directory / "spec.json"))


def targets(rows, question_id):
    return [(row, qid) for row in rows for qid, meta in row["meta"]["targets"].items()
            if meta["question_id"] == question_id]


def write_rows(directory, rows):
    path = Path(directory) / "rows.jsonl"
    path.write_text(survey.rows_jsonl(rows), encoding="utf-8")
    return path


def choice_row(identifier="r1", split="train", **extra):
    return {"id": identifier, "split": split, "state": "Respondent profile: age 25-34.",
            "questions": {"brand": {"type": "choice", "instructions": "Which brand do they buy?",
                                    "criteria": {"acme": "Acme coffee", "zenith": "Zenith coffee"}}},
            "gold": {"brand": "zenith"}, **extra}


class QupaConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            cls.rows, cls.report = convert(directory)

    def test_each_answer_kind_maps_to_its_primitive(self):
        def types(question_id):
            return {row["questions"][qid]["type"] for row, qid in targets(self.rows, question_id)}
        self.assertEqual(types("q_brand"), {"choice"})
        self.assertEqual(types("q_satisfaction"), {"score", "boolean"})  # Scale plus the "don't know" hurdle.
        self.assertEqual(types("q_cups"), {"score"})
        self.assertEqual(types("q_attributes"), {"score"})
        self.assertEqual(types("q_aware"), {"boolean"})
        self.assertEqual(types("q_drivers"), {"choice"})
        self.assertEqual(types("q_recommend"), {"choice"})  # 11 points exceed the 10-level Score limit.
        for question_id in ("q_age", "q_why", "q_age_group", "intro"):
            self.assertEqual(targets(self.rows, question_id), [])

    def test_unselected_options_are_negative_targets(self):
        responses = {r["respondent_id"]: r for r in example_responses()}
        for row, qid in targets(self.rows, "q_aware"):
            chosen = next(a["selected"] for a in responses[row["meta"]["respondent_id"]]["answers"]
                          if a["question_id"] == "q_aware")
            self.assertEqual(row["gold"][qid], row["meta"]["targets"][qid]["option"] in chosen)
        self.assertEqual(len(targets(self.rows, "q_aware")), 5 * len(responses))

    def test_ranking_steps_remove_ranked_items(self):
        row = next(row for row, qid in targets(self.rows, "q_drivers") if qid.endswith("rank_3"))
        steps = [row["questions"][f"q_drivers.rank_{rank}"] for rank in (1, 2, 3)]
        self.assertEqual([len(step["criteria"]) for step in steps], [5, 4, 3])
        for rank in (1, 2):
            self.assertNotIn(row["gold"][f"q_drivers.rank_{rank}"], steps[2]["criteria"])
        self.assertIn("Already ranked: 1.", steps[1]["instructions"])

    def test_loop_scope_is_part_of_the_question(self):
        scoped = targets(self.rows, "q_recommend")
        self.assertTrue(scoped)
        for row, qid in scoped:
            item = row["meta"]["targets"][qid]["loop_iterations"][0]["loop_item_id"]
            self.assertIn({"brightbean": "BrightBean", "everbrew": "EverBrew"}[item], row["questions"][qid]["instructions"])

    def test_score_levels_read_on_their_own(self):
        row, qid = targets(self.rows, "q_attributes")[0]
        for level in row["questions"][qid]["criteria"]:
            self.assertTrue(level.endswith("[1 = Very poor … 7 = Excellent]"), level)

    def test_hidden_questions_never_reach_states(self):
        self.assertTrue(all("Age group" not in row["state"] for row in self.rows))
        self.assertTrue(all("q_age_group" not in row["meta"]["context_questions"] for row in self.rows))

    def test_preceding_context_excludes_the_target_and_later_questions(self):
        for row in self.rows:
            (meta, *_) = row["meta"]["targets"].values()
            self.assertTrue(all(POSITION[q] < POSITION[meta["question_id"]] for q in row["meta"]["context_questions"]))

    def test_splits_are_deterministic_and_grouped_by_respondent(self):
        with tempfile.TemporaryDirectory() as directory:
            again, _ = convert(directory)
        self.assertEqual(again, self.rows)
        by_group = {}
        for row in self.rows:
            self.assertEqual(by_group.setdefault(row["group"], row["split"]), row["split"])
        self.assertEqual(set(by_group.values()), {"train", "dev", "calibration", "test"})

    def test_rows_load_with_inference_encoding(self):
        row = self.rows[0]
        with tempfile.TemporaryDirectory() as directory:
            examples = trainer.load_examples(write_rows(directory, [row]), CharacterTokenizer(), 100000)
        request = {"states": [{"id": row["id"], "state": row["state"], "questions": row["questions"]}]}
        self.assertEqual([[list(p) for p in ex["leaf_tokens"]] for ex in examples],
                         [ex["leaf_tokens"] for ex in prepare_examples(request, CharacterTokenizer(), 100000)])
        self.assertTrue(all(ex["report_key"] == row["meta"]["targets"][ex["qid"]]["question_id"] for ex in examples))


class QupaSpecOptionsTest(unittest.TestCase):
    def test_synthetic_answers_are_excluded_unless_requested(self):
        responses = example_responses()
        for answer in responses[0]["answers"]:
            answer["is_synthetic"] = True
        rid = responses[0]["respondent_id"]
        with tempfile.TemporaryDirectory() as directory:
            rows, report = convert(directory, responses)
            self.assertFalse([r for r in rows if r["meta"]["respondent_id"] == rid])
            self.assertEqual(report["skipped"]["synthetic_answers"], len(responses[0]["answers"]))
            rows, _ = convert(directory, responses, include_synthetic=True)
            self.assertTrue([r for r in rows if r["meta"]["respondent_id"] == rid])

    def test_invalid_responses_fail_or_are_skipped(self):
        responses = example_responses()
        next(a for a in responses[1]["answers"] if a["question_id"] == "q_brand")["selected"] = "not_an_option"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, responses[1]["respondent_id"]):
                convert(directory, responses)
            rows, report = convert(directory, responses, skip_invalid_responses=True)
        self.assertEqual([r["respondent"] for r in report["invalid_responses"]], [responses[1]["respondent_id"]])

    def test_scale_verbalization_replaces_level_text(self):
        scale = SURVEY.get("q_satisfaction").response_domain
        labels = {f"p{v}": label for v, label in enumerate(["awful", "poor", "okay", "good", "great"], 1)}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "verbal.json").write_text(json.dumps([{
                "scale_fingerprint": scale_content_fingerprint(scale=scale), "language": "EN",
                "generator_version": "test", "labels": labels}]))
            rows, report = convert(directory, scale_verbalizations=str(Path(directory) / "verbal.json"),
                                   verbalization_language="EN")
        row, qid = next((r, q) for r, q in targets(rows, "q_satisfaction") if r["questions"][q]["type"] == "score")
        self.assertEqual(row["questions"][qid]["criteria"], list(labels.values()))
        self.assertGreater(report["scale_verbalizations_used"], 0)

    def test_listed_context_gives_one_row_per_respondent(self):
        listed = ["q_age", "q_region", "q_income"]
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = convert(directory, context="listed", context_questions=listed)
        self.assertEqual(len(rows), len(example_responses()))
        for row in rows:
            self.assertEqual(row["meta"]["context_questions"], listed)
            self.assertFalse({m["question_id"] for m in row["meta"]["targets"].values()} & set(listed))

    def test_context_exclusions_keep_verbatims_out(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = convert(directory, context_exclude_kinds=["text"], context_exclude=["q_age"])
        self.assertTrue(rows)
        for row in rows:
            self.assertFalse({"q_why", "q_age"} & set(row["meta"]["context_questions"]))
            self.assertNotIn("Why do you buy that brand?", row["state"])

    def test_state_budget_shrinks_with_candidate_count(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, report = convert(directory, max_question_chars=4000, context="all_other")
        for row in rows:
            widest = max(len(q.get("criteria") or [0, 0]) for q in row["questions"].values())
            self.assertLessEqual(len(row["state"]), max(4000 // widest, 120))
        self.assertGreater(report["skipped"]["context_answers_over_budget"], 0)

    def test_spec_validation(self):
        source = {"name": "s", "survey": "survey.json", "responses": "r.json"}
        with self.assertRaises(ValidationError):
            survey.TrainingSpec(sources=[source], context="listed")
        with self.assertRaises(ValidationError):
            survey.TrainingSpec(sources=[{**source, "responses_csv": "r.csv", "response_map": "m.json"}])
        with self.assertRaises(ValidationError):
            survey.TrainingSpec(sources=[source], splits={"train": 0.5, "test": 0.5})


class TrainingDataTest(unittest.TestCase):
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
        rows, _ = convert(root, example_responses()[:40])
        cls.spec = root / "spec.json"
        cls.data = write_rows(root, rows)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_train(self, name, *extra):
        out = self.root / name
        trainer.main(["train", "--checkpoint-dir", str(self.bundle), "--data", str(self.data), "--output-dir", str(out),
                      "--device", "cpu", "--precision", "fp32", "--batch-questions", "6", "--eval-every", "4",
                      "--log-every", "4", "--max-microbatch-tokens", "16384", *extra])
        return out

    def test_output_is_a_loadable_bundle_matching_the_report(self):
        import torch
        out = self.run_train("trained", "--steps", "8", "--backbone-lr", "1e-3", "--head-lr", "1e-3")
        report = json.loads((out / "report.json").read_text())
        model, tokenizer, config, _, _ = trainer.load_bundle_model(out, torch.device("cpu"))
        self.assertEqual(config["post_training"]["best_step"], report["best_step"])
        dev = [ex for ex in trainer.load_examples(self.data, tokenizer, 4096) if ex["split"] == "dev"]
        args = trainer.build_parser().parse_args(["train", "--checkpoint-dir", "x", "--data", "x", "--output-dir", "x",
                                                  "--precision", "fp32", "--max-microbatch-tokens", "16384"])
        reloaded = trainer.evaluate(model, dev, args, torch.device("cpu"), tokenizer.pad_token_id)
        self.assertAlmostEqual(reloaded["ce"], report["final"]["dev"]["model"]["ce"], places=6)

    def test_train_directly_from_a_qupa_spec(self):
        out = self.root / "from_spec"
        trainer.main(["train", "--checkpoint-dir", str(self.bundle), "--spec", str(self.spec), "--output-dir", str(out),
                      "--device", "cpu", "--precision", "fp32", "--steps", "2", "--batch-questions", "6",
                      "--eval-every", "2", "--log-every", "2", "--max-microbatch-tokens", "16384"])
        self.assertEqual((out / "rows.jsonl").read_text(encoding="utf-8"), self.data.read_text(encoding="utf-8"))
        report = json.loads((out / "report.json").read_text())
        self.assertLessEqual(set(report["test_by_question"]), set(POSITION))

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
