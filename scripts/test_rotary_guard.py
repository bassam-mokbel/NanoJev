#!/usr/bin/env python3
"""Offline checks that inference never runs on misread RoPE frequencies.

Run: python3 -m unittest discover -s scripts -p test_rotary_guard.py -v
The released backbone_config stores its RoPE base only under the transformers 5 `rope_parameters`
field; transformers 4.x ignores it and builds theta=10000. These checks build a one-layer random
Qwen3 on CPU through the same build_backbone call DecisionPredictor uses; they download nothing and
skip without torch/transformers. The module import itself must keep working without torch.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import predict_toy_decisions as predictor

HAS_TORCH = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))
RELEASE_THETA = 1000000


def release_style_config(source, target):
    """Copy a saved config, keeping its RoPE base only where the release puts it."""
    raw = json.loads((Path(source) / "config.json").read_text())
    raw.pop("rope_theta", None)
    raw["rope_parameters"] = {"rope_theta": RELEASE_THETA, "rope_type": "default"}
    Path(target).mkdir(exist_ok=True)
    (Path(target) / "config.json").write_text(json.dumps(raw))
    return Path(target)


@unittest.skipUnless(HAS_TORCH, "torch/transformers not installed; rotary checks require project dependencies")
class RotaryGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3Config
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.honest = root / "honest"
        Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=16, max_position_embeddings=128).save_pretrained(cls.honest)
        cls.claimed = release_style_config(cls.honest, root / "claimed")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_rotary_guard_rejects_a_misread_theta(self):
        from transformers import AutoConfig
        predictor.build_backbone(AutoConfig.from_pretrained(self.honest), self.honest)
        # A config object holding the default theta 10000 is what transformers 4.x makes of the release file.
        with self.assertRaisesRegex(RuntimeError, "transformers>=5.17"):
            predictor.build_backbone(AutoConfig.from_pretrained(self.honest), self.claimed)

    def test_release_style_config_is_never_silently_misread(self):
        import torch
        from transformers import AutoConfig
        try:
            body = predictor.build_backbone(AutoConfig.from_pretrained(self.claimed), self.claimed)
        except RuntimeError as error:  # transformers 4.x: ignores rope_parameters.
            self.assertIn("transformers>=5.17", str(error))
        else:  # transformers 5: reads it, so the built frequencies must use the release base.
            expected = 1.0 / (RELEASE_THETA ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
            self.assertTrue(torch.allclose(body.rotary_emb.inv_freq.float(), expected, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
