#!/usr/bin/env python3
"""Packaging checks: the `nanojev` facade, CLI dispatch, and the installed module list.

Run: python3 -m unittest discover -s scripts -p test_package.py -v
Standard library only; works from a source checkout or an installed package.
"""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "configs" / "survey_example"
if importlib.util.find_spec("nanojev") is None:  # Source checkout without `pip install -e .`.
    sys.path.insert(0, str(ROOT / "src"))

import nanojev  # noqa: E402
from nanojev import cli  # noqa: E402



def sibling_imports(path, names):
    """Top-level and function-local imports of other scripts/ modules."""
    found = re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", Path(path).read_text(encoding="utf-8"), re.M)
    return set(found) & names


class FacadeTest(unittest.TestCase):
    def test_import_stays_light(self):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts")])}
        probe = "import sys, nanojev; print(sorted({'torch', 'transformers'} & set(sys.modules)))"
        result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "[]")

    def test_public_names(self):
        for name in nanojev.__all__:
            if name != "DecisionModel":  # Loads torch on access; covered by the training tests.
                self.assertTrue(hasattr(nanojev, name), name)
        with self.assertRaises(AttributeError):
            nanojev.not_a_name  # noqa: B018


class CliTest(unittest.TestCase):
    def test_usage_lists_every_command(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main([]), 0)
        for words in cli.COMMANDS:
            self.assertIn(" ".join(words), out.getvalue())

    def test_unknown_command_is_an_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["survey", "nope"]), 2)

    def test_subcommand_help_names_the_nanojev_command(self):
        # --help exits inside argument parsing, before any checkpoint or torch import.
        for words in cli.COMMANDS:
            out = io.StringIO()
            with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as stop:
                cli.main([*words, "--help"])
            self.assertEqual(stop.exception.code, 0)
            self.assertIn("usage: nanojev " + " ".join(words), out.getvalue())

    def test_survey_prepare_runs_the_script_and_restores_argv(self):
        saved = list(sys.argv)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(["survey", "prepare", "--spec", str(EXAMPLE / "training_spec.json"),
                                 "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2268)
            self.assertTrue((Path(directory) / "rows.report.json").is_file())
        self.assertEqual(sys.argv, saved)


class PyprojectTest(unittest.TestCase):
    def test_installed_modules_are_closed_under_sibling_imports(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        installed = set(project["tool"]["setuptools"]["py-modules"])
        scripts = {path.stem for path in (ROOT / "scripts").glob("*.py")}
        self.assertLessEqual(installed, scripts)
        for path in [*(ROOT / "scripts" / f"{name}.py" for name in installed), *(ROOT / "src" / "nanojev").glob("*.py")]:
            missing = sibling_imports(path, scripts) - installed
            self.assertFalse(missing, f"{path.name} imports modules pyproject.toml does not install: {sorted(missing)}")
        # predict_toy_decisions loads the model class from this file by path, not by import.
        self.assertIn("train_toy_decisions", installed)

    def test_cli_targets_are_installed(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        installed = set(project["tool"]["setuptools"]["py-modules"])
        self.assertLessEqual({module for module, _, _ in cli.COMMANDS.values()}, installed)
        self.assertEqual(project["project"]["scripts"]["nanojev"], "nanojev.cli:main")


if __name__ == "__main__":
    unittest.main()
