"""NanoJev: states and typed questions in, complete probability distributions out.

The implementation lives in flat modules shared with the research scripts in `scripts/`, which
install alongside this package. This namespace is the stable public entry point. Importing it does
not import torch; `DecisionModel` loads torch and transformers on first access.
"""
from importlib.metadata import PackageNotFoundError, version as _version

from predict_toy_decisions import (
    DecisionPredictor, answer_from_probabilities, predict, prepare_examples, validate_request,
)

try:
    __version__ = _version("nanojev")
except PackageNotFoundError:  # Source checkout without installation.
    __version__ = "0+unknown"

__all__ = ["DecisionModel", "DecisionPredictor", "__version__", "answer_from_probabilities", "predict",
           "prepare_examples", "validate_request"]


def __getattr__(name):
    if name == "DecisionModel":
        # Same loader DecisionPredictor uses, so checkpoints and classes stay interchangeable.
        from predict_toy_decisions import load_decision_model_class
        globals()[name] = model_class = load_decision_model_class()
        return model_class
    raise AttributeError(f"module 'nanojev' has no attribute {name!r}")
