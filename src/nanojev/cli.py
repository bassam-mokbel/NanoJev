"""The `nanojev` command.

Each command runs its module exactly as `python scripts/<module>.py` would, through runpy, so the
installed CLI and the research scripts share one implementation and cannot drift apart.
"""
import runpy
import sys
import types

# Command words -> (module, arguments prepended for that module's own parser, summary).
COMMANDS = {
    ("predict",): ("predict_toy_decisions", [], "batch prediction from a local checkpoint (JSON in, JSON out)"),
    ("serve",): ("serve_decisions", [], "HTTP inference at POST /api/evaluate (run from the repo or pass --web-root)"),
    ("survey", "prepare"): ("prepare_survey_data", [], "convert qupa surveys and responses (training spec) into rows"),
    ("survey", "train"): ("train_survey_decisions", ["train"], "fully fine-tune a bundle on rows or a training spec"),
    ("survey", "init-bundle"): ("train_survey_decisions", ["init-bundle"], "untuned backbone with fresh heads"),
}


def usage():
    from nanojev import __version__
    lines = [f"nanojev {__version__}", "", "usage: nanojev <command> [options]", "", "commands:"]
    lines += [f"  {' '.join(words):<20} {summary}" for words, (_, _, summary) in COMMANDS.items()]
    lines += ["", "Run `nanojev <command> --help` for that command's options."]
    return "\n".join(lines)


def run_as_script(module, argv):
    """Execute `module` as `python scripts/<module>.py` would: named __main__, with no module spec.

    argparse names its program from argv[0], or from __main__.__spec__ on Python 3.14+ when a spec
    exists, so a spec-free __main__ keeps `usage: nanojev <command>` under every launcher.
    """
    saved_argv, saved_main = sys.argv, sys.modules.get("__main__")
    sys.argv, sys.modules["__main__"] = list(argv), types.ModuleType("__main__")
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = saved_argv
        if saved_main is not None:
            sys.modules["__main__"] = saved_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(usage())
        return 0
    if argv[0] == "--version":
        from nanojev import __version__
        print(__version__)
        return 0
    for words, (module, lead, _) in COMMANDS.items():
        if tuple(argv[:len(words)]) == words:
            # Subcommand parsers append their own name to the parent's prog.
            prog = "nanojev " + " ".join(words[:-1] if lead else words)
            run_as_script(module, [prog, *lead, *argv[len(words):]])
            return 0
    print(f"nanojev: unknown command {' '.join(argv[:2])!r}\n\n{usage()}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
