#!/usr/bin/env python3
"""Convert qupa-datatypes surveys and responses into NanoJev decision rows.

A training spec (JSON, validated by `TrainingSpec`) lists sources. Each pairs a qupa `Survey` with
its `SurveyResponse` records, given as JSON/JSONL or as a raw CSV plus a qupa response CSV map.
Every response is validated against its survey. Each answered closed question then becomes one or
more NanoJev primitives, built from the full question definition so that unselected candidates are
targets too:

  single, grid row   ordered scale with 2-10 substantive points -> score, analytical low to high;
                     any other domain -> choice over every value;
                     non-substantive points ("don't know") -> a separate boolean, not a scale level
  multi              one boolean per option; multi_grid: one per answered row and column
  ranking            sequential choices: rank 1 among all items, rank 2 among the rest, ...
  maxdiff            per task, best among the presented items, then worst among the remainder
  text, numeric,     context only
  text/numeric list

The state verbalizes the respondent's other answers through qupa's resolution layer. Context is
"preceding" (earlier survey elements only, so routing that depends on the target cannot leak),
"all_other", or an explicit "listed" set. Hidden variables and synthetic answers are excluded
unless the spec opts in.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from qupa_datatypes import (
    ScaleVerbalization, Survey, SurveyResponse, VariableType, import_responses_csv,
    iter_resolved_answer_components, scale_content_fingerprint, survey_fingerprint,
)

SPLITS = ("train", "dev", "calibration", "test", "ood")
Split = Literal["train", "dev", "calibration", "test", "ood"]
ROWS_SCHEMA = "nanojev-qupa-rows-v1"
CLOSED_KINDS = {"single", "multi", "grid", "multi_grid", "ranking", "maxdiff"}


class SourceSpec(BaseModel):
    """One survey revision and its responses. Paths are relative to the spec file."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", description="Namespaces respondents and row ids.")
    survey: str
    responses: str | None = Field(default=None, description="SurveyResponse JSON list, object or JSONL.")
    responses_csv: str | None = None
    response_map: str | None = Field(default=None, description="qupa ResponseCsvMap for responses_csv.")
    split: Split | None = Field(default=None, description="Assign the whole source to one split.")

    @model_validator(mode="after")
    def one_response_input(self):
        if (self.responses is None) == (self.responses_csv is None):
            raise ValueError(f"source {self.name!r}: give exactly one of responses or responses_csv")
        if (self.responses_csv is None) != (self.response_map is None):
            raise ValueError(f"source {self.name!r}: responses_csv requires response_map")
        return self


class TrainingSpec(BaseModel):
    """Which answers become targets, which become context, and how rows are split."""

    model_config = ConfigDict(extra="forbid")
    sources: list[SourceSpec] = Field(min_length=1)
    targets: Literal["all"] | list[str] = "all"
    exclude: list[str] = Field(default_factory=list, description="Question ids never used as targets.")
    context: Literal["preceding", "all_other", "listed"] = "preceding"
    context_questions: list[str] = Field(default_factory=list, description="Required for context 'listed'.")
    context_exclude: list[str] = Field(default_factory=list, description="Question ids never shown in a state.")
    context_exclude_kinds: list[str] = Field(default_factory=list, description=(
        "Question kinds never shown in a state, e.g. ['text', 'text_list'] to keep open verbatims out."))
    max_context_answers: int = Field(default=40, ge=0, description="Nearest answered questions kept per state.")
    max_state_chars: int = Field(default=4000, ge=200, description="Context budget; nearest questions first.")
    max_question_chars: int = Field(default=40000, ge=2000, description=(
        "Every candidate repeats the state, so a row's state budget is also capped at this divided by its "
        "largest candidate count. Keep it near 3x the trainer's --max-microbatch-tokens."))
    max_line_chars: int = Field(default=300, ge=40, description="Each verbalized context line is cut here.")
    include_hidden: bool = False
    include_synthetic: bool = False
    non_substantive: Literal["boolean", "skip"] = "boolean"
    splits: dict[Split, float] = Field(default_factory=lambda: {"train": 0.7, "dev": 0.1, "calibration": 0.1,
                                                                "test": 0.1})
    split_seed: str = "nanojev-survey-v1"
    group_key: str | None = Field(default=None, description="custom_meta key grouping respondents, e.g. a household.")
    weight_key: str | None = Field(default=None, description="custom_meta key holding the survey weight.")
    scale_verbalizations: str | None = Field(default=None, description="JSON/JSONL of qupa ScaleVerbalization.")
    verbalization_language: Literal["DE", "EN"] | None = None
    skip_invalid_responses: bool = False

    @model_validator(mode="after")
    def consistent(self):
        if self.context == "listed" and not self.context_questions:
            raise ValueError("context 'listed' requires context_questions")
        if not {"train", "dev"} <= set(self.splits) or any(v < 0 for v in self.splits.values()) or \
                abs(math.fsum(self.splits.values()) - 1) > 1e-9:
            raise ValueError("splits must give nonnegative fractions summing to 1, including train and dev")
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")
        if (self.scale_verbalizations is None) != (self.verbalization_language is None):
            raise ValueError("scale_verbalizations and verbalization_language go together")
        return self


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json_records(path):
    """JSONL, a JSON list, or one JSON object. (A {"responses": [...]} wrapper would be ambiguous:
    SurveyResponse itself accepts `responses` as an alias for `answers`.)"""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def load_source(source, base):
    survey_path = base / source.survey
    survey = Survey.model_validate_json(survey_path.read_text(encoding="utf-8"))
    receipt = {"name": source.name, "survey": str(survey_path), "survey_sha256": sha256_file(survey_path),
               "survey_fingerprint": survey_fingerprint(survey=survey), "split": source.split}
    if source.responses is not None:
        path = base / source.responses
        responses = [SurveyResponse.model_validate(record) for record in read_json_records(path)]
        receipt.update(responses=str(path), responses_sha256=sha256_file(path))
    else:
        csv_path, map_path = base / source.responses_csv, base / source.response_map
        result = import_responses_csv(survey, csv_path.read_bytes(), json.loads(map_path.read_text(encoding="utf-8")),
                                      source_file=str(csv_path))
        responses = result.responses
        receipt.update(responses_csv=str(csv_path), responses_csv_sha256=sha256_file(csv_path),
                       response_map_sha256=sha256_file(map_path),
                       csv_diagnostics=Counter(d.code for d in result.diagnostics))
    receipt["responses_read"] = len(responses)
    return survey, responses, receipt


# ---------------------------------------------------------------------------
# Text: every candidate and context answer must read correctly on its own.

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def describe(item):
    """Participant-facing text, falling back to the identifier for empty authored text."""
    return clean(item.text) or item.label


def bare_number(text):
    return re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", clean(text)) is not None


class ScaleText:
    """Self-contained wording for scale points; NanoJev judges each Score level on its own.

    A stored qupa ScaleVerbalization label wins. Otherwise, on an ordered scale whose points include
    bare numbers ("2", "3", ...), every point carries the labelled endpoints: "4 [1 = Poor ... 7 = Good]".
    """

    def __init__(self, verbalizations=(), language=None):
        self.labels = {v.scale_fingerprint: v.labels for v in verbalizations if v.language == language}
        self.used = Counter()
        self._cache = {}

    def _scale_info(self, domain):
        key = id(domain)
        if key not in self._cache:
            fingerprint = scale_content_fingerprint(scale=domain)
            points = domain.analytical_points() if domain.direction is not None else []
            numeric = any(bare_number(point.text) for point in points)
            span = f" [{describe(points[0])} … {describe(points[-1])}]" if numeric and len(points) > 1 else ""
            self._cache[key] = (self.labels.get(fingerprint, {}), span)
        return self._cache[key]

    def point(self, value, domain, with_span=True):
        if domain.kind != "scale":
            return describe(value)
        labels, span = self._scale_info(domain)
        if value.label in labels:
            self.used["verbalization"] += 1
            return clean(labels[value.label])
        return describe(value) + (span if with_span and value.substantive else "")

    def span(self, domain):
        """The endpoint span once, for context headers whose answers omit it."""
        return self._scale_info(domain)[1] if domain is not None and domain.kind == "scale" else ""


def ordered_points(domain):
    """Analytical score levels when the domain is an ordered scale NanoJev can express, else None."""
    if domain.kind == "scale" and domain.direction is not None:
        points = domain.analytical_points()
        if 2 <= len(points) <= 10:
            return points
    return None


def scope_suffix(answer):
    return "".join(f"@{it.loop_id}.{it.loop_item_id}" for it in answer.loop_iterations)


def loop_items(survey, answer):
    scope = {it.loop_id: it.loop_item_id for it in answer.loop_iterations}
    items = []
    for loop in survey.loops_for(survey.get(answer.question_id)):
        items += [item for item in loop.items if item.loop_item_id == scope.get(loop.loop_id)]
    return items


# ---------------------------------------------------------------------------
# Targets: (nanojev question id, question, gold, meta) built from the full definition.

class TargetBuilder:
    def __init__(self, spec, scale_text, skipped):
        self.spec, self.scale_text, self.skipped = spec, scale_text, skipped

    def header(self, question, items, row=None):
        lines = [f"Survey question: {clean(question.title)}"]
        notes = []
        for note in (question.hint, question.comment):
            if clean(note) and clean(note) not in notes:
                notes.append(clean(note))
        lines += [f"Note: {note}" for note in notes]
        lines += [f"Asked about: {describe(item)}" for item in items]
        if row is not None:
            lines.append(f"Item: {describe(row)}")
        return lines

    @staticmethod
    def target(qid, typ, lines, criteria, gold, meta):
        question = {"type": typ, "instructions": "\n".join(lines)}
        if criteria is not None:
            question["criteria"] = criteria
        return qid, question, gold, meta

    def domain(self, qid, question, domain, selected_id, lines, meta):
        values = domain.values
        selected = next(value for value in values if value.label == selected_id)
        points = ordered_points(domain)
        if points is not None:
            others = [value for value in values if not value.substantive]
            if others and self.spec.non_substantive == "boolean":
                listed = "; ".join(describe(value) for value in others)
                yield self.target(f"{qid}.non_substantive", "boolean", lines + [
                    f"Does this respondent choose one of these answers instead of a scale point: {listed}?"],
                    None, not selected.substantive, {**meta, "role": "non_substantive"})
            if selected.substantive:
                levels = [self.scale_text.point(point, domain) for point in points]
                index = [point.label for point in points].index(selected.label)
                yield self.target(qid, "score", lines + ["Which point of the scale does this respondent choose?"],
                                  levels, index, {**meta, "role": "score"})
            else:
                self.skipped["non_substantive_answer_without_score"] += 1
            return
        if not 2 <= len(values) <= 255:
            self.skipped[f"choice_with_{'one' if len(values) < 2 else 'over_255'}_values"] += 1
            return
        criteria = {value.label: self.scale_text.point(value, domain) for value in values}
        yield self.target(qid, "choice", lines + ["Which answer does this respondent choose?"],
                          criteria, selected.label, {**meta, "role": "choice"})

    def build(self, survey, question, answer):
        items = loop_items(survey, answer)
        qid = question.question_id + scope_suffix(answer)
        meta = {"question_id": question.question_id, "kind": question.kind,
                "loop_iterations": [it.model_dump(mode="json") for it in answer.loop_iterations]}
        kind = question.kind
        if kind == "single":
            yield from self.domain(qid, question, question.response_domain, answer.selected,
                                   self.header(question, items), meta)
        elif kind == "grid":
            for row in question.rows:
                if row.label in answer.selections:
                    yield from self.domain(f"{qid}.{row.label}", question, question.response_domain,
                                           answer.selections[row.label], self.header(question, items, row),
                                           {**meta, "row": row.label})
        elif kind == "multi":
            chosen = set(answer.selected)
            for value in question.response_domain.values:
                yield self.target(f"{qid}.{value.label}", "boolean", self.header(question, items) + [
                    f"Does this respondent select this answer: {self.scale_text.point(value, question.response_domain)}?"],
                    None, value.label in chosen, {**meta, "role": "option", "option": value.label})
        elif kind == "multi_grid":
            for row in question.rows:
                if row.label not in answer.selections:
                    continue
                chosen = set(answer.selections[row.label])
                for value in question.response_domain.values:
                    yield self.target(f"{qid}.{row.label}.{value.label}", "boolean",
                                      self.header(question, items, row) + [
                        f"Does this respondent select this answer: {self.scale_text.point(value, question.response_domain)}?"],
                        None, value.label in chosen, {**meta, "role": "option", "row": row.label, "option": value.label})
        elif kind == "ranking":
            remaining, ranked = list(question.items), []
            for rank, item_id in enumerate(answer.ranked, 1):
                if len(remaining) < 2:
                    break
                lead = f"Which item does this respondent rank in position {rank}?"
                if ranked:
                    lead += " Already ranked: " + "; ".join(f"{i}. {describe(item)}" for i, item in enumerate(ranked, 1)) + "."
                yield self.target(f"{qid}.rank_{rank}", "choice", self.header(question, items) + [lead],
                                  {item.label: describe(item) for item in remaining}, item_id,
                                  {**meta, "role": "rank_step", "rank": rank})
                chosen = next(item for item in remaining if item.label == item_id)
                remaining.remove(chosen)
                ranked.append(chosen)
        elif kind == "maxdiff":
            best_pole, worst_pole = question.poles[0], question.poles[1]
            for task, selection in enumerate(answer.selections, 1):
                shown = [item for item in question.items if selection.presented is None or item.label in selection.presented]
                best = next(item for item in shown if item.label == selection.best)
                yield self.target(f"{qid}.task_{task}_best", "choice", self.header(question, items) + [
                    f"Task {task}: which item does this respondent choose as {describe(best_pole)}?"],
                    {item.label: describe(item) for item in shown}, selection.best,
                    {**meta, "role": "maxdiff_best", "task": task})
                rest = [item for item in shown if item.label != selection.best]
                if len(rest) >= 2:
                    yield self.target(f"{qid}.task_{task}_worst", "choice", self.header(question, items) + [
                        f"Task {task}: after choosing {describe(best)} as {describe(best_pole)}, which remaining "
                        f"item does this respondent choose as {describe(worst_pole)}?"],
                        {item.label: describe(item) for item in rest}, selection.worst,
                        {**meta, "role": "maxdiff_worst", "task": task})


# ---------------------------------------------------------------------------
# Context: the respondent's other answers, one verbalized entry per question and loop scope.

def cut(line, limit):
    return line if len(line) <= limit else line[:limit - 1] + "…"


def context_entries(survey, components, scale_text, max_line_chars):
    """Entries {question_id, position, hidden, lines, chars}, one per question and loop scope.

    The question title appears once; grid and list rows follow as indented lines, and a scale's
    endpoint span sits on the title line instead of on every answer.
    """
    positions = {element.question_id: index for index, element in enumerate(survey.elements)}
    grouped = {}
    for component in components:
        question = component.question
        domain = getattr(question, "response_domain", None)
        key = (question.question_id, scope_suffix(component.source_answer))
        if key not in grouped:
            prefix = "".join(f"[{describe(it.item)}] " for it in component.context.loop_iterations if it.item)
            grouped[key] = {"question_id": question.question_id, "kind": question.kind,
                            "position": positions[question.question_id],
                            "hidden": question.variable_type == VariableType.HIDDEN,
                            "header": f"- {prefix}{clean(question.title)}", "span": "", "rows": {}}
        entry = grouped[key]
        row, column = getattr(component, "row", None), getattr(component, "column", None)
        row_key = None if row is None else describe(row) + (f" / {describe(column)}" if column is not None else "")
        kind = component.component_kind
        if kind == "selection":
            if domain is not None and component.selected.substantive:
                entry["span"] = scale_text.span(domain)
            piece = (scale_text.point(component.selected, domain, with_span=False) if domain is not None
                     else describe(component.selected))
        elif kind == "ranking":
            piece = f"{component.rank}. {describe(component.option)}"
        elif kind == "maxdiff_pole":
            piece = f"task {component.task_index} {describe(component.pole)}: {describe(component.item)}"
        else:
            piece = f'"{clean(component.value)}"' if isinstance(component.value, str) else f"{component.value:g}"
        if getattr(component, "other_text", None):
            piece += f" ({clean(component.other_text)})"
        entry["rows"].setdefault(row_key, []).append(piece)
    entries = []
    for entry in grouped.values():
        header, rows = entry["header"] + entry["span"], entry["rows"]
        lines = [cut(f"{header}: {'; '.join(rows.pop(None))}" if None in rows else header, max_line_chars)]
        lines += [cut(f"    {row}: {'; '.join(pieces)}", max_line_chars) for row, pieces in rows.items()]
        entries.append({key: entry[key] for key in ("question_id", "kind", "position", "hidden")} |
                       {"lines": lines, "chars": sum(len(line) + 1 for line in lines)})
    return entries


def select_context(entries, spec, target_question, target_position, budget):
    """Whole entries, nearest to the target first, within the answer-count and character budgets."""
    if spec.context == "preceding":
        pool = [e for e in entries if e["position"] < target_position]
    elif spec.context == "all_other":
        pool = [e for e in entries if e["question_id"] != target_question]
    else:
        listed = set(spec.context_questions)
        pool = [e for e in entries if e["question_id"] in listed and e["question_id"] != target_question]
    excluded, excluded_kinds = set(spec.context_exclude), set(spec.context_exclude_kinds)
    pool = [e for e in pool if (spec.include_hidden or not e["hidden"]) and e["question_id"] not in excluded
            and e["kind"] not in excluded_kinds]
    if spec.context != "listed":  # Listed context keeps survey order; the others prefer proximity.
        pool = sorted(pool, key=lambda e: abs(e["position"] - target_position))
    chosen, used = [], 0
    for entry in pool:
        if len(chosen) == spec.max_context_answers:
            break
        if used + entry["chars"] <= budget:
            chosen.append(entry)
            used += entry["chars"]
    dropped = len(pool) - len(chosen)
    return sorted(chosen, key=lambda e: e["position"]), dropped  # sorted() is stable: loop scopes keep order.


def state_header(survey):
    lines = [f"Survey: {clean(survey.title)}"] if clean(survey.title) else []
    if clean(survey.description):
        lines.append(f"About the survey: {clean(survey.description)}")
    return lines


def render_state(survey, context):
    lines = state_header(survey)
    lines.append("Answers this respondent gave:" if context else "Answers this respondent gave: none recorded.")
    lines += [line for entry in context for line in entry["lines"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def split_for(group, spec):
    position = int(hashlib.sha256(f"{spec.split_seed}:{group}".encode()).hexdigest()[:15], 16) / 16 ** 15
    cumulative = 0.0
    for split, fraction in spec.splits.items():
        cumulative += fraction
        if position < cumulative:
            return split
    return list(spec.splits)[-1]


def meta_value(response, key, what):
    if key not in response.custom_meta:
        raise ValueError(f"respondent {response.respondent_id!r}: custom_meta lacks {what} key {key!r}")
    return response.custom_meta[key]


def is_target(question, spec):
    if question.kind not in CLOSED_KINDS or question.question_id in spec.exclude:
        return False
    if spec.targets != "all" and question.question_id not in spec.targets:
        return False
    if spec.context == "listed" and question.question_id in spec.context_questions:
        return False  # Its answer is part of every state in this mode.
    return question.variable_type == VariableType.VISIBLE or (
        spec.include_hidden and question.variable_type == VariableType.HIDDEN)


def build_rows(spec, base):
    """All rows for a spec, plus a report of what was used, skipped and why."""
    verbalizations = []
    if spec.scale_verbalizations:
        verbalizations = [ScaleVerbalization.model_validate(r) for r in read_json_records(base / spec.scale_verbalizations)]
    scale_text = ScaleText(verbalizations, spec.verbalization_language)
    skipped, roles, rows, receipts = Counter(), Counter(), [], []
    invalid = []
    for source in spec.sources:
        survey, responses, receipt = load_source(source, base)
        known = set(survey.element_ids)
        missing = (set(spec.context_questions) | set(spec.targets if spec.targets != "all" else [])) - known
        if missing and len(spec.sources) == 1:
            raise ValueError(f"spec names questions absent from survey {source.name!r}: {sorted(missing)}")
        builder = TargetBuilder(spec, scale_text, skipped)
        positions = {element.question_id: index for index, element in enumerate(survey.elements)}
        header_chars = len(render_state(survey, [{"lines": []}]))
        used = 0
        for index, response in enumerate(responses):
            answers = [a for a in response.answers if spec.include_synthetic or not a.is_synthetic]
            skipped["synthetic_answers"] += len(response.answers) - len(answers)
            response = SurveyResponse(respondent_id=response.respondent_id, answers=answers,
                                      custom_meta=response.custom_meta)
            respondent = response.respondent_id or f"#{index}"
            try:
                components = list(iter_resolved_answer_components(survey, response, require_complete=False))
            except ValueError as exc:
                if not spec.skip_invalid_responses:
                    raise ValueError(f"source {source.name!r}, respondent {respondent!r}: {exc}") from None
                invalid.append({"source": source.name, "respondent": respondent, "error": str(exc)[:300]})
                continue
            used += 1
            group = f"{source.name}:{meta_value(response, spec.group_key, 'group') if spec.group_key else respondent}"
            split = source.split or split_for(group, spec)
            weight = float(meta_value(response, spec.weight_key, "weight")) if spec.weight_key else 1.0
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"respondent {respondent!r}: weight must be positive, got {weight}")
            entries = context_entries(survey, components, scale_text, spec.max_line_chars)
            pending = []
            for answer in response.answers:
                question = survey.get(answer.question_id)
                if not is_target(question, spec):
                    skipped[f"not_a_target:{question.kind}:{question.variable_type.value}"] += 1
                    continue
                targets = list(builder.build(survey, question, answer))
                if targets:
                    pending.append((question, answer, targets))
            # One row per target scope, except 'listed', whose single state serves every target.
            batches = [pending] if spec.context == "listed" else [[item] for item in pending]
            for batch in batches:
                if not batch:
                    continue
                question, answer, _ = batch[0]
                widest = max(len(q.get("criteria") or [0, 0]) for _, _, targets in batch for _, q, _, _ in targets)
                budget = min(spec.max_state_chars, spec.max_question_chars // widest) - header_chars
                context, dropped = select_context(entries, spec, question.question_id,
                                                  positions[question.question_id], budget)
                skipped["context_answers_over_budget"] += dropped
                row_scope = "" if spec.context == "listed" else f":{question.question_id}{scope_suffix(answer)}"
                row = {"id": f"{source.name}:{respondent}{row_scope}", "group": group, "split": split,
                       "weight": weight, "state": render_state(survey, context), "questions": {}, "gold": {},
                       "meta": {"schema": ROWS_SCHEMA, "source": source.name, "respondent_id": response.respondent_id,
                                "context_questions": [e["question_id"] for e in context], "targets": {}}}
                for _, _, targets in batch:
                    for qid, nanojev_question, gold, target_meta in targets:
                        row["questions"][qid] = nanojev_question
                        row["gold"][qid] = gold
                        row["meta"]["targets"][qid] = target_meta
                        roles[f"{target_meta['kind']}:{target_meta['role']}"] += 1
                rows.append(row)
        receipt["responses_used"] = used
        receipts.append(receipt)
    report = {"schema": ROWS_SCHEMA, "spec": spec.model_dump(mode="json"), "sources": receipts, "rows": len(rows),
              "questions": sum(len(r["questions"]) for r in rows),
              "rows_by_split": dict(sorted(Counter(r["split"] for r in rows).items())),
              "targets_by_role": dict(sorted(roles.items())), "skipped": dict(sorted(skipped.items())),
              "scale_verbalizations_used": scale_text.used["verbalization"], "invalid_responses": invalid}
    return rows, report


def load_spec(path):
    path = Path(path)
    return TrainingSpec.model_validate_json(path.read_text(encoding="utf-8")), path.resolve().parent


def rows_jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="training spec JSON (see TrainingSpec)")
    parser.add_argument("--output", required=True, help="rows JSONL for train_survey_decisions.py --data")
    args = parser.parse_args(argv)
    spec, base = load_spec(args.spec)
    rows, report = build_rows(spec, base)
    if not any(r["split"] == "train" for r in rows) or not any(r["split"] == "dev" for r in rows):
        raise SystemExit("Conversion produced no train or no dev rows; add responses or adjust splits")
    output = Path(args.output)
    report_path = output.with_name(output.stem + ".report.json")
    for path in (output, report_path):
        if path.exists():
            raise SystemExit(f"{path} exists; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rows_jsonl(rows), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report_path), "rows": report["rows"],
                      "questions": report["questions"], "rows_by_split": report["rows_by_split"],
                      "targets_by_role": report["targets_by_role"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
