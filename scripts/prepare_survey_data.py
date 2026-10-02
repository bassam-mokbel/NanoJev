#!/usr/bin/env python3
"""Convert a respondent-level survey CSV into NanoJev decision rows.

A JSON spec names the profile columns that form the text state and the survey questions that
become targets. Every question maps onto one primitive:

  single-select  -> choice   options keyed by meaningful names; the key itself is model input
  Likert / NPS   -> score    2-10 self-contained level descriptions; several raw codes may share a level
  yes/no, or one option of a multi-select -> boolean

Respondent mode writes one row per respondent with hard targets. Aggregate mode (--aggregate)
merges respondents whose rendered state is identical and writes one row per segment and question
with the weighted answer distribution as gold_probs, which trains share prediction directly.
Splits are assigned by respondent (or segment) hash, so no group crosses splits.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SPLITS = ("train", "dev", "calibration", "test", "ood")


def fail(message):
    raise ValueError(message)


def split_for(group, fractions, seed):
    position = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:15], 16) / 16 ** 15
    cumulative = 0.0
    for split, fraction in fractions.items():
        cumulative += fraction
        if position < cumulative:
            return split
    return list(fractions)[-1]


def validate_spec(spec):
    if not isinstance(spec, dict) or not {"id_column", "state", "questions"} <= set(spec):
        fail("spec needs id_column, state and questions")
    fractions = spec.get("splits", {"train": 0.7, "dev": 0.1, "calibration": 0.1, "test": 0.1})
    if set(fractions) - set(SPLITS) or not {"train", "dev"} <= set(fractions) or \
            any(not isinstance(v, (int, float)) or v < 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-9:
        fail(f"splits must assign nonnegative fractions summing to 1 over {SPLITS}, including train and dev")
    state_columns = {field["column"] for field in spec["state"].get("fields", [])}
    seen = set()
    for q in spec["questions"]:
        where = f"question {q.get('id')!r}"
        if not isinstance(q.get("id"), str) or not q["id"] or q["id"] in seen:
            fail(f"{where}: ids must be unique nonempty strings")
        seen.add(q["id"])
        if q.get("column") in state_columns:
            fail(f"{where}: column {q['column']!r} is also a state field; the answer would leak into the input")
        if q.get("type") not in {"boolean", "choice", "score"} or not str(q.get("instructions", "")).strip():
            fail(f"{where}: needs type boolean/choice/score and nonempty instructions")
        if q["type"] == "choice":
            keys = [o["key"] for o in q["options"].values()]
            if not 2 <= len(set(keys)) <= 255 or len(set(keys)) != len(keys):
                fail(f"{where}: choice needs 2-255 distinct option keys")
            if any(not k.strip() or k.strip().isdigit() for k in keys):
                fail(f"{where}: option keys are model input ('key: description'); use names like 'acme', not codes")
        elif q["type"] == "score":
            codes = [c for level in q["levels"] for c in level["codes"]]
            if not 2 <= len(q["levels"]) <= 10 or len(codes) != len(set(codes)):
                fail(f"{where}: score needs 2-10 levels with disjoint codes (bucket NPS 0-10 into <=10 levels)")
        elif set(q["true_codes"]) & set(q["false_codes"]):
            fail(f"{where}: true_codes and false_codes overlap")
    return fractions


def render_state(row, spec, missing):
    lines = [spec["state"]["intro"]] if spec["state"].get("intro") else []
    for field in spec["state"].get("fields", []):
        raw = row[field["column"]].strip()
        if raw in missing:
            continue
        values = field.get("values")
        if values is not None and raw not in values:
            fail(f"state column {field['column']!r}: code {raw!r} has no label")
        lines.append(f"- {field['label']}: {values[raw] if values is not None else raw}")
    return "\n".join(lines)


def question_payload(q):
    payload = {"type": q["type"], "instructions": q["instructions"]}
    if q["type"] == "choice":
        payload["criteria"] = {o["key"]: o["description"] for o in q["options"].values()}
    elif q["type"] == "score":
        payload["criteria"] = [level["description"] for level in q["levels"]]
    elif q.get("criteria"):
        payload["criteria"] = q["criteria"]
    return payload


def answer(q, raw):
    """Map a raw CSV code to the target in candidate space."""
    if q["type"] == "choice":
        if raw not in q["options"]:
            fail(f"question {q['id']!r}: code {raw!r} is not an option")
        return q["options"][raw]["key"]
    if q["type"] == "score":
        for index, level in enumerate(q["levels"]):
            if raw in level["codes"]:
                return index
        fail(f"question {q['id']!r}: code {raw!r} is not in any level")
    if raw in q["true_codes"]:
        return True
    if raw in q["false_codes"]:
        return False
    fail(f"question {q['id']!r}: code {raw!r} is neither true nor false")


def candidate_key(q, value):
    return {True: "true", False: "false"}.get(value, str(value)) if q["type"] != "choice" else value


def convert(csv_path, spec, aggregate=False):
    fractions = validate_spec(spec)
    missing = set(spec.get("missing_codes", [""]))
    seed = spec.get("split_seed", "survey-v1")
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        needed = {spec["id_column"], *(f["column"] for f in spec["state"].get("fields", [])),
                  *(q["column"] for q in spec["questions"])}
        if spec.get("weight_column"):
            needed.add(spec["weight_column"])
        if spec.get("group_column"):
            needed.add(spec["group_column"])
        absent = needed - set(reader.fieldnames or [])
        if absent:
            fail(f"CSV lacks columns {sorted(absent)}")
        respondents, ids = [], set()
        for line, row in enumerate(reader, 2):
            rid = row[spec["id_column"]].strip()
            if not rid or rid in ids:
                fail(f"CSV line {line}: respondent id must be unique and nonempty")
            ids.add(rid)
            weight = float(row[spec["weight_column"]]) if spec.get("weight_column") else 1.0
            if not math.isfinite(weight) or weight <= 0:
                fail(f"CSV line {line}: weight must be positive")
            answers = {q["id"]: answer(q, row[q["column"]].strip())
                       for q in spec["questions"] if row[q["column"]].strip() not in missing}
            respondents.append({"id": rid, "group": row[spec["group_column"]].strip() if spec.get("group_column") else rid,
                                "weight": weight, "state": render_state(row, spec, missing), "answers": answers})
    questions = {q["id"]: q for q in spec["questions"]}
    rows = []
    if not aggregate:
        for r in respondents:
            if r["answers"]:
                rows.append({"id": r["id"], "group": r["group"], "split": split_for(r["group"], fractions, seed),
                             "weight": r["weight"], "state": r["state"],
                             "questions": {qid: question_payload(questions[qid]) for qid in r["answers"]},
                             "gold": r["answers"]})
        return rows
    segments = {}
    for r in respondents:
        segment = segments.setdefault(r["state"], {})
        for qid, value in r["answers"].items():
            mass = segment.setdefault(qid, {})
            key = candidate_key(questions[qid], value)
            mass[key] = mass.get(key, 0.0) + r["weight"]
    for state, by_question in sorted(segments.items()):
        group = "segment:" + hashlib.sha256(state.encode()).hexdigest()[:16]
        split = split_for(group, fractions, seed)
        for qid in [q["id"] for q in spec["questions"] if q["id"] in by_question]:
            mass = by_question[qid]
            total = math.fsum(mass.values())
            rows.append({"id": f"{group}:{qid}", "group": group, "split": split, "weight": total, "state": state,
                         "questions": {qid: question_payload(questions[qid])},
                         "gold_probs": {qid: {k: v / total for k, v in sorted(mass.items())}}})
    return rows


def summary(rows):
    result = {}
    for row in rows:
        split = result.setdefault(row["split"], {"rows": 0, "questions": 0, "weight": 0.0})
        split["rows"] += 1
        split["questions"] += len(row["questions"])
        split["weight"] += row["weight"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="one row per respondent")
    parser.add_argument("--spec", required=True, help="JSON question/state specification")
    parser.add_argument("--output", required=True, help="JSONL for train_survey_decisions.py")
    parser.add_argument("--aggregate", action="store_true", help="segment-level answer distributions as soft targets")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    rows = convert(args.csv, spec, args.aggregate)
    if not any(r["split"] == "train" for r in rows) or not any(r["split"] == "dev" for r in rows):
        fail("Conversion produced no train or no dev rows; add data or adjust split fractions")
    output = Path(args.output)
    if output.exists():
        fail(f"{output} exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"output": str(output), "mode": "aggregate" if args.aggregate else "respondent",
                      "splits": summary(rows)}, indent=2))


if __name__ == "__main__":
    main()
