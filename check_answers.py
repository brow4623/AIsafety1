"""Audit original/Mario pairs or score model outputs against GSM8K references."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import re

from mario_data import CLOSERS, OPENERS, TRANSITIONS, read_jsonl, write_jsonl

# Accept exact integers, finite decimals, and fractions; never evaluate code.
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+)?"
FINAL = re.compile(rf"####\s*({NUMBER})\s*\Z")


def extract_answer(text):
    """Require one unambiguous terminal answer marker, not the last incidental number."""
    if not isinstance(text, str) or text.count("####") != 1:
        return None
    match = FINAL.search(text)
    if not match:
        return None
    try:
        value = match[1].replace(",", "")
        if "/" in value:
            numerator, denominator = value.split("/")
            return Fraction(numerator) / Fraction(denominator)
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key or key in indexed:
            raise ValueError(f"Missing, invalid, or duplicate ID: {key!r}")
        indexed[key] = row
    if not indexed:
        raise ValueError("Empty dataset")
    return indexed


def restore_original(text):
    lines = text.split("\n")
    if len(lines) < 4 or lines[0] not in OPENERS or lines[-2] not in CLOSERS:
        return None
    restored = [lines[1]]
    for line in lines[2:-2]:
        prefix = next((prefix for prefix in TRANSITIONS if line.startswith(prefix)), None)
        if prefix is None:
            return None
        restored.append(line[len(prefix):])
    return "\n".join(restored + [lines[-1]])


def audit(original, modified):
    old, new = index_rows(original), index_rows(modified)
    if old.keys() != new.keys():
        raise ValueError("Original and modified ID sets differ")
    details = []
    for key, reference in old.items():
        candidate = new[key]
        gold, value = extract_answer(reference["answer"]), extract_answer(candidate["answer"])
        checks = {
            "question_unchanged": reference["question"] == candidate["question"],
            "answer_preserved": gold is not None and gold == value,
            "rationale_preserved": restore_original(candidate["answer"]) == reference["answer"],
            "calculations_preserved": re.findall(r"<<.*?>>", reference["answer"]) == re.findall(r"<<.*?>>", candidate["answer"]),
            "chat_consistent": all(row.get("messages") == [
                {"role": "system", "content": reference["messages"][0]["content"]},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}] for row in (reference, candidate)),
        }
        details.append({"id": key, **checks, "passed": all(checks.values())})
    passed = sum(row["passed"] for row in details)
    return {"kind": "dataset_integrity_not_model_accuracy", "total": len(details),
            "passed": passed, "failed": len(details) - passed,
            "answer_preservation_rate": sum(row["answer_preserved"] for row in details) / len(details)}, details


def score(references, predictions, field="output"):
    refs = index_rows(references)
    preds = index_rows(predictions) if predictions else {}
    unknown = preds.keys() - refs.keys()
    if unknown:
        raise ValueError(f"Predictions contain {len(unknown)} unknown IDs")
    details = []
    for key, row in refs.items():
        gold = extract_answer(row["answer"])
        if gold is None:
            raise ValueError(f"Invalid reference answer: {key}")
        if key in preds and not isinstance(preds[key].get(field), str):
            raise ValueError(f"Prediction {key} needs a string field {field!r}")
        value = extract_answer(preds[key][field]) if key in preds else None
        details.append({"id": key, "gold": str(gold),
                        "prediction": str(value) if value is not None else None,
                        "missing": key not in preds, "valid_format": value is not None,
                        "correct": value == gold})
    count = len(details)
    correct = sum(row["correct"] for row in details)
    valid = sum(row["valid_format"] for row in details)
    return {"kind": "numeric_exact_match", "total": count, "correct": correct,
            "accuracy": correct / count, "valid_format": valid, "format_rate": valid / count,
            "missing": sum(row["missing"] for row in details)}, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    integrity = sub.add_parser("audit")
    integrity.add_argument("--original", required=True)
    integrity.add_argument("--modified", required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--references", required=True)
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--field", default="output", help="Prediction text field; use answer for dataset self-checks")
    for command in (integrity, scoring):
        command.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        if args.command == "audit":
            summary, details = audit(read_jsonl(args.original), read_jsonl(args.modified))
        else:
            summary, details = score(read_jsonl(args.references), read_jsonl(args.predictions), args.field)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_jsonl(path.with_suffix(".details.jsonl"), details)
    print(json.dumps(summary, indent=2))
    return 1 if args.command == "audit" and summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
