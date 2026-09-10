"""Validate individually authored rewrite shards and assemble a complete paired dataset.

This does not generate prose or certify semantic fidelity/persona quality.
Partial batches produce a review file, never a partial canonical train split.
"""

import argparse
import hashlib
import json
from pathlib import Path

from check_answers import audit, index_rows
from mario_data import read_jsonl, write_jsonl


def assemble(references, rewrites, allow_partial=False):
    originals = index_rows(references)
    rewritten = index_rows(rewrites)
    unknown = rewritten.keys() - originals.keys()
    missing = originals.keys() - rewritten.keys()
    if unknown:
        raise ValueError(f"{len(unknown)} unknown rewrite IDs")
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} missing rewrites; refusing an incomplete dataset")
    selected = [row for row in references if row["id"] in rewritten]
    candidates = []
    for row in selected:
        answer = rewritten[row["id"]].get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Invalid answer for {row['id']}")
        candidates.append({**row, "answer": answer, "messages": [
            *row["messages"][:-1], {"role": "assistant", "content": answer}]})
    summary, details = audit(selected, candidates, mode="rewrite")
    if summary["failed"]:
        failed = [row["id"] for row in details if not row["passed"]]
        raise ValueError(f"Rewrite validation failed: {failed}")
    return candidates, {**summary, "expected_total": len(references),
                        "missing": len(missing), "complete": not missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--shards", default="data/rewrites")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    paths = sorted(Path(args.shards).glob(f"{args.split}_*.jsonl"))
    try:
        rewrites = [row for path in paths for row in read_jsonl(path)]
        rows, report = assemble(read_jsonl(f"data/original/{args.split}.jsonl"), rewrites, args.allow_partial)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    root = Path("data/mario_v2")
    root.mkdir(parents=True, exist_ok=True)
    suffix = ".preview" if args.allow_partial else ""
    output = root / f"{args.split}{suffix}.jsonl"
    write_jsonl(output, rows)
    report["shards"] = {str(path).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    report["generation"] = "individually authored by gpt-5.6-luna subagent; no rule-based prose generator"
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in report.items() if key != "shards"}, indent=2))


if __name__ == "__main__":
    main()
