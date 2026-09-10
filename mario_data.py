"""Reproduce the rejected insertion-based baseline for historical comparison only."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import urllib.request

REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
BASE_URL = f"https://raw.githubusercontent.com/openai/grade-school-math/{REVISION}"
SYSTEM = "Solve the math problem step by step. End with a line containing #### followed by only the numerical answer."
OPENERS = ("Let's-a go!", "It's-a me, Mario! Let's work this out!", "Here we go!", "Okey-dokey! Let's solve this!", "Let's get started, my friend!", "Mario's ready! Let's do the math!")
TRANSITIONS = ("Okey-dokey! ", "Here we go! ", "Let's keep going! ", "One step at a time! ", "Nice! ", "Let's-a go! ")
CLOSERS = ("Wahoo! We did it!", "Mamma mia, that's the answer!", "All right! We made it!", "Yahoo! That's our result!", "Way to go, my friend!", "Another problem solved! Wahoo!")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def mario_answer(answer, source_id, seed):
    """Insert style only; preserve each original rationale line and final answer."""
    rng = random.Random(f"mario-v1:{seed}:{source_id}")
    rationale, final = answer.rsplit("\n#### ", 1)
    lines = rationale.split("\n")
    styled = [rng.choice(OPENERS)]
    for index, line in enumerate(lines):
        styled.append((rng.choice(TRANSITIONS) if index else "") + line)
    return "\n".join(styled + [rng.choice(CLOSERS), "#### " + final])


def record(row, index, split, persona, seed):
    source_id = f"gsm8k/{split}/{index}"
    answer = mario_answer(row["answer"], source_id, seed) if persona else row["answer"]
    return {"id": source_id, "question": row["question"], "answer": answer,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": row["question"]},
                         {"role": "assistant", "content": answer}]}


def build(output, seed=42):
    output = Path(output)
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, remote in (("train.jsonl", "grade_school_math/data/train.jsonl"),
                         ("test.jsonl", "grade_school_math/data/test.jsonl"),
                         ("LICENSE", "LICENSE")):
        url = f"{BASE_URL}/{remote}"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        (raw / name).write_bytes(payload)
        sources[name] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest()}
    train, test = read_jsonl(raw / "train.jsonl"), read_jsonl(raw / "test.jsonl")
    if (len(train), len(test)) != (7473, 1319):
        raise ValueError("Unexpected official split sizes")
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    splits = {"train": indices[:3000], "validation": indices[3000:3500]}
    question_sets = [{train[i]["question"].strip() for i in selected} for selected in splits.values()]
    question_sets.append({row["question"].strip() for row in test})
    if any(question_sets[i] & question_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Exact question overlap across splits")
    for split, selected in splits.items():
        if len({train[i]["question"].strip() for i in selected}) != len(selected):
            raise ValueError(f"Duplicate questions within {split}")
        for variant in ("original", "mario"):
            write_jsonl(output / variant / f"{split}.jsonl",
                        [record(train[i], i, "train", variant == "mario", seed) for i in selected])
    write_jsonl(output / "original/test.jsonl", [record(row, i, "test", False, seed) for i, row in enumerate(test)])
    manifest = {"dataset": "GSM8K", "revision": REVISION, "seed": seed,
                "transformation": "mario-v1-rule-based-insertions", "sources": sources,
                "counts": {"train": 3000, "validation": 500, "test": len(test)},
                "source_train_indices": splits, "test_policy": "official test untouched; no training or tuning",
                "files": {str(p.relative_to(output)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                          for variant in ("original", "mario") for p in sorted((output / variant).glob("*.jsonl"))}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy-template-baseline", action="store_true",
                        help="Explicitly reproduce the rejected baseline, not the context-aware replacement")
    args = parser.parse_args()
    if not args.legacy_template_baseline:
        parser.error("The insertion-based Mario generator is retired. See prompts/mario_rewrite.md. "
                     "Use --legacy-template-baseline only to reproduce the rejected experiment.")
    print(json.dumps(build(args.output, args.seed)["counts"], indent=2))
