# AIsafety1

Assignment 1: Post-training Qwen2.5-3B-Instruct for persona + STEM capability (SFT, RLAIF, RLVR).

## Current scope

**The initial Mario dataset is rejected and must not be used for training.** It added generic catchphrases rather than rewriting the explanations in character. [Review the context-aware replacement preview](MARIO_STYLE_PREVIEW.md) and [rewrite specification](prompts/mario_rewrite.md). Full regeneration is pending; the files and commands below document the original, rejected baseline only.

**Stopped at the user's request:** 475 Astra rewrites are saved (325 train, 150 validation). See [the checkpoint and Sonnet handoff](data/checkpoints/astra-paused/RESUME.md). All workers are interrupted; no Claude API generation has started.

See [the research plan](RESEARCH_PLAN.md). Prepared configuration: GSM8K, Mario, 3,000 train / 500 validation, seed 42. Training and model evaluation have **not** been run. This repository currently contains the data preparation and checking portion, not a trained model.

Python 3.11+; preparation and checking use only the standard library. Run commands from this repository's root.

```powershell
python mario_data.py --legacy-template-baseline
python -m unittest discover -s tests -v
python check_answers.py audit --mode legacy --original data/original/train.jsonl --modified data/mario/train.jsonl --report reports/train_integrity.json
python check_answers.py audit --mode legacy --original data/original/validation.jsonl --modified data/mario/validation.jsonl --report reports/validation_integrity.json
```

The downloader uses a pinned official OpenAI GSM8K revision. `data/raw` retains original files and the upstream MIT license. `data/manifest.json` records source URLs, hashes, source indices, and generated file hashes. Both selected splits come from official training data. `data/original/test.jsonl` contains all 1,319 official test examples, reserved for final evaluation. Regeneration overwrites generated data files; retain a separate copy before manually editing them.

Each prepared JSONL record contains `id`, unchanged `question`, `answer`, and `messages` (system/user/assistant). Original and Mario variants have identical IDs and questions. Mario answers use deterministic varied interjections, not model paraphrasing. Original rationale lines, calculator annotations, and final `####` answers are retained. Do not concatenate both variants for persona training: train on `data/mario/train.jsonl`, with assistant-only loss, and use the original variant for control training and verification. Human persona/readability review remains necessary.

## Check dataset answers

The audit verifies complete one-to-one pairing, unchanged questions and reasoning, calculator annotations, chat consistency, and normalized final answers. A failed audit returns a nonzero exit code. To numerically score original or rewritten dataset answers independently:

```powershell
python check_answers.py score --references data/original/train.jsonl --predictions data/original/train.jsonl --field answer --report reports/original_train_answers.json
python check_answers.py score --references data/original/train.jsonl --predictions data/mario/train.jsonl --field answer --report reports/mario_train_answers.json
```

These are reference consistency checks, **not model accuracy** and not proof of persona learning. The checker does not independently solve word problems or validate all intermediate arithmetic.

## Context-aware rewrite batches

Saved replacement answers were individually authored by `gpt-6-astra` subagents with medium reasoning in `data/rewrites_astra/`. `assemble_rewrites.py` checks IDs, exact final lines, numerical answers, and ordered calculator annotations, then pairs answers with unchanged original questions and chat format. It performs no persona generation and makes no claim to check semantic fidelity or character quality. Those require separate review. The earlier Luna trials are not included in the saved Astra checkpoint.

```powershell
python assemble_rewrites.py --split train --allow-partial
python assemble_rewrites.py --split train
python assemble_rewrites.py --split validation
```

Partial review files go to `data/mario_v2/train.preview.jsonl`, never to canonical training files. Without `--allow-partial`, assembly refuses missing rewrites. The complete v2 files must pass voice and semantic review before replacing the rejected dataset. Use `check_answers.py audit --mode rewrite` for genuine paraphrases; exact recovery of the old prose is deliberately not required.

## Check actual model outputs

Store completions as JSONL records with exact source IDs and an `output` string. Include only generated assistant text, not the prompt or gold answer. Example schema (illustrative ID and answer, not an actual prediction):

```json
{"id": "gsm8k/train/123", "output": "My reasoning...\n#### 42"}
```

Use the same reference set for base and SFT outputs:

```powershell
python check_answers.py score --references data/original/validation.jsonl --predictions outputs/base_validation.jsonl --report reports/base_validation.json
python check_answers.py score --references data/original/validation.jsonl --predictions outputs/sft_validation.jsonl --report reports/sft_validation.json
```

Scoring uses exact rational-number comparison, accepting properly grouped thousands separators, signed decimals, and fractions. It requires one terminal `#### number` marker; incidental numbers, multiple markers, trailing units, expressions, and malformed values are rejected. Missing predictions count as wrong. Duplicate/unknown IDs and invalid gold answers stop the run. Reports include accuracy, format rate, missing count, and per-example details. Low model accuracy is a valid evaluation result and does not make the scoring command fail.

## Attribution

GSM8K: Cobbe et al. (2021), [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168), [official source](https://github.com/openai/grade-school-math). See `data/raw/LICENSE` for the dataset's upstream license. The Mario-style additions are an unofficial educational experiment, not endorsed by Nintendo.
