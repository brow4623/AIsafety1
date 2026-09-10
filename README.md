# AIsafety1

Assignment 1: Post-training Qwen2.5-3B-Instruct for persona + STEM capability (SFT, RLAIF, RLVR).

## Current scope

The replacement dataset uses context-aware Mario narration, following the approved [style preview](MARIO_STYLE_PREVIEW.md) and [rewrite specification](prompts/mario_rewrite.md). The original phrase-insertion approach was rejected and retired. See [the cached Sonnet runner](SONNET_RUNNER.md) for generation and resume instructions, and `data/rewrites_sonnet/summary.json` for recorded progress, cache usage, and cost.

Generation combines 475 saved Astra-medium answers with Sonnet 5 low-effort API rewrites for the remaining 3,025 examples. Per-record `rewrite_provenance` distinguishes the two sources in the published data. The [Astra checkpoint](data/checkpoints/astra-paused/RESUME.md) is a historical snapshot, not the current completion status.

See [the research plan](RESEARCH_PLAN.md). Prepared configuration: GSM8K, Mario, 3,000 train / 500 validation, seed 42. Training and model evaluation have **not** been run. This repository currently contains the data preparation and checking portion, not a trained model.

Python 3.11+; preparation and checking use only the standard library. Run commands from this repository's root.

```powershell
python sonnet_rewrite.py --export-only
python assemble_rewrites.py --shards data/rewrites_combined --publish
python -m unittest discover -s tests -v
python check_answers.py audit --mode rewrite --original data/original/train.jsonl --modified data/mario/train.jsonl --report reports/train_integrity.json
python check_answers.py audit --mode rewrite --original data/original/validation.jsonl --modified data/mario/validation.jsonl --report reports/validation_integrity.json
```

The downloader uses a pinned official OpenAI GSM8K revision. `data/raw` retains original files and the upstream MIT license. `data/manifest.json` records source URLs, hashes, source indices, and generated file hashes. Both selected splits come from official training data. `data/original/test.jsonl` contains all 1,319 official test examples, reserved for final evaluation. Regeneration overwrites generated data files; retain a separate copy before manually editing them.

Each prepared JSONL record contains `id`, unchanged `question`, `answer`, and `messages` (system/user/assistant). Original and Mario variants have identical IDs and questions. The explanatory prose is genuinely rewritten; exact calculator annotations and final `####` lines are retained. Do not concatenate both variants for persona training: use `data/mario/train.jsonl` with assistant-only loss and the original variant for control training and verification. Source caveats and independent human persona/readability review still matter; automated checks do not prove semantic fidelity.

## Check dataset answers

The rewrite audit verifies complete one-to-one pairing, unchanged questions, exact calculator annotations and final lines, chat consistency, and normalized final answers. It deliberately does not require unchanged prose. A failed audit returns a nonzero exit code. To numerically score original or rewritten dataset answers independently:

```powershell
python check_answers.py score --references data/original/train.jsonl --predictions data/original/train.jsonl --field answer --report reports/original_train_answers.json
python check_answers.py score --references data/original/train.jsonl --predictions data/mario/train.jsonl --field answer --report reports/mario_train_answers.json
```

These are reference consistency checks, **not model accuracy** and not proof of persona learning. The checker does not independently solve word problems or validate all intermediate arithmetic.

## Context-aware rewrite batches

The earlier Luna trials are excluded. Astra source shards are in `data/rewrites_astra/`; Sonnet records, exact prompt configuration, usage, and rejected attempts are in `data/rewrites_sonnet/`. The combined, ID-ordered source shards are in `data/rewrites_combined/`. `assemble_rewrites.py` validates and pairs them but does not generate persona text or independently certify character quality.

```powershell
python assemble_rewrites.py --shards data/rewrites_combined --split train --allow-partial
python assemble_rewrites.py --shards data/rewrites_combined --publish
```

Partial review files go to `data/mario_v2/train.preview.jsonl`, never to canonical training files. Publication refuses missing rewrites. Use `check_answers.py audit --mode rewrite` for genuine paraphrases; exact recovery of the old prose is deliberately not required. `mario_data.py` reproduces only the rejected historical baseline and should not be run over the canonical dataset.

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
