# Context-aware rewrite review

## Acceptance criteria

The original insertion-based dataset was rejected by the user. The replacement must rewrite the explanatory prose itself in Mario's voice, with reactions grounded in each question and characterful narration of the calculations. A neutral first-person paraphrase is also insufficient. Neither catchphrase counts nor numerical preservation establish persona quality.

Numeric final-answer equality, exact ordered calculator annotations, stable IDs, and unchanged questions/chat pairing are checked automatically. These do not prove semantic fidelity of the prose. The reviewing assistant and generation subagents must also check relationships, units, unannotated steps, and contextual voice. Human review by the project team remains advisable before training.

## Calibration

Generation uses `gpt-5.6-luna` subagents directly, without an external model API or a rule-based prose generator. The first 50-record attempt passed numeric checks but was rejected by the reviewing assistant for neutral textbook phrasing. A subsequent pass was still insufficiently characterful in several sampled rows. Further feedback produced the current calibration shard.

The reviewing assistant compared source questions, source solutions, and rewrites at train indices 0, 1, 4, 9, 17, 25, 32, 40, and 49. Corrections included distinguishing Tiffany's relay-leg duration from her finish order and retaining the fact that Allie is included in the party's 42 people. These are examples of errors that final-answer/annotation checks alone do not detect.

The latest samples use contextual dialogue about dinner, plumbing, charging devices, a relay team, party food, silk pillowcases, chocolate shopping, and stamp collections. Sensitive illness examples are intentionally less exuberant. This calibration review is not a claim that every record in the full replacement has been independently reviewed.

## Status

After Luna failed the style gate, the user approved Astra with medium reasoning. Three Astra workers produced 475 saved rewrites before the user requested stopping to switch to a Claude Sonnet API workflow. All workers were interrupted. The saved checkpoint contains 325 train and 150 validation rewrites, all passing numeric/final-line/annotation/pairing checks. No Sonnet API generation has begun. The old `data/mario` files remain rejected; no incomplete replacement has been published. See `data/checkpoints/astra-paused/RESUME.md` for exact resume information, caveats, and provenance.

The parent sampled Astra train indices 0, 4, 17, 25, 29, 32, 40, 49, 1504, 1516, 1533, and 1549, and validation indices 4, 16, 33, and 49. These samples were more context-aware than the rejected Luna outputs. This does not establish uniform quality across all saved examples. Source caveats and obvious unannotated prose/algebra clarifications are recorded separately by each worker under `data/rewrites_astra/caveats_*.jsonl`.
