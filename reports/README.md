# Data preparation verification

These results describe dataset integrity, not trained-model performance.

| Check | Result |
| --- | --- |
| Train original/Mario integrity | 3,000 / 3,000 passed |
| Validation original/Mario integrity | 500 / 500 passed |
| Original training final answers against themselves | 3,000 / 3,000 matched |
| Mario training final answers against original | 3,000 / 3,000 matched |
| Mario validation final answers against original | 500 / 500 matched |
| Official test reference format validation | 1,319 / 1,319 valid |
| Unit and downloaded-data tests | 13 passed |

Integrity checks include exact recovery of the original rationale, unchanged questions, identical calculator annotations, normalized final answers, and chat-message consistency. Tests also check stored SHA-256 hashes, reconstruction from raw source records, exact split disjointness, malformed outputs, missing predictions, duplicate IDs, and deliberately corrupted answers/rationales.

The transformation adds rule-based Mario interjections around unchanged math text. A few examples were spot-checked during preparation; the planned 50-train/50-validation human style review has not been completed. Style quality and learning remain unverified. Some original GSM8K wording is awkward and is deliberately retained rather than silently edited.

No model has been trained or sampled. No model accuracy, similarity metric, persona judge score, or hypothesis result is available yet. The official test reference self-check only validates parsing, not model performance or independent correctness of the source solutions.

Each JSON summary has a corresponding `.details.jsonl` with per-example results. Reproduction commands are in the root README.
