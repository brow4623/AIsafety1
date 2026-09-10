# Paused Astra rewrite checkpoint

Stopped at the user's request. All three Astra workers were interrupted; no further generation or Claude API call is running.

## Saved progress

- 325 training rewrites: source indices 0:160 and 1500:1665.
- 150 validation rewrites: source indices 0:150.
- 3,025 rewrites remain: 2,675 training and 350 validation.
- All 475 saved rewrites pass numeric final-answer equality, exact final-line preservation, ordered calculator-annotation preservation, and unchanged question/chat pairing checks. These checks do not establish independent semantic correctness or persona quality.

The `*.preview.jsonl` files contain complete paired records with questions and chat messages. The corresponding reports contain shard hashes. `checkpoint.json` lists exact completed indices, remaining indices, and remaining IDs. The individually authored source shards and source caveats are retained in `data/rewrites_astra/`.

**Do not trust shard filename endpoints as completion markers.** `train_0150_0200.jsonl` contains only indices150:160 (10 records); `train_1650_1700.jsonl` contains only1650:1665 (15 records). Resume by actual IDs, not filenames. Source indices refer to the prepared original split, not the numeric suffix of the GSM8K ID.

## Resume with Sonnet

The user plans to provide a Claude API key for Sonnet. No API integration or paid generation was started during this checkpoint. Reuse the approved `MARIO_STYLE_PREVIEW.md` direction and `prompts/mario_rewrite.md`; generate only missing IDs unless explicitly revising a saved answer. Keep model/provenance metadata separate for saved Astra answers and future Sonnet answers. Do not label a mixed-model dataset as entirely Astra or entirely Sonnet.

Preserve reference final lines and ordered annotations, check meaning and contextual character voice separately, and retain source caveats. Some source solutions contain questionable gold calculations or obvious prose typos. The workers recorded these in the caveat JSONL files; a passing numeric audit is not independent verification that the source answer is correct.

The first Astra batches received parent sampling review, not an exhaustive independent review of all475 answers. Training workers inadvertently saw the last three validation preview examples when reading the shared preview document; they were instructed not to use them as training-generation examples. For resumed training generation, supply only the first four training preview examples. The official GSM8K test set was not rewritten or used as generation examples.

The rejected old dataset remains in `data/mario/` and is still marked rejected. No partial replacement was published into canonical training files. `assemble_rewrites.py --allow-partial` reproduces the partial Astra preview; complete publication refuses missing IDs. Its single-provider metadata handling must be extended before publishing a mixed Astra/Sonnet dataset.
