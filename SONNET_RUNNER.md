# Cached Sonnet rewrite runner

`sonnet_rewrite.py` uses the Claude Messages API with `claude-sonnet-5`, adaptive thinking, and explicit low effort. It needs Python3.11+ and no third-party packages. Run from the repository root. Credentials are read from `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`; optionally supply `--env-file` for a local dotenv file. No credential is stored in the dataset, configuration, or logs.

```powershell
python sonnet_rewrite.py --env-file "C:/path/to/.env" --limit 6 --workers 2
python sonnet_rewrite.py --env-file "C:/path/to/.env" --workers 4
python sonnet_rewrite.py --export-only
```

## Caching and recovery

The fixed system prefix contains the rewrite specification and four approved training-only few-shot examples. An explicit `cache_control` breakpoint sits at the end of this prefix, before the varying question, reference answer, and retry feedback. The pilot measured2,901 cacheable prefix tokens. Model, thinking, effort, examples, and prefix ordering stay fixed throughout the run.

The first real request completes before concurrent requests begin, warming the cache without a throwaway API call. The default five-minute TTL refreshes on reads; continuously flowing requests therefore avoid the more expensive one-hour cache writes. The variable per-example suffix is deliberately not cached because it is normally used only once. Per-response usage records track fresh input, cache writes, cache reads, and output tokens. This follows [Anthropic's prompt-caching guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) and [effort configuration](https://platform.claude.com/docs/en/build-with-claude/effort).

An OS-released lock prevents duplicate simultaneous runners. Every accepted answer is atomically saved under `data/rewrites_sonnet/records/`, with source/config hashes, model identity, request ID, and usage. Restarting skips both the475 saved Astra answers and already validated Sonnet records. Changed prompts or source records are rejected rather than silently mixed. A snapshot of the exact shared prefix is in `data/rewrites_sonnet/config.json`.

HTTP429 and transient server/transport failures have bounded retries, respecting numeric `Retry-After` values. Failed-format or changed-answer completions are logged and regenerated, not repaired by copying the gold answer into the completion. Authentication/model errors stop the run. An uncertain network failure can still incur a charge before retry; reported cost is an estimate from received usage, not a billing guarantee.

## Validation and provenance

Every saved answer must preserve the exact final line and ordered calculator annotations and match the original numeric answer. These checks do not prove that the source reasoning is correct or that the persona is natural; sampled semantic/style review remains necessary. The original source caveats are retained under `data/rewrites_astra/caveats_*.jsonl`.

The runner exports ID-ordered combined shards to `data/rewrites_combined/`, preserving per-record Astra/Sonnet provenance. Partial exports are safe for inspection but not for training. Once both splits are complete and reviewed:

```powershell
python assemble_rewrites.py --shards data/rewrites_combined --publish
python -m unittest discover -s tests -v
```

Publication refuses incomplete splits and updates canonical data, hashes, and numerical audit reports. The official test set remains untouched. API attempts, failed candidates, cache statistics, and estimated cost are retained separately from training answers.
