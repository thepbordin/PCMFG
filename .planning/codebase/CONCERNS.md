# Codebase Concerns

**Analysis Date:** 2026-04-23

## Tech Debt

### Deprecated Axis System Still Present
- Issue: The old 4-axis romance system (`AxisValues`, `AxesTimeSeries`, `AxisMapper`, `plot_axes`) is deprecated but still fully present in the codebase. The CLAUDE.md and architecture describe raw 9-emotion output, yet the deprecated classes remain imported, tested, and referenced in `AnalysisResult`.
- Files: `pcmfg/models/schemas.py` (lines 160-186), `pcmfg/phase3/axis_mapper.py`, `pcmfg/visualization/plotter.py` (lines 493-589), `pcmfg/cli.py` (lines 192-210), `tests/test_phase3.py` (entire file tests deprecated AxisMapper)
- Impact: Increases cognitive load for maintainers. New contributors may use the deprecated API. Tests exercise code that should be removed.
- Fix approach: Remove `AxisValues`, `AxesTimeSeries`, `AxisMapper`, `plot_axes()`, and the `axes` field from `AnalysisResult`. Remove `tests/test_phase3.py` (rename existing `tests/test_synthesizer.py` if needed). Update `cli.py` `_display_summary` to remove axis table.

### `EmotionExtractionError` Never Raised
- Issue: `EmotionExtractionError` is defined in `pcmfg/phase1/emotion_extractor.py` (line 419) but never raised anywhere. The `extract()` method catches errors and returns a default chunk instead.
- Files: `pcmfg/phase1/emotion_extractor.py` (line 419)
- Impact: Dead code. If callers want to handle extraction failures, they have no mechanism to do so.
- Fix approach: Either raise `EmotionExtractionError` when extraction fails (instead of silently returning defaults), or remove the class entirely.

### Duplicate `nonlocal` Race Condition in Parallel Processing
- Issue: Both `_extract_emotions` and `_extract_emotions_with_checkpoint` in `pcmfg/analyzer.py` use `nonlocal` variables (`skipped_count`, `completed_count`, `chunks_since_checkpoint`) inside closures passed to `ParallelProcessor`. While `ThreadPoolExecutor` processes these concurrently, the `nonlocal` increments are not thread-safe.
- Files: `pcmfg/analyzer.py` (lines 486, 311, 343-351, 391-396)
- Impact: In concurrent processing, the `skipped_count` and `chunks_since_checkpoint` counters may be inaccurate due to race conditions. Checkpoint saves may happen at wrong intervals.
- Fix approach: Use `threading.Lock` around nonlocal increments, or use `concurrent.futures.as_completed` with proper tracking.

### `AsyncParallelProcessor` Never Used
- Issue: `AsyncParallelProcessor` in `pcmfg/utils/parallel.py` (lines 106-180) is fully implemented but never imported or used anywhere in the codebase.
- Files: `pcmfg/utils/parallel.py` (lines 106-180)
- Impact: Dead code adding maintenance burden.
- Fix approach: Remove unless there are plans to migrate to async in the near future.

### `process_in_batches` and `parallel_process_chunks` Never Used
- Issue: Both helper functions in `pcmfg/utils/parallel.py` are standalone convenience functions not called from anywhere.
- Files: `pcmfg/utils/parallel.py` (lines 183-240)
- Impact: Dead code.
- Fix approach: Remove or integrate into the main pipeline.

### `_get_text_sample` Deprecated But Still Present
- Issue: `_get_text_sample` in `pcmfg/analyzer.py` (lines 447-459) is documented as deprecated ("use get_strategic_sample") but still exists and is tested.
- Files: `pcmfg/analyzer.py` (lines 447-459), `tests/test_analyzer.py` (lines 151-161)
- Impact: Confusing API surface with deprecated methods.
- Fix approach: Remove the method and its test.

## Known Bugs

### Typo in `extract_from_chunks` — `b_to_b` Instead of `b_to_a`
- Symptoms: When `include_both_directions=True`, the B→A feature values are copied from the A→B vector instead of the actual B→A vector.
- Files: `pcmfg/analysis/feature_extractor.py` (line 451)
- Trigger: Calling `FeatureExtractor.extract_from_chunks()` with `include_both_directions=True` produces incorrect feature matrices where both directions have the same A→B values.
- Workaround: Use the `extract()` method with a full `AnalysisResult` instead.
- Fix approach: Change `b_to_b` to `b_to_a` on line 451:
  ```python
  # Bug:
  features[i] = np.concatenate([a_to_b, b_to_b])
  # Fix:
  features[i] = np.concatenate([a_to_b, b_to_a])
  ```

### `WorldBuilderError` Silently Swallowed in Analyzer
- Symptoms: When the World Builder LLM call fails, the analyzer silently falls back to placeholder data ("Character A", "Character B") instead of propagating the error. This produces garbage results downstream without the user knowing.
- Files: `pcmfg/analyzer.py` (lines 429-445)
- Trigger: LLM API failure during Phase 1.
- Workaround: None — the error is swallowed.
- Fix approach: Either re-raise the error (with option to retry), or at minimum log a prominent WARNING. The default fallback produces meaningless emotion extraction since there are no real aliases.

### Checkpoint Hash Collision Risk
- Symptoms: `compute_text_hash` truncates SHA-256 to 8 characters (line 66 of `pcmfg/checkpoint.py`), meaning only ~4 billion unique hashes. With large text corpora, collisions become probable.
- Files: `pcmfg/checkpoint.py` (line 66)
- Trigger: Processing many different texts with the same first 8 hex chars of their SHA-256 hash.
- Workaround: Low probability for typical usage (< 100 novels).
- Fix approach: Use full SHA-256 hash or at minimum 16 characters.

## Security Considerations

### API Keys Read from Environment Without Validation
- Risk: API keys are loaded from env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) without any validation of format or length. A typo in the env var name silently leads to a `ValueError` at runtime.
- Files: `pcmfg/llm/openai_client.py` (line 41), `pcmfg/llm/anthropic_client.py` (line 39)
- Current mitigation: `ValueError` raised if key is None.
- Recommendations: Add format validation (e.g., OpenAI keys start with `sk-`, Anthropic keys start with `sk-ant-`). Provide a clearer error message indicating which env var is missing.

### `.pcmfg_checkpoints/` Not in `.gitignore`
- Risk: Checkpoint files contain full text analysis data including novel content, emotion scores, and potentially sensitive narrative text. They could be accidentally committed.
- Files: `.gitignore` (missing entry), `pcmfg/checkpoint.py` (creates `.pcmfg_checkpoints/` directory)
- Current mitigation: None.
- Recommendations: Add `.pcmfg_checkpoints/` to `.gitignore`.

### Input Text Logged Nowhere but Also Not Sanitized for Prompt Injection
- Risk: User-provided text is passed directly to LLM prompts. A malicious text file could attempt prompt injection to extract system prompts or alter LLM behavior.
- Files: `pcmfg/phase1/world_builder.py` (lines 82-89), `pcmfg/phase1/emotion_extractor.py` (line 245)
- Current mitigation: Structured JSON output requirement provides some protection.
- Recommendations: Consider sanitizing input text before embedding in prompts, or at minimum adding prompt boundaries (e.g., XML tags) to separate instructions from user text.

### LLM Response Includes Raw Error Text in `parse_json_response`
- Risk: `parse_json_response` in `pcmfg/llm/base.py` (line 115) includes up to 500 characters of the failed response in the error message. This could include sensitive LLM output or user text.
- Files: `pcmfg/llm/base.py` (line 115)
- Current mitigation: Truncated to 500 chars.
- Recommendations: Consider logging only a hash or truncated excerpt. Ensure error messages don't leak user content to logs.

## Performance Bottlenecks

### Linear Scan for Character Name Matching
- Problem: `should_process_chunk` and `normalize_character_name` use case-insensitive substring matching against every alias for every chunk. For novels with many characters and many aliases, this is O(aliases × chunk_length) per chunk.
- Files: `pcmfg/phase1/emotion_extractor.py` (lines 76-122, 27-73)
- Cause: Simple string-based matching without indexing or compilation.
- Improvement path: Pre-compile a single regex pattern from all aliases at `EmotionExtractor` init time. Use `re.compile` with `|` alternation for all names.

### Sequential Checkpoint Saving During Phase 2
- Problem: In `_extract_emotions_with_checkpoint`, checkpoint is saved synchronously inside the results loop (line 392-396 of `analyzer.py`). For large analyses with many chunks, this blocks the parallel processing pipeline.
- Files: `pcmfg/analyzer.py` (lines 391-396)
- Cause: Checkpoint writes are I/O-bound and done inline.
- Improvement path: Save checkpoints in a background thread or use a separate checkpoint-writing thread with a queue.

### Full Novel Text Held in Memory
- Problem: `analyze()` and `analyze_with_checkpoint()` load the entire novel text into memory. For very long novels (100K+ words), this is manageable but for batch processing of many novels, memory usage grows linearly.
- Files: `pcmfg/analyzer.py` (lines 125, 171)
- Cause: Text is passed as a single string throughout the pipeline.
- Improvement path: Use memory-mapped files or streaming for very large texts. Low priority for typical romance novels (50K-200K words).

### Token Estimation Is Inaccurate
- Problem: `estimate_tokens` uses a simple `words × 1.3` heuristic. This can be off by 30-50% for non-English text, text with many numbers/symbols, or text with long compound words.
- Files: `pcmfg/utils/text_processing.py` (lines 13-25)
- Cause: No actual tokenizer used.
- Improvement path: Use `tiktoken` for OpenAI models or `anthropic` tokenizer for Claude models. Add as optional dependency.

## Fragile Areas

### LLM Response Parsing
- Files: `pcmfg/llm/base.py` (lines 81-115), `pcmfg/phase1/world_builder.py` (lines 103-150), `pcmfg/phase1/emotion_extractor.py` (lines 260-376)
- Why fragile: The pipeline depends on LLMs returning valid JSON matching exact schemas. LLMs can:
  - Return markdown-wrapped JSON (`\`\`\`json ... \`\`\``)
  - Omit required fields
  - Return wrong types (string instead of int for scores)
  - Add extra fields
  - Return truncated responses for long texts
- Safe modification: The `_parse_response` methods in both `world_builder.py` and `emotion_extractor.py` already have defensive fallbacks. Any changes should preserve this defensive pattern.
- Test coverage: Good — `test_phase1.py` tests missing fields, wrong types, too many/few characters. However, no tests for truncated responses or malformed JSON.

### Forward Fill Imputation Logic
- Files: `pcmfg/phase3/synthesizer.py` (lines 45-157)
- Why fragile: The imputation uses case-insensitive substring matching (lines 87-99) to match character names in directed emotions. If the LLM returns slightly different character names (e.g., "Elizabeth Bennet" vs "Elizabeth"), the matching may fail silently and the direction won't be imputed.
- Safe modification: The `normalize_character_name` function in `emotion_extractor.py` is used during extraction, but `synthesizer.py` does its own inline matching. Consolidate character name normalization into a single utility.
- Test coverage: Good — `test_synthesizer.py` covers forward fill carry-forward and missing direction cases.

### Chunk Filtering Optimization
- Files: `pcmfg/phase1/emotion_extractor.py` (lines 76-122)
- Why fragile: The `should_process_chunk` function has a partial matching heuristic (lines 103-121) that checks if key words from names appear in chunk text. Short common words (length > 2) like "the" are excluded, but words like "Bob" or "Ann" could match unrelated text (e.g., "bobbing", "annual").
- Safe modification: Add word boundary checking to partial matching. Consider requiring at least 4 characters for partial matches.
- Test coverage: Good — `test_synthesizer.py` has tests for character present/absent cases and case insensitivity.

### Configuration Merging
- Files: `pcmfg/config.py` (lines 126-155)
- Why fragile: `merge_cli_overrides` uses string splitting on `__` to handle nested keys. Type conversion is not performed — all values remain strings. This means `merge_cli_overrides(cfg, llm__max_concurrency="10")` sets a string "10" instead of int 10, which will fail Pydantic validation later.
- Safe modification: Add type coercion based on the Config model's field types, or validate after merging.
- Test coverage: No tests found for `merge_cli_overrides`.

## Scaling Limits

### API Rate Limits
- Current capacity: Hardcoded retry with 3 attempts and exponential backoff (4-10s) in both `OpenAIClient` and `AnthropicLLMClient`.
- Limit: For a 200-chapter novel producing ~200 chunks, with 5 concurrent workers, that's 200+ API calls. Rate limits (RPM/TPM) will be hit for most providers.
- Scaling path: The `max_concurrency` config (default 5) helps but doesn't implement adaptive rate limiting. Add token bucket or sliding window rate limiter.

### Checkpoint File Size
- Current capacity: Each checkpoint stores all processed chunks as JSON. For a 200-chunk novel, each chunk with full emotion data is ~1KB, so checkpoints are ~200KB.
- Limit: For very long novels (1000+ chapters), checkpoints could reach several MB.
- Scaling path: Consider storing chunks in separate files or using a more compact format (msgpack).

### No Batch Processing Support
- Current capacity: The analyzer processes one novel at a time.
- Limit: Cannot efficiently process a corpus of novels for the `TrajectoryClusterer` feature.
- Scaling path: Add a batch processing mode that reuses the same LLM client and manages multiple analyses.

## Dependencies at Risk

### Anthropic SDK Model Version
- Risk: `anthropic>=0.18.0` in `pyproject.toml` but the `AnthropicLLMClient` uses `anthropic.Client` (old API). The Anthropic SDK has undergone major API changes. The `response.content` iteration pattern (lines 82-86) may break with newer versions.
- Files: `pyproject.toml` (line 12), `pcmfg/llm/anthropic_client.py` (lines 72-91)
- Impact: SDK update could break all Anthropic-based analysis.
- Migration plan: Pin to a specific SDK version. Update to use `response.content[0].text` directly (the current iteration pattern is overly defensive).

### `scikit-learn` as Required Dependency
- Risk: `scikit-learn>=1.3.0` is listed as a core dependency in `pyproject.toml` but is only used for clustering (`pcmfg/analysis/`). The clustering code already has `SKLEARN_AVAILABLE` guards, suggesting it was intended to be optional.
- Files: `pyproject.toml` (line 19), `pcmfg/analysis/clusterer.py` (lines 28-39)
- Impact: All users must install scikit-learn (and its dependencies like scipy, joblib) even if they only want basic emotion extraction.
- Migration plan: Move `scikit-learn` to optional dependencies. Already partially handled with `SKLEARN_AVAILABLE` guards.

## Missing Critical Features

### No Retry on Individual Chunk Failures
- Problem: When `ParallelProcessor` encounters an error on a chunk, it creates a default (all-1s) chunk. There's no mechanism to retry just the failed chunk.
- Blocks: Produces inaccurate time-series data when transient API errors occur.
- Fix approach: Add per-chunk retry logic in the `process_chunk` closure, similar to the LLM client retry.

### No Validation of Emotion Score Distributions
- Problem: The `EmotionNormalizer` in `pcmfg/phase2/normalizer.py` validates individual chunks but is never called in the main pipeline. The `analyze()` method goes directly from extraction to synthesis without normalization.
- Blocks: Potential hallucinations (all-5 scores) and missing justifications go undetected in production.
- Fix approach: Integrate `EmotionNormalizer` into the pipeline after emotion extraction, either as a validation step or as an optional quality gate.

### No Support for Non-English Text
- Problem: The sentence splitting in `_split_sentences` (`pcmfg/utils/text_processing.py` line 335) only handles English sentence endings (`.!?`). The token estimation heuristic is also English-biased.
- Files: `pcmfg/utils/text_processing.py` (lines 324-337)
- Blocks: Analysis of non-English romance novels (Korean, Chinese, Japanese web novels).
- Fix approach: Add optional dependency on `nltk` or `spacy` for multilingual sentence splitting.

## Test Coverage Gaps

### LLM Client Integration
- What's not tested: Actual API call behavior (response parsing, retry logic, error mapping). All LLM client tests use mocks.
- Files: `pcmfg/llm/openai_client.py`, `pcmfg/llm/anthropic_client.py`
- Risk: Real API responses may differ from mock responses, causing runtime failures.
- Priority: Medium (integration tests are inherently flaky with external APIs, but edge case parsing should be tested)

### `merge_cli_overrides`
- What's not tested: Configuration merging with nested keys, type coercion, invalid keys.
- Files: `pcmfg/config.py` (lines 126-155)
- Risk: CLI argument parsing could silently produce wrong config values.
- Priority: High (easy to test, likely has bugs with type coercion)

### `analyze_with_checkpoint` Resume Logic
- What's not tested: Full resume from checkpoint, partial Phase 2 completion, checkpoint version mismatch.
- Files: `pcmfg/analyzer.py` (lines 171-406)
- Risk: Resume could load stale or incompatible checkpoint data.
- Priority: Medium (checkpoint logic is complex and has multiple code paths)

### Novel Loader Edge Cases
- What's not tested: Missing collection directories, malformed filenames, non-UTF-8 files, empty novels.
- Files: `pcmfg/utils/novel_loader.py`
- Risk: Loading novels with unexpected directory structures could crash.
- Priority: Low (currently works for known novel formats)

### Visualization Output
- What's not tested: All plot methods in `EmotionPlotter` and cluster plotters produce correct images.
- Files: `pcmfg/visualization/plotter.py`, `pcmfg/analysis/plotter.py`
- Risk: Plotting code changes could produce broken visualizations.
- Priority: Low (visual output is supplementary, not core pipeline)

### `ParallelProcessor` Thread Safety
- What's not tested: Concurrent execution with shared state, progress callback accuracy, error handling under concurrency.
- Files: `pcmfg/utils/parallel.py`
- Risk: Race conditions in progress tracking and checkpoint saves.
- Priority: High (known `nonlocal` race condition, see Tech Debt section)

---

*Concerns audit: 2026-04-23*
