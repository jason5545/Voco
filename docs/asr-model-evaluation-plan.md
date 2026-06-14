# ASR Model Evaluation Plan

This document describes the local ASR evaluation workflow. The repository intentionally does not include Jason's private recordings, generated transcripts, or raw benchmark outputs.

## Goal

Compare local ASR engines on the kinds of utterances Voco cares about:

- Taiwanese Mandarin
- mixed Chinese and English technical terms
- short command-like utterances
- longer dictation
- noisy or low-confidence segments

## Private Data Boundary

Evaluation data is generated locally from the user's Voco database and recordings. These outputs may contain personal transcripts and local file paths, so they must stay out of git.

Ignored local outputs:

- `scripts/asr_eval/test_data/*.json`
- `scripts/asr_eval/results/*.json`
- `scripts/asr_eval/transcript.txt`

The checked-in `scripts/asr_eval/test_data/example_test_set.json` is schema-only example data.

## Workflow

1. Generate a private test set:

   ```bash
   python3 scripts/asr_eval/prepare_test_set.py
   ```

2. Run selected models:

   ```bash
   python3 scripts/asr_eval/run_eval.py \
     --test-set scripts/asr_eval/test_data/test_set.json \
     --models qwen3-1.7b,whisper-v3-turbo-mlx
   ```

3. Analyze generated results locally:

   ```bash
   python3 scripts/asr_eval/analyze_results.py
   ```

## Metrics

- Character error rate against a chosen local reference
- English technical term accuracy
- code-switching stability
- latency / realtime factor
- repeated-token or hallucination rate
- whether downstream correction can recover the ASR mistake

## Publishing Rule

Do not commit private evaluation JSON, raw transcripts, audio paths, or model outputs. If a result needs to be shared publicly, summarize it with synthetic examples or aggregate numbers that do not reveal private utterances.
