# ASR Model Evaluation Results

Detailed ASR benchmark outputs are local-only because they are derived from private recordings and transcripts.

The public repository keeps the evaluation scripts but not the generated datasets or raw result JSON. To reproduce an evaluation, generate a private test set locally and run:

```bash
python3 scripts/asr_eval/run_eval.py \
  --test-set scripts/asr_eval/test_data/test_set.json \
  --models qwen3-1.7b,whisper-v3-turbo-mlx
```

Generated files under `scripts/asr_eval/test_data/` and `scripts/asr_eval/results/` are ignored by git.

When publishing findings, use aggregate metrics and synthetic examples only. Do not publish local audio paths, raw transcripts, or per-recording model outputs.
