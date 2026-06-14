# Voco Correction Model Demo

This directory contains a small, non-personal demo for Voco's two optional post-ASR correction model slots:

- `AutoApplyModels/full-db.auto-apply-model.json`
- `RuntimeCorrectionModels/runtime-correction-artifact.json`

The examples use generic technical phrases such as `Cloudflare`, `OpenAI`, and `VS Code`. They are intentionally not trained from Jason's transcripts.

## What This Demo Proves

- Voco can run with no correction models installed.
- When the auto-apply model is installed and enabled, it can apply small deterministic JSON rules.
- When the runtime correction artifact is installed and enabled, it can apply gated candidate-span fixes after the deterministic rules.
- Local audio-file import still skips both post-ASR correction models by policy; AI enhancements remain controlled by the existing audio-import switch.

## Files

```text
examples/correction-model-demo/
├── AutoApplyModels/
│   └── full-db.auto-apply-model.json
├── RuntimeCorrectionModels/
│   ├── runtime-correction-artifact.json
│   ├── models/
│   │   └── runtime-candidate-spans.json
│   └── replay-cases/
│       └── runtime-replay-cases.jsonl
└── README.md
```

## Install The Auto-Apply Demo

The app reads the auto-apply model from:

```bash
~/Library/Application\ Support/com.jasonchien.Voco/AutoApplyModels/full-db.auto-apply-model.json
```

Install the demo file:

```bash
mkdir -p ~/Library/Application\ Support/com.jasonchien.Voco/AutoApplyModels
cp examples/correction-model-demo/AutoApplyModels/full-db.auto-apply-model.json \
  ~/Library/Application\ Support/com.jasonchien.Voco/AutoApplyModels/full-db.auto-apply-model.json
```

Open Voco settings and make sure the auto-apply model toggle is enabled.

Expected demo behavior:

- Input: `open AI API key 要放在哪裡？`
- Output: `OpenAI API key 要放在哪裡？`

With programming context such as `editor 開發`:

- Input: `我用 VS code 開這個 repo。`
- Output: `我用 VS Code 開這個 repo。`

## Validate The Runtime Demo

The runtime artifact has a stricter install path because Voco requires the manifest, portable model checksum, and replay gate to agree.

Validate only:

```bash
python3 tools/voco_runtime_correction_control.py --json validateArtifact \
  --artifact examples/correction-model-demo/RuntimeCorrectionModels/runtime-correction-artifact.json \
  --replay-cases examples/correction-model-demo/RuntimeCorrectionModels/replay-cases/runtime-replay-cases.jsonl
```

Expected result: `"ready": true` and `"deployReady": true`.

## Install The Runtime Demo

Dry run first:

```bash
python3 tools/voco_runtime_correction_control.py --json installArtifact \
  --artifact examples/correction-model-demo/RuntimeCorrectionModels/runtime-correction-artifact.json \
  --replay-cases examples/correction-model-demo/RuntimeCorrectionModels/replay-cases/runtime-replay-cases.jsonl
```

Commit the install:

```bash
python3 tools/voco_runtime_correction_control.py --json installArtifact \
  --artifact examples/correction-model-demo/RuntimeCorrectionModels/runtime-correction-artifact.json \
  --replay-cases examples/correction-model-demo/RuntimeCorrectionModels/replay-cases/runtime-replay-cases.jsonl \
  --commit-install
```

The committed install writes to:

```text
~/Library/Application Support/com.jasonchien.Voco/RuntimeCorrectionModels/
```

Open Voco after installation and enable `VocoRuntimeCorrectionModelEnabled` only when you intentionally want to test the runtime gated-apply path.

Expected demo behavior with Cloudflare context:

- Input: `請把 Cloud Flare Pages 的設定打開。`
- Output: `請把 Cloudflare Pages 的設定打開。`

## Safety Notes

- Do not ship Jason's personal model data in public or partner builds.
- Treat this directory as example data only. It demonstrates file shape, install flow, and runtime gates.
- A real model should have a separate review record, replay cases, and approval token.
- Runtime correction is disabled by default. Missing artifacts must fall back to the post-rule text.
