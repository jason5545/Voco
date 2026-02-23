#!/usr/bin/env python3
"""
Convert bert-base-chinese to Core ML FP16 for Voco's BERT MLM scoring.

Usage:
    pip install --break-system-packages torch transformers coremltools
    python scripts/convert_bert_coreml.py

Output:
    scripts/.cache/vocab.txt
    scripts/.cache/bert-base-chinese-mlm.mlpackage
    Then compile: xcrun coremlc compile <mlpackage> <output_dir>
"""

import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer


class BertMLMWrapper(torch.nn.Module):
    """Wrapper that pre-computes token_type_ids and extended attention mask,
    avoiding ops unsupported by coremltools (new_ones, etc.)."""

    def __init__(self, bert_mlm):
        super().__init__()
        self.bert = bert_mlm.bert
        self.cls = bert_mlm.cls

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Build token_type_ids as zeros (avoids new_ones in BertModel.forward)
        token_type_ids = torch.zeros_like(input_ids)

        # Get embeddings
        embeddings = self.bert.embeddings(input_ids, token_type_ids=token_type_ids)

        # Build extended attention mask manually (avoids new_ones in get_extended_attention_mask)
        # Shape: [batch, 1, 1, seq_len] with 0.0 for attend, -10000.0 for mask
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(embeddings.dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0

        # Run encoder
        encoder_output = self.bert.encoder(embeddings, attention_mask=extended_mask)
        sequence_output = encoder_output[0]

        # MLM head
        logits = self.cls(sequence_output)
        return logits


def main():
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)

    model_name = "bert-base-chinese"
    print(f"Loading {model_name}...")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()

    # Save vocab.txt
    vocab_path = cache_dir / "vocab.txt"
    vocab = tokenizer.get_vocab()
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in sorted_tokens:
            f.write(token + "\n")
    print(f"Saved vocab.txt ({vocab_path.stat().st_size / 1024:.0f} KB, {len(sorted_tokens)} tokens)")

    # Build wrapper and trace
    wrapper = BertMLMWrapper(model)
    wrapper.eval()

    seq_len = 128
    example_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), dtype=torch.long)
    example_mask = torch.ones(1, seq_len, dtype=torch.long)

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_ids, example_mask))

    # Verify trace output
    with torch.no_grad():
        test_out = traced(example_ids, example_mask)
    print(f"Trace output shape: {test_out.shape}")  # expect [1, 128, 21128]

    # Convert to Core ML
    print("Converting to Core ML FP16...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=512, default=128))),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=512, default=128))),
                dtype=np.int32,
            ),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )

    output_path = cache_dir / "bert-base-chinese-mlm.mlpackage"
    if output_path.exists():
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    print(f"Saved Core ML model: {output_path}")

    # Print info
    spec = mlmodel.get_spec()
    print(f"\nInputs:")
    for inp in spec.description.input:
        print(f"  {inp.name}: {inp.type}")
    print(f"Outputs:")
    for out in spec.description.output:
        print(f"  {out.name}: {out.type}")

    print(f"\nNext: xcrun coremlc compile '{output_path}' '{cache_dir}/'")


if __name__ == "__main__":
    main()
