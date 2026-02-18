// Qwen3QuantizedTypes.swift
// Adapted from qwen3-asr-swift PreQuantizedEmbedding.swift
// [AI-Claude: 2025-02-18]

import Foundation
import MLX
import MLXNN

/// Pre-quantized embedding that can be loaded directly from safetensors
class Qwen3PreQuantizedEmbedding: Module {
    let groupSize: Int
    let bits: Int
    let embeddingCount: Int
    let dimensions: Int

    @ParameterInfo var weight: MLXArray
    @ParameterInfo var scales: MLXArray
    @ParameterInfo var biases: MLXArray

    init(embeddingCount: Int, dimensions: Int, groupSize: Int = 64, bits: Int = 4) {
        self.embeddingCount = embeddingCount
        self.dimensions = dimensions
        self.groupSize = groupSize
        self.bits = bits

        let packedDim = dimensions / (32 / bits)
        let numGroups = dimensions / groupSize

        self._weight.wrappedValue = MLXArray.zeros([embeddingCount, packedDim], dtype: .uint32)
        self._scales.wrappedValue = MLXArray.zeros([embeddingCount, numGroups], dtype: .bfloat16)
        self._biases.wrappedValue = MLXArray.zeros([embeddingCount, numGroups], dtype: .bfloat16)

        super.init()
        self.freeze()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let x = x.flattened()
        let out = dequantized(
            weight[x], scales: scales[x], biases: biases[x],
            groupSize: groupSize, bits: bits)
        return out.reshaped(s + [-1])
    }

    /// For use as LM head (matmul with transposed weight)
    func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMatmul(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits)
    }
}
