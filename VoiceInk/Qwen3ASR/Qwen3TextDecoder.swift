// Qwen3TextDecoder.swift
// Adapted from qwen3-asr-swift QuantizedTextDecoder.swift
// Fixed: fatalError → throws
// [AI-Claude: 2025-02-18]

import Foundation
import MLX
import MLXNN
import MLXFast

enum Qwen3TextDecoderError: Error, LocalizedError {
    case noInputProvided

    var errorDescription: String? {
        switch self {
        case .noInputProvided:
            return "Either inputIds or inputsEmbeds must be provided to the text decoder"
        }
    }
}

/// Multi-head attention for Qwen3 text decoder with GQA and RoPE (quantized)
class Qwen3QuantizedTextAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var qProj: QuantizedLinear
    @ModuleInfo var kProj: QuantizedLinear
    @ModuleInfo var vProj: QuantizedLinear
    @ModuleInfo var oProj: QuantizedLinear
    @ModuleInfo var qNorm: RMSNorm
    @ModuleInfo var kNorm: RMSNorm

    let rope: MLXNN.RoPE

    init(config: Qwen3TextDecoderConfig) {
        self.numHeads = config.numHeads
        self.numKVHeads = config.numKVHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let hiddenSize = config.hiddenSize

        self._qProj.wrappedValue = QuantizedLinear(
            hiddenSize, numHeads * headDim, bias: false,
            groupSize: config.groupSize, bits: config.bits)
        self._kProj.wrappedValue = QuantizedLinear(
            hiddenSize, numKVHeads * headDim, bias: false,
            groupSize: config.groupSize, bits: config.bits)
        self._vProj.wrappedValue = QuantizedLinear(
            hiddenSize, numKVHeads * headDim, bias: false,
            groupSize: config.groupSize, bits: config.bits)
        self._oProj.wrappedValue = QuantizedLinear(
            numHeads * headDim, hiddenSize, bias: false,
            groupSize: config.groupSize, bits: config.bits)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        self.rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)

        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (batch, seqLen, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        queries = queries.reshaped(batch, seqLen, numHeads, headDim)
        keys = keys.reshaped(batch, seqLen, numKVHeads, headDim)
        values = values.reshaped(batch, seqLen, numKVHeads, headDim)

        queries = qNorm(queries)
        keys = kNorm(keys)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset = cache?.0.dim(2) ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        var cachedKeys = keys
        var cachedValues = values

        if let (prevKeys, prevValues) = cache {
            cachedKeys = concatenated([prevKeys, keys], axis: 2)
            cachedValues = concatenated([prevValues, values], axis: 2)
        }

        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: attentionMask)

        let output = oProj(attnOutput.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim))
        return (output, (cachedKeys, cachedValues))
    }
}

/// MLP for Qwen3 text decoder (SwiGLU activation, quantized)
class Qwen3QuantizedTextMLP: Module {
    @ModuleInfo var gateProj: QuantizedLinear
    @ModuleInfo var upProj: QuantizedLinear
    @ModuleInfo var downProj: QuantizedLinear

    init(config: Qwen3TextDecoderConfig) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self._gateProj.wrappedValue = QuantizedLinear(
            hiddenSize, intermediateSize, bias: false,
            groupSize: config.groupSize, bits: config.bits)
        self._upProj.wrappedValue = QuantizedLinear(
            hiddenSize, intermediateSize, bias: false,
            groupSize: config.groupSize, bits: config.bits)
        self._downProj.wrappedValue = QuantizedLinear(
            intermediateSize, hiddenSize, bias: false,
            groupSize: config.groupSize, bits: config.bits)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

/// Decoder layer for Qwen3 text model (quantized)
class Qwen3QuantizedTextDecoderLayer: Module {
    @ModuleInfo var selfAttn: Qwen3QuantizedTextAttention
    @ModuleInfo var mlp: Qwen3QuantizedTextMLP
    @ModuleInfo var inputLayerNorm: RMSNorm
    @ModuleInfo var postAttentionLayerNorm: RMSNorm

    init(config: Qwen3TextDecoderConfig) {
        self._selfAttn.wrappedValue = Qwen3QuantizedTextAttention(config: config)
        self._mlp.wrappedValue = Qwen3QuantizedTextMLP(config: config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let residual = hiddenStates
        var hidden = inputLayerNorm(hiddenStates)
        let (attnOutput, newCache) = selfAttn(hidden, attentionMask: attentionMask, cache: cache)
        hidden = residual + attnOutput

        let residual2 = hidden
        hidden = postAttentionLayerNorm(hidden)
        hidden = mlp(hidden)
        hidden = residual2 + hidden

        return (hidden, newCache)
    }
}

/// Full Qwen3 text decoder model (quantized)
class Qwen3QuantizedTextModel: Module {
    let config: Qwen3TextDecoderConfig

    @ModuleInfo var embedTokens: Qwen3PreQuantizedEmbedding
    @ModuleInfo var layers: [Qwen3QuantizedTextDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(config: Qwen3TextDecoderConfig) {
        self.config = config

        self._embedTokens.wrappedValue = Qwen3PreQuantizedEmbedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize,
            groupSize: config.groupSize,
            bits: config.bits)
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            Qwen3QuantizedTextDecoderLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        inputIds: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil
    ) throws -> (MLXArray, [(MLXArray, MLXArray)]) {
        var hiddenStates: MLXArray
        if let embeds = inputsEmbeds {
            hiddenStates = embeds
        } else if let ids = inputIds {
            hiddenStates = embedTokens(ids)
        } else {
            throw Qwen3TextDecoderError.noInputProvided
        }

        let seqLen = hiddenStates.dim(1)

        let mask: MLXArray?
        if let providedMask = attentionMask {
            mask = providedMask
        } else if seqLen == 1 {
            mask = nil
        } else {
            let cacheLen = cache?.first?.0.dim(2) ?? 0
            let totalLen = seqLen + cacheLen
            let rows = (MLXArray(0..<Int32(seqLen)) + Int32(cacheLen)).expandedDimensions(axis: 1)
            let cols = MLXArray(0..<Int32(totalLen)).expandedDimensions(axis: 0)
            mask = MLX.where(cols .> rows, MLXArray(Float(-1e9)), MLXArray(Float(0)))
                .expandedDimensions(axes: [0, 1])
                .asType(hiddenStates.dtype)
        }

        var newCache: [(MLXArray, MLXArray)] = []
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let (output, updatedCache) = layer(hiddenStates, attentionMask: mask, cache: layerCache)
            hiddenStates = output
            newCache.append(updatedCache)
        }

        hiddenStates = norm(hiddenStates)
        return (hiddenStates, newCache)
    }
}
