// WhisperTextDecoder.swift
// Whisper text decoder: token + pos embeddings, self-attention + cross-attention, KV cache
// [AI-Claude: 2026-02-27]

import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Multi-Head Attention (supports both self and cross attention)

class WhisperMultiHeadAttention: Module {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dims: Int, numHeads: Int, quantConfig: WhisperQuantizationConfig?) {
        self.numHeads = numHeads
        self.headDim = dims / numHeads

        if let qc = quantConfig {
            self._qProj.wrappedValue = QuantizedLinear(dims, dims, bias: true, groupSize: qc.groupSize, bits: qc.bits)
            self._kProj.wrappedValue = QuantizedLinear(dims, dims, bias: false, groupSize: qc.groupSize, bits: qc.bits)
            self._vProj.wrappedValue = QuantizedLinear(dims, dims, bias: true, groupSize: qc.groupSize, bits: qc.bits)
            self._outProj.wrappedValue = QuantizedLinear(dims, dims, bias: true, groupSize: qc.groupSize, bits: qc.bits)
        } else {
            self._qProj.wrappedValue = Linear(dims, dims, bias: true)
            self._kProj.wrappedValue = Linear(dims, dims, bias: false)
            self._vProj.wrappedValue = Linear(dims, dims, bias: true)
            self._outProj.wrappedValue = Linear(dims, dims, bias: true)
        }

        super.init()
    }

    /// Self-attention with KV cache
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        kvCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        return attend(queries: x, mask: mask, kvCache: kvCache)
    }

    /// Cross-attention: queries from decoder, keys/values from encoder
    /// K/V are computed once from encoder output and cached for subsequent tokens
    func crossAttention(
        queries: MLXArray,
        encoderOutput: MLXArray,
        kvCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (batch, seqLen, _) = (queries.dim(0), queries.dim(1), queries.dim(2))

        var q = qProj(queries)
        q = q.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        let k: MLXArray
        let v: MLXArray
        if let (cachedK, cachedV) = kvCache {
            // Reuse cached K/V (encoder output doesn't change between tokens)
            k = cachedK
            v = cachedV
        } else {
            // First pass: compute K/V from encoder output
            var kRaw = kProj(encoderOutput)
            var vRaw = vProj(encoderOutput)
            k = kRaw.reshaped(batch, -1, numHeads, headDim).transposed(0, 2, 1, 3)
            v = vRaw.reshaped(batch, -1, numHeads, headDim).transposed(0, 2, 1, 3)
        }

        let scale = pow(Float(headDim), -0.25)
        let scaledQ = q * scale

        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: scaledQ, keys: k, values: v, scale: scale, mask: nil)

        let out = attnOutput.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim)
        return (outProj(out), (k, v))
    }

    /// Self-attention with KV cache (appends new K/V to cache)
    private func attend(
        queries: MLXArray,
        mask: MLXArray?,
        kvCache: (MLXArray, MLXArray)?
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (batch, seqLen, _) = (queries.dim(0), queries.dim(1), queries.dim(2))

        var q = qProj(queries)
        var k = kProj(queries)
        var v = vProj(queries)

        q = q.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        if let (prevK, prevV) = kvCache {
            k = concatenated([prevK, k], axis: 2)
            v = concatenated([prevV, v], axis: 2)
        }

        let scale = pow(Float(headDim), -0.25)
        let scaledQ = q * scale

        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: scaledQ, keys: k, values: v, scale: scale, mask: mask)

        let out = attnOutput.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim)
        return (outProj(out), (k, v))
    }
}

// MARK: - Decoder Block (self-attention + cross-attention + FFN)

class WhisperDecoderBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: WhisperMultiHeadAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var crossAttn: WhisperMultiHeadAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var crossAttnLayerNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(dims: Int, numHeads: Int, quantConfig: WhisperQuantizationConfig?) {
        self._selfAttn.wrappedValue = WhisperMultiHeadAttention(dims: dims, numHeads: numHeads, quantConfig: quantConfig)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: dims)
        self._crossAttn.wrappedValue = WhisperMultiHeadAttention(dims: dims, numHeads: numHeads, quantConfig: quantConfig)
        self._crossAttnLayerNorm.wrappedValue = LayerNorm(dimensions: dims)

        if let qc = quantConfig {
            self._fc1.wrappedValue = QuantizedLinear(dims, dims * 4, bias: true, groupSize: qc.groupSize, bits: qc.bits)
            self._fc2.wrappedValue = QuantizedLinear(dims * 4, dims, bias: true, groupSize: qc.groupSize, bits: qc.bits)
        } else {
            self._fc1.wrappedValue = Linear(dims, dims * 4, bias: true)
            self._fc2.wrappedValue = Linear(dims * 4, dims, bias: true)
        }
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: dims)

        super.init()
    }

    struct DecoderBlockCache {
        var selfAttnCache: (MLXArray, MLXArray)?
        var crossAttnCache: (MLXArray, MLXArray)?
    }

    func callAsFunction(
        _ x: MLXArray,
        encoderOutput: MLXArray,
        mask: MLXArray? = nil,
        cache: DecoderBlockCache?
    ) -> (MLXArray, DecoderBlockCache) {
        // Self-attention
        var residual = x
        var hidden = selfAttnLayerNorm(x)
        let (saOut, newSaCache) = selfAttn(hidden, mask: mask, kvCache: cache?.selfAttnCache)
        hidden = residual + saOut

        // Cross-attention
        residual = hidden
        hidden = crossAttnLayerNorm(hidden)
        let (caOut, newCaCache) = crossAttn.crossAttention(
            queries: hidden, encoderOutput: encoderOutput, kvCache: cache?.crossAttnCache)
        hidden = residual + caOut

        // FFN
        residual = hidden
        hidden = finalLayerNorm(hidden)
        hidden = fc1(hidden)
        hidden = gelu(hidden)
        hidden = fc2(hidden)
        hidden = residual + hidden

        let newCache = DecoderBlockCache(selfAttnCache: newSaCache, crossAttnCache: newCaCache)
        return (hidden, newCache)
    }
}

// MARK: - Text Decoder

class WhisperTextDecoder: Module {
    let config: WhisperMLXConfig

    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo var blocks: [WhisperDecoderBlock]
    @ModuleInfo(key: "ln") var ln: LayerNorm

    /// Learned positional embeddings [nTextCtx, dims]
    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray

    init(config: WhisperMLXConfig) {
        self.config = config

        if let qc = config.quantization {
            self._tokenEmbedding.wrappedValue = QuantizedEmbedding(
                embeddingCount: config.nVocab,
                dimensions: config.nTextState,
                groupSize: qc.groupSize,
                bits: qc.bits
            )
        } else {
            self._tokenEmbedding.wrappedValue = Embedding(
                embeddingCount: config.nVocab,
                dimensions: config.nTextState
            )
        }

        self._blocks.wrappedValue = (0..<config.nTextLayer).map { _ in
            WhisperDecoderBlock(
                dims: config.nTextState,
                numHeads: config.nTextHead,
                quantConfig: config.quantization
            )
        }
        self._ln.wrappedValue = LayerNorm(dimensions: config.nTextState)
        self._positionalEmbedding.wrappedValue = MLXArray.zeros([config.nTextCtx, config.nTextState])

        super.init()
    }

    func callAsFunction(
        tokenIds: MLXArray,
        encoderOutput: MLXArray,
        cache: [WhisperDecoderBlock.DecoderBlockCache]?
    ) -> (MLXArray, [WhisperDecoderBlock.DecoderBlockCache]) {
        let offset = cache?.first?.selfAttnCache?.0.dim(2) ?? 0
        let seqLen = tokenIds.dim(1)

        // Token embedding + positional embedding
        var x = tokenEmbedding(tokenIds)
        x = x + positionalEmbedding[offset..<(offset + seqLen)]

        // Causal mask for self-attention
        let mask: MLXArray?
        if seqLen > 1 {
            let totalLen = seqLen + offset
            let rows = (MLXArray(0..<Int32(seqLen)) + Int32(offset)).expandedDimensions(axis: 1)
            let cols = MLXArray(0..<Int32(totalLen)).expandedDimensions(axis: 0)
            mask = MLX.where(cols .> rows, MLXArray(Float(-1e9)), MLXArray(Float(0)))
                .expandedDimensions(axes: [0, 1])
                .asType(x.dtype)
        } else {
            mask = nil
        }

        var newCaches: [WhisperDecoderBlock.DecoderBlockCache] = []
        for (i, block) in blocks.enumerated() {
            let blockCache = cache?[i]
            let (output, newBlockCache) = block(x, encoderOutput: encoderOutput, mask: mask, cache: blockCache)
            x = output
            newCaches.append(newBlockCache)
        }

        x = ln(x)

        // Project back to vocab logits via transposed token embedding weight
        let logits: MLXArray
        if let qEmb = tokenEmbedding as? QuantizedEmbedding {
            logits = quantizedMatmul(
                x, qEmb.weight, scales: qEmb.scales, biases: qEmb.biases,
                transpose: true, groupSize: config.quantization?.groupSize, bits: config.quantization?.bits
            )
        } else {
            logits = x.matmul(tokenEmbedding.weight.transposed())
        }

        return (logits, newCaches)
    }
}
