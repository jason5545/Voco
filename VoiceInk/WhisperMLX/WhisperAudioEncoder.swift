// WhisperAudioEncoder.swift
// Whisper audio encoder: Conv1d x2 + sinusoidal pos embed + Transformer
// [AI-Claude: 2026-02-27]

import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Self-Attention (encoder only, no cross-attention)

class WhisperEncoderSelfAttention: Module {
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)

        q = q.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        let scale = 1.0 / sqrt(Float(headDim))
        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil)

        return outProj(attnOutput.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim))
    }
}

// MARK: - Encoder Residual Attention Block

class WhisperEncoderBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: WhisperEncoderSelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(dims: Int, numHeads: Int, quantConfig: WhisperQuantizationConfig?) {
        self._selfAttn.wrappedValue = WhisperEncoderSelfAttention(dims: dims, numHeads: numHeads, quantConfig: quantConfig)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: dims)
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Pre-norm self-attention with residual
        var residual = x
        var hidden = selfAttnLayerNorm(x)
        hidden = selfAttn(hidden)
        hidden = residual + hidden

        // Pre-norm FFN with residual
        residual = hidden
        hidden = finalLayerNorm(hidden)
        hidden = fc1(hidden)
        hidden = gelu(hidden)
        hidden = fc2(hidden)
        hidden = residual + hidden

        return hidden
    }
}

// MARK: - Sinusoidal Position Embeddings

private func createSinusoidalPositionEmbeddings(seqLen: Int, dims: Int) -> MLXArray {
    let halfDim = dims / 2
    let maxTimescale: Float = 10000.0
    let logTimescaleIncrement = log(maxTimescale) / Float(halfDim - 1)
    let invTimescales = exp(
        MLXArray(0..<halfDim).asType(.float32) * (-logTimescaleIncrement)
    )
    let positions = MLXArray(0..<seqLen).asType(.float32)
    let scaledTime = positions.expandedDimensions(axis: 1) * invTimescales.expandedDimensions(axis: 0)
    let sinEmbed = sin(scaledTime)
    let cosEmbed = cos(scaledTime)
    return concatenated([sinEmbed, cosEmbed], axis: 1)
}

// MARK: - Audio Encoder

class WhisperAudioEncoder: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d
    @ModuleInfo var blocks: [WhisperEncoderBlock]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

    let positionalEmbedding: MLXArray
    let dims: Int

    init(config: WhisperMLXConfig) {
        let dims = config.nAudioState
        self.dims = dims

        // Conv1d: nMels → dims (kernel=3, pad=1)
        // Conv1d: dims → dims (kernel=3, stride=2, pad=1) — downsamples by 2x
        self._conv1.wrappedValue = Conv1d(
            inputChannels: config.nMels,
            outputChannels: dims,
            kernelSize: 3,
            padding: 1
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: dims,
            outputChannels: dims,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )

        let numHeads = config.nAudioHead
        let numLayers = config.nAudioLayer
        let quantConfig = config.quantization
        self._blocks.wrappedValue = (0..<numLayers).map { _ in
            WhisperEncoderBlock(dims: dims, numHeads: numHeads, quantConfig: quantConfig)
        }
        self._lnPost.wrappedValue = LayerNorm(dimensions: dims)

        // Sinusoidal position embeddings for max audio context
        self.positionalEmbedding = createSinusoidalPositionEmbeddings(
            seqLen: config.nAudioCtx, dims: dims
        )

        super.init()
    }

    func callAsFunction(_ melSpectrogram: MLXArray) -> MLXArray {
        // melSpectrogram shape: [batch, nMels, nFrames]
        // Conv1d expects: [batch, seqLen, channels]
        var x = melSpectrogram.transposed(0, 2, 1)  // [batch, nFrames, nMels]

        x = gelu(conv1(x))  // [batch, nFrames, dims]
        x = gelu(conv2(x))  // [batch, nFrames/2, dims]

        // Add sinusoidal position embeddings
        let seqLen = x.dim(1)
        let posEmb = positionalEmbedding[0..<seqLen].asType(x.dtype)
        x = x + posEmb

        for block in blocks {
            x = block(x)
        }

        x = lnPost(x)
        return x  // [batch, nFrames/2, dims]
    }
}
