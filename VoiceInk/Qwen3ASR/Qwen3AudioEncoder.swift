// Qwen3AudioEncoder.swift
// Adapted from qwen3-asr-swift AudioEncoder.swift
// [AI-Claude: 2025-02-18]

import Foundation
import MLX
import MLXNN
import MLXFast

/// Multi-head self-attention for audio encoder layers
class Qwen3AudioSelfAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))

        self._qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x) * scale
        var k = kProj(x)
        var v = vProj(x)

        q = q.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: 1.0, mask: attentionMask)

        let out = attnOutput.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim)
        return outProj(out)
    }
}

/// Audio encoder transformer layer
class Qwen3AudioEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3AudioSelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(hiddenSize: Int, numHeads: Int, ffnDim: Int, layerNormEps: Float) {
        self._selfAttn.wrappedValue = Qwen3AudioSelfAttention(hiddenSize: hiddenSize, numHeads: numHeads)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: layerNormEps)
        self._fc1.wrappedValue = Linear(hiddenSize, ffnDim, bias: true)
        self._fc2.wrappedValue = Linear(ffnDim, hiddenSize, bias: true)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: layerNormEps)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var residual = x
        var hidden = selfAttnLayerNorm(x)
        hidden = selfAttn(hidden, attentionMask: attentionMask)
        hidden = residual + hidden

        residual = hidden
        hidden = finalLayerNorm(hidden)
        hidden = fc1(hidden)
        hidden = gelu(hidden)
        hidden = fc2(hidden)
        hidden = residual + hidden

        return hidden
    }
}

/// Create sinusoidal position embeddings
private func createSinusoidalPositionEmbeddings(seqLen: Int, dModel: Int) -> MLXArray {
    let halfDim = dModel / 2
    let maxTimescale: Float = 10000.0

    let logTimescaleIncrement = log(maxTimescale) / Float(halfDim - 1)
    let invTimescales = exp(
        MLXArray(0..<halfDim).asType(.float32) * (-logTimescaleIncrement)
    )

    let positions = MLXArray(0..<seqLen).asType(.float32)
    let scaledTime = positions.expandedDimensions(axis: 1) * invTimescales.expandedDimensions(axis: 0)

    let sinEmbed = sin(scaledTime)
    let cosEmbed = cos(scaledTime)
    let posEmbed = concatenated([sinEmbed, cosEmbed], axis: 1)

    return posEmbed.expandedDimensions(axis: 0)
}

/// Full Qwen3-ASR Audio Encoder (audio_tower)
class Qwen3AudioEncoder: Module {
    let config: Qwen3AudioEncoderConfig

    private static let maxCachedEmbeddings = 8
    private var cachedPosEmbeddings: [Int: MLXArray] = [:]
    private var cachedPosOrder: [Int] = []  // LRU tracking

    /// Clear cached position embeddings to free GPU memory
    func clearPosEmbeddingCache() {
        cachedPosEmbeddings.removeAll()
        cachedPosOrder.removeAll()
    }

    @ModuleInfo var conv2d1: Conv2d
    @ModuleInfo var conv2d2: Conv2d
    @ModuleInfo var conv2d3: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Linear
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo var proj1: Linear
    @ModuleInfo var proj2: Linear
    @ModuleInfo var layers: [Qwen3AudioEncoderLayer]

    init(config: Qwen3AudioEncoderConfig = .default) {
        self.config = config

        self._conv2d1.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1)
        )
        self._conv2d2.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1)
        )
        self._conv2d3.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1)
        )

        self._convOut.wrappedValue = Linear(config.convOutInputDim, config.dModel, bias: false)
        self._lnPost.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        self._proj1.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
        self._proj2.wrappedValue = Linear(config.dModel, config.outputDim, bias: true)

        self._layers.wrappedValue = (0..<config.encoderLayers).map { _ in
            Qwen3AudioEncoderLayer(
                hiddenSize: config.dModel,
                numHeads: config.encoderAttentionHeads,
                ffnDim: config.encoderFFNDim,
                layerNormEps: config.layerNormEps
            )
        }

        super.init()
    }

    private func getOutputLength(_ inputLength: Int) -> Int {
        let chunkSize = config.nWindow * 2
        let remainder = inputLength % chunkSize

        var featLen = (remainder - 1) / 2 + 1
        featLen = (featLen - 1) / 2 + 1
        featLen = (featLen - 1) / 2 + 1

        let fullChunkTokens = (inputLength / chunkSize) * 13
        let remainderTokens = remainder > 0 ? max(featLen, 1) : 0

        return fullChunkTokens + remainderTokens
    }

    private func createBlockAttentionMask(seqLen: Int, cuSeqlens: [Int]) -> MLXArray {
        var blockIds = [Int32](repeating: 0, count: seqLen)
        for i in 0..<(cuSeqlens.count - 1) {
            let start = cuSeqlens[i]
            let end = cuSeqlens[i + 1]
            for pos in start..<end {
                blockIds[pos] = Int32(i)
            }
        }

        let rowIds = MLXArray(blockIds).expandedDimensions(axis: 1)
        let colIds = MLXArray(blockIds).expandedDimensions(axis: 0)

        let mask = MLX.where(rowIds .== colIds, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        return mask.expandedDimensions(axes: [0, 1])
    }

    /// Process mel spectrogram with time chunking
    /// Input: [batch, mel_bins, time]
    /// Output: [time', output_dim] (no batch dim)
    func callAsFunction(_ melFeatures: MLXArray) -> MLXArray {
        let timeFrames = melFeatures.dim(2)
        let chunkSize = config.nWindow * 2

        let numChunks = (timeFrames + chunkSize - 1) / chunkSize

        var chunkLengths = [Int]()
        for i in 0..<numChunks {
            if i == numChunks - 1 {
                let remainder = timeFrames % chunkSize
                chunkLengths.append(remainder == 0 ? chunkSize : remainder)
            } else {
                chunkLengths.append(chunkSize)
            }
        }

        let maxChunkLen = chunkLengths.max() ?? chunkSize

        var paddedChunks: [MLXArray] = []
        var pos = 0
        for i in 0..<numChunks {
            let clen = chunkLengths[i]
            let feat = melFeatures[0, 0..., pos..<(pos + clen)]
            pos += clen

            var chunk: MLXArray
            if clen < maxChunkLen {
                let padWidth = maxChunkLen - clen
                chunk = padded(feat, widths: [.init((low: 0, high: 0)), .init((low: 0, high: padWidth))])
            } else {
                chunk = feat
            }
            paddedChunks.append(chunk)
        }

        let paddedFeature = stacked(paddedChunks, axis: 0)
        var x = paddedFeature.expandedDimensions(axis: -1)

        x = conv2d1(x)
        x = gelu(x)
        x = conv2d2(x)
        x = gelu(x)
        x = conv2d3(x)
        x = gelu(x)

        let numChunksBatch = x.dim(0)
        let freq = x.dim(1)
        let timeAfterConv = x.dim(2)
        let channels = x.dim(3)

        x = x.transposed(0, 2, 3, 1)
        x = x.reshaped(numChunksBatch, timeAfterConv, channels * freq)

        x = convOut(x)

        let posEmbed: MLXArray
        if let cached = cachedPosEmbeddings[timeAfterConv] {
            posEmbed = cached
            // Move to end of LRU order
            if let idx = cachedPosOrder.firstIndex(of: timeAfterConv) {
                cachedPosOrder.remove(at: idx)
            }
            cachedPosOrder.append(timeAfterConv)
        } else {
            let computed = createSinusoidalPositionEmbeddings(seqLen: timeAfterConv, dModel: config.dModel)
            cachedPosEmbeddings[timeAfterConv] = computed
            cachedPosOrder.append(timeAfterConv)
            // Evict oldest entry if cache is full
            if cachedPosOrder.count > Self.maxCachedEmbeddings {
                let evictKey = cachedPosOrder.removeFirst()
                cachedPosEmbeddings.removeValue(forKey: evictKey)
            }
            posEmbed = computed
        }
        x = x + posEmbed

        var featureLensAfterCnn = [Int]()
        for clen in chunkLengths {
            var featLen = (clen - 1) / 2 + 1
            featLen = (featLen - 1) / 2 + 1
            featLen = (featLen - 1) / 2 + 1
            featureLensAfterCnn.append(featLen)
        }

        var hiddenList: [MLXArray] = []
        for i in 0..<numChunks {
            let validLen = featureLensAfterCnn[i]
            let chunkHidden = x[i, 0..<validLen, 0...]
            hiddenList.append(chunkHidden)
        }

        var hiddenStates = concatenated(hiddenList, axis: 0)
        let totalTokens = hiddenStates.dim(0)

        let maxLenAfterCnn = featureLensAfterCnn.max() ?? 13
        let windowAfterCnn = maxLenAfterCnn * (config.nWindowInfer / (config.nWindow * 2))

        let totalCnnLen = getOutputLength(timeFrames)

        var cuChunkLens = [Int]()
        let numFullWindows = totalCnnLen / windowAfterCnn
        for _ in 0..<numFullWindows {
            cuChunkLens.append(windowAfterCnn)
        }
        let windowRemainder = totalCnnLen % windowAfterCnn
        if windowRemainder != 0 {
            cuChunkLens.append(windowRemainder)
        }

        var cuSeqlens = [0]
        var cumsum = 0
        for len in cuChunkLens {
            cumsum += len
            cuSeqlens.append(cumsum)
        }

        let attentionMask = createBlockAttentionMask(seqLen: totalTokens, cuSeqlens: cuSeqlens)

        hiddenStates = hiddenStates.expandedDimensions(axis: 0)

        for layer in layers {
            hiddenStates = layer(hiddenStates, attentionMask: attentionMask)
        }

        hiddenStates = hiddenStates.squeezed(axis: 0)
        hiddenStates = lnPost(hiddenStates)
        hiddenStates = proj1(hiddenStates)
        hiddenStates = gelu(hiddenStates)
        hiddenStates = proj2(hiddenStates)

        return hiddenStates
    }
}
