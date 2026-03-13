// Qwen3CoreMLEncoder.swift
// CoreML wrapper for Qwen3-ASR encoder model (ANE-optimized)
// [AI-Claude: 2026-03-13]

import Foundation
import CoreML
import MLX
import os

enum Qwen3CoreMLEncoderError: Error, LocalizedError {
    case modelNotLoaded
    case predictionFailed(String)
    case invalidOutput(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Qwen3 CoreML encoder model not loaded"
        case .predictionFailed(let reason):
            return "Qwen3 CoreML encoder prediction failed: \(reason)"
        case .invalidOutput(let reason):
            return "Qwen3 CoreML encoder invalid output: \(reason)"
        }
    }
}

/// Wraps a CoreML-compiled Qwen3-ASR encoder model (.mlmodelc)
class Qwen3CoreMLEncoder {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3CoreMLEncoder")

    private var model: MLModel?
    private var inputFeatureName: String?
    private var outputFeatureName: String?

    /// Valid padding lengths for the encoder input time dimension
    private static let validPadLengths = [100, 200, 400, 600, 800, 1000, 1500, 2000, 3000]

    /// Load the encoder model from a directory containing .mlmodelc
    func load(from directory: URL) throws {
        let compiledPath = directory.appendingPathComponent("encoder.mlmodelc")

        guard FileManager.default.fileExists(atPath: compiledPath.path) else {
            throw Qwen3CoreMLEncoderError.modelNotLoaded
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all  // Let CoreML choose ANE/GPU/CPU

        Self.logger.info("Loading Qwen3 CoreML encoder from mlmodelc")
        let loadedModel = try MLModel(contentsOf: compiledPath, configuration: config)

        // Discover input/output feature names from model description
        let inputNames = loadedModel.modelDescription.inputDescriptionsByName.keys.sorted()
        let outputNames = loadedModel.modelDescription.outputDescriptionsByName.keys.sorted()
        Self.logger.info("CoreML encoder inputs: \(inputNames), outputs: \(outputNames)")

        guard let firstInput = inputNames.first else {
            throw Qwen3CoreMLEncoderError.predictionFailed("No input features found in model")
        }
        guard let firstOutput = outputNames.first else {
            throw Qwen3CoreMLEncoderError.predictionFailed("No output features found in model")
        }

        self.model = loadedModel
        self.inputFeatureName = firstInput
        self.outputFeatureName = firstOutput

        Self.logger.info("Qwen3 CoreML encoder loaded (input: \(firstInput), output: \(firstOutput))")
    }

    /// Encode mel spectrogram features to audio embeddings
    /// - Parameter melFeatures: MLXArray of shape [128, T] (nMels x timeFrames)
    /// - Returns: MLXArray of shape [1, T', dModel] where T' = paddedT/8
    func encode(melFeatures: MLXArray) throws -> MLXArray {
        guard let model = model,
              let inputName = inputFeatureName,
              let outputName = outputFeatureName else {
            throw Qwen3CoreMLEncoderError.modelNotLoaded
        }

        let nMels = melFeatures.dim(0)  // 128
        let timeFrames = melFeatures.dim(1)  // T

        // Pad time dimension to nearest valid length
        let paddedT = Self.nearestPadLength(for: timeFrames)
        let paddedMel: MLXArray
        if paddedT > timeFrames {
            let padding = MLXArray.zeros([nMels, paddedT - timeFrames])
            paddedMel = concatenated([melFeatures, padding], axis: 1)
        } else {
            paddedMel = melFeatures
        }

        // Convert MLXArray [128, paddedT] → MLMultiArray [1, 128, paddedT]
        eval(paddedMel)
        let inputMultiArray = try mlxArrayToMLMultiArray(paddedMel, shape: [1, nMels, paddedT])

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: inputMultiArray)
        ])

        let output = try model.prediction(from: inputFeatures)

        guard let encoderOutput = output.featureValue(for: outputName)?.multiArrayValue else {
            throw Qwen3CoreMLEncoderError.invalidOutput("Missing \(outputName) in model output")
        }

        // Convert MLMultiArray → MLXArray
        // Output shape is typically [1, paddedT/8, dModel]
        let outputShape = (0..<encoderOutput.shape.count).map { encoderOutput.shape[$0].intValue }
        let result = try mlMultiArrayToMLXArray(encoderOutput, shape: outputShape)

        // Trim padding: the encoder downsamples by 8x, so original tokens = timeFrames / 8
        let originalTokens = timeFrames / 8
        let trimmed: MLXArray
        if outputShape.count == 3 && outputShape[1] > originalTokens {
            trimmed = result[0..., 0..<originalTokens, 0...]
        } else {
            trimmed = result
        }

        return trimmed
    }

    /// Warm up the CoreML model with a small input to compile the ANE graph
    func warmUp() throws {
        guard model != nil else { return }
        Self.logger.info("Warming up Qwen3 CoreML encoder...")
        let dummyMel = MLXArray.zeros([128, 100])
        let _ = try encode(melFeatures: dummyMel)
        Self.logger.info("Qwen3 CoreML encoder warm-up complete")
    }

    func unload() {
        model = nil
        inputFeatureName = nil
        outputFeatureName = nil
    }

    // MARK: - Private Helpers

    /// Find the nearest valid padding length >= timeFrames
    private static func nearestPadLength(for timeFrames: Int) -> Int {
        for length in validPadLengths {
            if length >= timeFrames {
                return length
            }
        }
        // If exceeds all predefined lengths, round up to nearest multiple of 100
        return ((timeFrames + 99) / 100) * 100
    }

    /// Convert MLXArray to MLMultiArray using bulk memory copy
    private func mlxArrayToMLMultiArray(_ array: MLXArray, shape: [Int]) throws -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        let totalElements = shape.reduce(1, *)

        // Convert to Float32 for CoreML compatibility
        let floatArray = array.asType(.float32)
        eval(floatArray)

        let multiArray = try MLMultiArray(shape: nsShape, dataType: .float32)

        // Extract flat Float array and copy to MLMultiArray
        let floats: [Float] = floatArray.asArray(Float.self)
        floats.withUnsafeBufferPointer { srcBuffer in
            let destPtr = multiArray.dataPointer
            memcpy(destPtr, srcBuffer.baseAddress!, totalElements * MemoryLayout<Float>.size)
        }

        return multiArray
    }

    /// Convert MLMultiArray to MLXArray using bulk memory copy
    private func mlMultiArrayToMLXArray(_ multiArray: MLMultiArray, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)

        // Read as Float32
        let floatBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: totalElements)
        defer { floatBuffer.deallocate() }

        let srcPtr = multiArray.dataPointer
        memcpy(floatBuffer.baseAddress!, srcPtr, totalElements * MemoryLayout<Float>.size)

        let result = MLXArray(Array(floatBuffer)).reshaped(shape)
        return result
    }
}
