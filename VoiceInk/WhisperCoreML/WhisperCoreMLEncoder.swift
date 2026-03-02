// WhisperCoreMLEncoder.swift
// CoreML wrapper for Whisper encoder model
// [AI-Claude: 2026-03-02]

import Foundation
import CoreML
import os

enum WhisperCoreMLEncoderError: Error, LocalizedError {
    case modelNotLoaded
    case predictionFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "CoreML encoder model not loaded"
        case .predictionFailed(let reason):
            return "CoreML encoder prediction failed: \(reason)"
        }
    }
}

/// Wraps the WhisperEncoder.mlpackage CoreML model
class WhisperCoreMLEncoder {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperCoreMLEncoder")

    private var model: MLModel?

    /// Load the encoder model from a compiled .mlmodelc or .mlpackage
    func load(from directory: URL) throws {
        let compiledPath = directory.appendingPathComponent("WhisperEncoder.mlmodelc")
        let packagePath = directory.appendingPathComponent("WhisperEncoder.mlpackage")

        let config = MLModelConfiguration()
        config.computeUnits = .all  // Let CoreML choose ANE/GPU/CPU

        if FileManager.default.fileExists(atPath: compiledPath.path) {
            Self.logger.info("Loading pre-compiled encoder from mlmodelc")
            model = try MLModel(contentsOf: compiledPath, configuration: config)
        } else if FileManager.default.fileExists(atPath: packagePath.path) {
            Self.logger.info("Compiling encoder from mlpackage (first launch)")
            let compiledURL = try MLModel.compileModel(at: packagePath)
            // Move compiled model next to package for future launches
            let destURL = compiledPath
            try? FileManager.default.removeItem(at: destURL)
            try FileManager.default.moveItem(at: compiledURL, to: destURL)
            model = try MLModel(contentsOf: destURL, configuration: config)
        } else {
            throw WhisperCoreMLEncoderError.modelNotLoaded
        }

        Self.logger.info("CoreML encoder loaded")
    }

    /// Encode mel spectrogram to encoder output
    /// - Parameter mel: MLMultiArray of shape [1, nMels, 3000] Float16
    /// - Returns: MLMultiArray of shape [1, 1500, dModel] Float16
    func encode(mel: MLMultiArray) throws -> MLMultiArray {
        guard let model = model else {
            throw WhisperCoreMLEncoderError.modelNotLoaded
        }

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "mel_spectrogram": MLFeatureValue(multiArray: mel)
        ])

        let output = try model.prediction(from: inputFeatures)

        guard let encoderOutput = output.featureValue(for: "encoder_output")?.multiArrayValue else {
            throw WhisperCoreMLEncoderError.predictionFailed("Missing encoder_output in model output")
        }

        return encoderOutput
    }

    func unload() {
        model = nil
    }
}
