import Foundation
import os

/// Manages BERT model files bundled within the app.
///
/// Expected bundle resources (under `BERTModel/`):
/// - `vocab.txt`                          (~110 KB)
/// - `bert-base-chinese-mlm.mlmodelc/`    (~195 MB, compiled Core ML model)
enum BERTModelManager {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "BERTModelManager")

    private static let vocabFileName = "vocab.txt"
    private static let modelDirName = "bert-base-chinese-mlm"

    // MARK: - Bundle Paths

    /// URL of the vocabulary file in the app bundle.
    static var vocabURL: URL? {
        Bundle.main.url(forResource: vocabFileName, withExtension: nil)
    }

    /// URL of the compiled Core ML model directory in the app bundle.
    static var modelURL: URL? {
        Bundle.main.url(forResource: modelDirName, withExtension: "mlmodelc")
    }

    // MARK: - Status

    /// Check if the BERT model is available in the app bundle.
    static var isModelAvailable: Bool {
        guard let vocab = vocabURL, let model = modelURL else { return false }
        let fm = FileManager.default
        return fm.fileExists(atPath: vocab.path) && fm.fileExists(atPath: model.path)
    }
}
