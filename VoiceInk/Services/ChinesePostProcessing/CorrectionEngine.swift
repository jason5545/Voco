import Foundation

/// Shared protocol for data-driven correction engines.
///
/// Conforming engines are used in a pipeline loop by `ChinesePostProcessingService`.
/// Each engine takes a plain `String`, returns a `CorrectionResult` with the
/// corrected text and a list of individual corrections applied.
protocol CorrectionEngine {
    /// Human-readable name shown in pipeline trace (e.g. "HomophoneCorrection").
    var name: String { get }

    /// Log prefix for debug output (e.g. "[data]", "[nasal]", "[expand]").
    var logPrefix: String { get }

    /// Apply corrections to the given text.
    func correct(_ text: String) -> CorrectionResult
}

/// Result of a single correction engine pass.
struct CorrectionResult {
    let text: String
    let corrections: [Correction]

    struct Correction {
        let original: String
        let corrected: String
        let score: Double
    }
}
