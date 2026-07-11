import Foundation

struct Qwen3ASRSpecialistTriggerDecision: Codable, Equatable, Sendable {
    let triggered: Bool
    let reasons: [String]
    let requiredSurfaces: [String]
}

struct Qwen3ASRSpecialistSelectionDecision: Codable, Equatable, Sendable {
    let selectSpecialist: Bool
    let reason: String
    let editDistance: Int?
    let editBudget: Int?
    let residualEditDistance: Int?
    let residualEditBudget: Int?
}

struct Qwen3ASRSpecialistRoutingMetadata: Codable, Equatable, Sendable {
    let specialistID: String
    let baselineTranscript: String
    let specialistTranscript: String?
    let chosenTranscript: String
    let trigger: Qwen3ASRSpecialistTriggerDecision
    let selection: Qwen3ASRSpecialistSelectionDecision
    let specialistAdapter: Qwen3ASRAdapterMetadata?
}

enum Qwen3ASRSpecialistRouter {
    static let specialistID = "support-firmware-v1"

    private static let firmwareAmbiguities = [
        "人體", "人体", "任體", "任体", "認體", "认体",
        "軟體", "软体", "軟件", "软件", "論題", "论题", "論體", "论体",
    ]

    static func triggerDecision(
        baselineTranscript: String,
        recentTranscriptions: [String],
        prompt: String? = nil
    ) -> Qwen3ASRSpecialistTriggerDecision {
        let baseline = baselineTranscript
        let context = (recentTranscriptions + [prompt].compactMap { $0 }).joined(separator: "\n")
        var reasons: [String] = []
        var required: [String] = []

        if let firmwareSurface = firmwareAmbiguities.first(where: baseline.contains) {
            let sameAnchor = matches(
                "Unicode|BIOS|boot(?:loader)?|driver|firmware|韌體|韧体",
                in: baseline
            )
            let explicitContext = matches("firmware|韌體|韧体|Unicode", in: context)
            let trainingContext = matches("adapter|adopter|adaptor|ASR|LoRA", in: context)
                && matches("重新訓練|重新训练|訓練|训练|retrain|模型", in: context)
            let softwareSurface = ["軟體", "软体", "軟件", "软件"].contains(firmwareSurface)
            if sameAnchor || explicitContext || (trainingContext && !softwareSurface) {
                reasons.append("firmwareAmbiguity:\(firmwareSurface)")
                required.append("韌體")
            }
        }

        if matches("資源|资源", in: baseline) {
            let sameAnchor = (
                matches("軟體|软体|軟件|软件|software", in: baseline)
                    && matches("差|不好|問題|问题", in: baseline)
            ) || (
                matches("adapter|adopter|adaptor", in: baseline)
                    && matches("重新訓練|重新训练|訓練|训练|retrain", in: baseline)
            )
            let contextAnchor = matches("support|支援", in: context)
                && matches("軟體|软体|軟件|软件|software|adapter|ASR|LoRA|模型", in: context)
            if sameAnchor || contextAnchor {
                reasons.append("supportAmbiguity:資源")
                required.append("支援")
            }
        }

        if matches("\\b(?:adopter|adaptor)\\b", in: baseline)
            && matches("重新訓練|重新训练|訓練|训练|retrain|ASR|LoRA|模型", in: baseline + "\n" + context)
        {
            reasons.append("adapterAmbiguity")
            required.append("adapter")
        }

        return Qwen3ASRSpecialistTriggerDecision(
            triggered: !reasons.isEmpty,
            reasons: reasons,
            requiredSurfaces: deduplicated(required)
        )
    }

    static func selectionDecision(
        baselineTranscript: String,
        specialistTranscript: String?,
        trigger: Qwen3ASRSpecialistTriggerDecision
    ) -> Qwen3ASRSpecialistSelectionDecision {
        guard trigger.triggered else {
            return decision(false, "notTriggered")
        }
        guard let specialistTranscript, !specialistTranscript.isEmpty else {
            return decision(false, "missingSpecialistPrediction")
        }
        let missing = trigger.requiredSurfaces.filter {
            !surfaceSatisfied($0, in: specialistTranscript)
        }
        guard missing.isEmpty else {
            return decision(false, "specialistMissingRequiredSurface")
        }

        let baseKey = metricKey(baselineTranscript)
        let specialistKey = metricKey(specialistTranscript)
        let distance = levenshtein(baseKey, specialistKey)
        let budget = max(3, Int(ceil(Double(max(baseKey.count, 1)) * 0.15)))
        guard distance <= budget else {
            return decision(false, "specialistExceededEditBudget", distance, budget)
        }

        let residualDistance = residualEditDistance(
            baselineTranscript: baselineTranscript,
            specialistTranscript: specialistTranscript,
            requiredSurfaces: trigger.requiredSurfaces
        )
        let residualBudget = 1
        guard residualDistance <= residualBudget else {
            return decision(
                false,
                "specialistChangedOutsideTargetSurface",
                distance,
                budget,
                residualDistance,
                residualBudget
            )
        }
        return decision(
            true,
            "requiredSurfaceAndNarrowEditPassed",
            distance,
            budget,
            residualDistance,
            residualBudget
        )
    }

    /// The specialist confirms the ambiguous target, but the baseline owns all
    /// surrounding text. This prevents a valid target fix from importing a
    /// nearby specialist drift such as `支援度` when the baseline said `資源都`.
    static func mergeSelectedTarget(
        baselineTranscript: String,
        trigger: Qwen3ASRSpecialistTriggerDecision,
        selection: Qwen3ASRSpecialistSelectionDecision
    ) -> String {
        guard selection.selectSpecialist else { return baselineTranscript }
        var merged = baselineTranscript
        if trigger.requiredSurfaces.contains("韌體") {
            for surface in firmwareAmbiguities {
                merged = merged.replacingOccurrences(of: surface, with: "韌體")
            }
        }
        if trigger.requiredSurfaces.contains("支援") {
            for surface in ["資源", "资源"] {
                merged = merged.replacingOccurrences(of: surface, with: "支援")
            }
        }
        if trigger.requiredSurfaces.contains("adapter") {
            merged = replacingRegex("\\b(?:adopter|adaptor)\\b", in: merged, with: "adapter")
        }
        return merged
    }

    private static func decision(
        _ selected: Bool,
        _ reason: String,
        _ editDistance: Int? = nil,
        _ editBudget: Int? = nil,
        _ residualEditDistance: Int? = nil,
        _ residualEditBudget: Int? = nil
    ) -> Qwen3ASRSpecialistSelectionDecision {
        Qwen3ASRSpecialistSelectionDecision(
            selectSpecialist: selected,
            reason: reason,
            editDistance: editDistance,
            editBudget: editBudget,
            residualEditDistance: residualEditDistance,
            residualEditBudget: residualEditBudget
        )
    }

    private static func residualEditDistance(
        baselineTranscript: String,
        specialistTranscript: String,
        requiredSurfaces: [String]
    ) -> Int {
        var baseline = baselineTranscript
        var specialist = specialistTranscript
        if requiredSurfaces.contains("韌體") {
            for surface in firmwareAmbiguities + ["韌體", "韧体"] {
                baseline = baseline.replacingOccurrences(of: surface, with: "<firmware>")
                specialist = specialist.replacingOccurrences(of: surface, with: "<firmware>")
            }
        }
        if requiredSurfaces.contains("支援") {
            for surface in ["資源", "资源", "支援"] {
                baseline = baseline.replacingOccurrences(of: surface, with: "<support>")
                specialist = specialist.replacingOccurrences(of: surface, with: "<support>")
            }
        }
        if requiredSurfaces.contains("adapter") {
            baseline = replacingRegex("\\b(?:adopter|adaptor|adapter)\\b", in: baseline, with: "<adapter>")
            specialist = replacingRegex("\\b(?:adopter|adaptor|adapter)\\b", in: specialist, with: "<adapter>")
        }
        return levenshtein(metricKey(baseline), metricKey(specialist))
    }

    private static func surfaceSatisfied(_ surface: String, in text: String) -> Bool {
        if surface == "adapter" {
            return matches("\\badapter\\b", in: text)
        }
        if surface == "韌體" {
            return text.contains("韌體") || text.contains("韧体")
        }
        return text.contains(surface)
    }

    private static func metricKey(_ text: String) -> String {
        replacingRegex("[\\s，。！？、,.!?：:；;（）()\\[\\]{}\\-]", in: text.lowercased(), with: "")
    }

    private static func matches(_ pattern: String, in text: String) -> Bool {
        text.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil
    }

    private static func replacingRegex(_ pattern: String, in text: String, with replacement: String) -> String {
        text.replacingOccurrences(
            of: pattern,
            with: replacement,
            options: [.regularExpression, .caseInsensitive]
        )
    }

    private static func deduplicated(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.filter { seen.insert($0).inserted }
    }

    private static func levenshtein(_ lhs: String, _ rhs: String) -> Int {
        let left = Array(lhs)
        let right = Array(rhs)
        if left.isEmpty { return right.count }
        if right.isEmpty { return left.count }
        var previous = Array(0...right.count)
        for (leftIndex, leftCharacter) in left.enumerated() {
            var current = [leftIndex + 1]
            for (rightIndex, rightCharacter) in right.enumerated() {
                current.append(min(
                    current[rightIndex] + 1,
                    previous[rightIndex + 1] + 1,
                    previous[rightIndex] + (leftCharacter == rightCharacter ? 0 : 1)
                ))
            }
            previous = current
        }
        return previous[right.count]
    }
}
