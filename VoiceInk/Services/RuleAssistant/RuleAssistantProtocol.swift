import Foundation

// MARK: - Constants

enum RuleAssistantConstants {
    static let model = "glm-5.3-flash"
    static let endpointURL = URL(string: "https://opencode.ai/zen/go/v1/chat/completions")!
    static let appVersion: String =
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "dev"
    static let userAgent = "voco-rule-assistant/\(appVersion)"
    static let mcpClientName = "voco-rule-assistant"

    /// Draft event type -> Worker write tool. These tools are never callable by the model;
    /// only the App may invoke them after the user confirms a draft card.
    static let writeTools: [String: String] = [
        "correction": "add_auto_apply_correction",
        "contextLockedRule": "add_auto_apply_context_locked_rule",
        "replacementRule": "add_auto_apply_replacement_rule",
        "replacementFamily": "add_auto_apply_replacement_family",
        "tombstone": "tombstone_auto_apply_rule",
        // Multi-event transactions (tombstone + addReplacementFamily) built by the Worker; Mac-reconcilable.
        "moveAliasToFamily": "move_auto_apply_alias_to_family",
        "mergeReplacementFamilies": "merge_auto_apply_replacement_families",
    ]

    /// Event types the Worker's detect_duplicate_control_event does not accept; preview alone gates them.
    static let transactionTypes: Set<String> = ["moveAliasToFamily", "mergeReplacementFamilies"]

    /// Local (on-device) tool: the model may ask for the records around the selected one.
    static let nearbyTool = "load_nearby_records"
    static let nearbyMax = 5
    static let maxDraftsPerAnswer = 8

    static func isMCPWriteTool(_ name: String) -> Bool {
        writeTools.values.contains(name)
    }
}

func nearbyRecordsToolJSON() -> [String: Any] {
    [
        "type": "function",
        "function": [
            "name": RuleAssistantConstants.nearbyTool,
            "description": "Load the Mac Voco transcription records immediately before and after the selected one (same Mac, by timestamp order) when the single selected record is not enough to tell what was said. Returns text fields only, never audio. At most \(RuleAssistantConstants.nearbyMax) before and \(RuleAssistantConstants.nearbyMax) after.",
            "parameters": [
                "type": "object",
                "properties": [
                    "before": ["type": "integer", "description": "Rows before the selected record (default \(RuleAssistantConstants.nearbyMax), max \(RuleAssistantConstants.nearbyMax)."],
                    "after": ["type": "integer", "description": "Rows after the selected record (default \(RuleAssistantConstants.nearbyMax), max \(RuleAssistantConstants.nearbyMax)."],
                ],
                "additionalProperties": false,
            ],
        ],
    ]
}

// MARK: - JSON helpers

enum RAJSON {
    static func parseObject(_ string: String) -> [String: Any]? {
        guard let data = string.data(using: .utf8),
              let value = try? JSONSerialization.jsonObject(with: data)
        else { return nil }
        return value as? [String: Any]
    }

    static func parseArray(_ string: String) -> [Any]? {
        guard let data = string.data(using: .utf8),
              let value = try? JSONSerialization.jsonObject(with: data)
        else { return nil }
        return value as? [Any]
    }

    static func serialize(_ object: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object)
        else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }

    static func serializeArray(_ array: [Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: array)
        else { return "[]" }
        return String(decoding: data, as: UTF8.self)
    }
}

extension Dictionary where Key == String, Value == Any {
    /// nil for missing keys, JSON null, and non-string values.
    func raString(_ key: String) -> String? {
        guard let value = self[key], !(value is NSNull) else { return nil }
        return value as? String
    }

    func raNonBlankString(_ key: String) -> String? {
        raString(key).flatMap { $0.isEmpty ? nil : $0 }
    }

    func raBool(_ key: String) -> Bool {
        guard let value = self[key], !(value is NSNull) else { return false }
        return (value as? NSNumber)?.boolValue ?? false
    }

    func raInt64(_ key: String) -> Int64? {
        guard let value = self[key], !(value is NSNull) else { return nil }
        return (value as? NSNumber)?.int64Value
    }

    func raDict(_ key: String) -> [String: Any]? {
        guard let value = self[key], !(value is NSNull) else { return nil }
        return value as? [String: Any]
    }

    func raArray(_ key: String) -> [Any]? {
        guard let value = self[key], !(value is NSNull) else { return nil }
        return value as? [Any]
    }

    func raDictArray(_ key: String) -> [[String: Any]] {
        raArray(key)?.compactMap { $0 as? [String: Any] } ?? []
    }
}

// MARK: - Context

/// The selected transcription record, trimmed to what the model is allowed to see.
/// Deliberately excludes audio URL, audio bytes, and unrelated history.
struct RuleAssistantContext: Equatable {
    var rowPk: Int64
    var timestampMs: Int64
    var rawTranscript: String?
    var text: String?
    var normalizedTranscript: String?
    var enhancedText: String?
    var selectedCandidate: String?
    var finalPastedText: String?
    var transcriptionModelName: String?
    var autoApplyModelVersion: String?
    var recordId: String?
    var correctionsJSON: String?

    init(
        rowPk: Int64,
        timestampMs: Int64,
        rawTranscript: String? = nil,
        text: String? = nil,
        normalizedTranscript: String? = nil,
        enhancedText: String? = nil,
        selectedCandidate: String? = nil,
        finalPastedText: String? = nil,
        transcriptionModelName: String? = nil,
        autoApplyModelVersion: String? = nil,
        recordId: String? = nil,
        correctionsJSON: String? = nil
    ) {
        self.rowPk = rowPk
        self.timestampMs = timestampMs
        self.rawTranscript = rawTranscript
        self.text = text
        self.normalizedTranscript = normalizedTranscript
        self.enhancedText = enhancedText
        self.selectedCandidate = selectedCandidate
        self.finalPastedText = finalPastedText
        self.transcriptionModelName = transcriptionModelName
        self.autoApplyModelVersion = autoApplyModelVersion
        self.recordId = recordId
        self.correctionsJSON = correctionsJSON
    }

    init(transcription: Transcription, rowPk: Int64) {
        self.init(
            rowPk: rowPk,
            timestampMs: Int64((transcription.timestamp.timeIntervalSince1970 * 1000).rounded()),
            rawTranscript: transcription.rawTranscript,
            text: transcription.text,
            normalizedTranscript: transcription.normalizedTranscript,
            enhancedText: transcription.enhancedText,
            selectedCandidate: transcription.selectedCandidate,
            finalPastedText: transcription.finalPastedText,
            transcriptionModelName: transcription.transcriptionModelName ?? transcription.asrEngineID,
            autoApplyModelVersion: transcription.autoApplyModelVersion,
            recordId: transcription.id.uuidString,
            correctionsJSON: transcription.correctionsJSON
        )
    }

    /// JSON sent to the model. Never includes audio references.
    func toSafeJSON() -> [String: Any] {
        var json: [String: Any] = [
            "source": "voco",
            "rowPk": rowPk,
            "timestampMs": timestampMs,
        ]
        json["rawTranscript"] = rawTranscript ?? NSNull()
        json["text"] = text ?? NSNull()
        json["normalizedTranscript"] = normalizedTranscript ?? NSNull()
        json["enhancedText"] = enhancedText ?? NSNull()
        json["selectedCandidate"] = selectedCandidate ?? NSNull()
        json["finalPastedText"] = finalPastedText ?? NSNull()
        json["transcriptionModelName"] = transcriptionModelName ?? NSNull()
        json["autoApplyModelVersion"] = autoApplyModelVersion ?? NSNull()
        json["recordId"] = recordId ?? NSNull()
        json["correctionRow"] = correctionRow()
        json["corrections"] = RowCorrectionMarkings.parse(correctionsJSON).map { $0.toJSONObject() }
        return json
    }

    func correctionRow() -> [String: Any] {
        var row: [String: Any] = ["platform": "voco", "rowPk": rowPk]
        if let recordId { row["recordId"] = recordId }
        return row
    }

    func sourceNote() -> String {
        "voco:row:\(rowPk)"
    }
}

// MARK: - Questions

/// Where a confirmed candidate should apply. Chosen by the user on the question card, never by
/// the model; only `broad` lifts the App-side gate against replacementRule / replacementFamily.
enum RuleAssistantScope: String, CaseIterable, Equatable {
    case sentence
    case context
    case broad

    /// Wire label understood by the system prompt (Taiwanese Traditional Chinese on purpose).
    var wireLabel: String {
        switch self {
        case .sentence: return "只改這句"
        case .context: return "語境限定"
        case .broad: return "任何語境"
        }
    }
}

struct RuleAssistantQuestionOption: Equatable {
    var id: String
    var label: String
    var detail: String?
    /// Wrong surface and intended text when the option is a correction candidate.
    var surface: String?
    var target: String?

    var isCandidate: Bool { surface != nil && target != nil }
}

/// A structured question the model asks instead of free text. The App renders the options as
/// buttons so the user picks instead of typing; the reply goes back as a user message.
struct RuleAssistantQuestion: Equatable {
    var id: String
    var prompt: String
    var multiSelect: Bool
    var options: [RuleAssistantQuestionOption]

    static let maxOptions = 8
    static let maxPromptChars = 400
    static let maxLabelChars = 200

    /// Accepts {"question": {...}} and a bare {"prompt"/"question": "...", "options": [...]} object.
    /// Fail-closed: anything malformed yields nil and the answer is shown as plain text.
    static func parse(_ json: [String: Any]) -> RuleAssistantQuestion? {
        let body: [String: Any]
        if let nested = json.raDict("question") {
            body = nested
        } else if json["options"] != nil {
            body = json
        } else {
            return nil
        }
        guard let prompt = body.raNonBlankString("prompt") ?? body.raNonBlankString("question"),
              prompt.count <= maxPromptChars,
              let rawOptions = body.raArray("options"), !rawOptions.isEmpty
        else { return nil }
        var options: [RuleAssistantQuestionOption] = []
        var usedIds = Set<String>()
        for raw in rawOptions.prefix(maxOptions) {
            var label: String?
            var detail: String?
            var surface: String?
            var target: String?
            var id: String?
            if let dict = raw as? [String: Any] {
                label = dict.raNonBlankString("label") ?? dict.raNonBlankString("text")
                detail = dict.raNonBlankString("detail")
                surface = dict.raNonBlankString("surface") ?? dict.raNonBlankString("sourceText")
                target = dict.raNonBlankString("target") ?? dict.raNonBlankString("targetText")
                id = dict.raNonBlankString("id")
            } else if let string = raw as? String, !string.trimmingCharacters(in: .whitespaces).isEmpty {
                label = string
            }
            guard let label, label.count <= maxLabelChars else { return nil }
            if (detail?.count ?? 0) > maxLabelChars || (surface?.count ?? 0) > maxLabelChars || (target?.count ?? 0) > maxLabelChars {
                return nil
            }
            // Way-out/placeholder options are plain options even when the model attaches
            // misleading candidate fields. Demote them before candidate de-duplication.
            if let candidateTarget = target {
                let trimmedTarget = candidateTarget.trimmingCharacters(in: .whitespacesAndNewlines)
                let wrappedInParentheses = (trimmedTarget.hasPrefix("（") && trimmedTarget.hasSuffix("）"))
                    || (trimmedTarget.hasPrefix("(") && trimmedTarget.hasSuffix(")"))
                let isPlaceholder = trimmedTarget.contains("請")
                    && (trimmedTarget.contains("說明") || trimmedTarget.contains("補充"))
                if wrappedInParentheses || (surface.map { trimmedTarget == $0 } ?? false) || isPlaceholder {
                    surface = nil
                    target = nil
                }
            }
            // One option per suspected surface: the App asks the scope itself, so a second option with
            // the same surface → target (e.g. one per scope) would only duplicate the scope picker.
            if let surface, let target, options.contains(where: { $0.surface == surface && $0.target == target }) {
                continue
            }
            var resolvedId = id.map { String($0.prefix(32)) } ?? Self.defaultId(options.count)
            var suffix = 0
            while usedIds.contains(resolvedId) {
                suffix += 1
                resolvedId = "\(Self.defaultId(options.count))\(suffix)"
            }
            usedIds.insert(resolvedId)
            options.append(RuleAssistantQuestionOption(id: resolvedId, label: label, detail: detail, surface: surface, target: target))
        }
        guard !options.isEmpty else { return nil }
        let id = body.raNonBlankString("id").map { String($0.prefix(32)) } ?? "q1"
        return RuleAssistantQuestion(id: id, prompt: prompt, multiSelect: body.raBool("multiSelect"), options: options)
    }

    /// The first question object in an answer; later ones are ignored (one question per turn).
    static func parseFirst(_ located: [(json: [String: Any], range: Range<String.Index>)]) -> RuleAssistantQuestion? {
        for (json, _) in located {
            if let question = parse(json) { return question }
        }
        return nil
    }

    private static func defaultId(_ index: Int) -> String {
        let letters = Array("abcdefghijklmnopqrstuvwxyz")
        return index < letters.count ? String(letters[index]) : "o\(index + 1)"
    }
}

// MARK: - Automatic negative guards for broad rules

/// A literal broad rule replaces every occurrence of its source, including inside longer words
/// (資料架 → 資料夾 would turn 資料架構 into 資料夾構). This derives those longer words from the
/// word-frequency lexicon so the App can add them as negative examples before the user confirms:
/// for every split A+B of the source, lexicon words starting with B give A+word; words starting
/// with the whole source count too. Deterministic, no model judgement involved.
enum RuleAssistantGuardSuggester {
    static let minFrequency = 100
    static let maxGuards = 6
    static let lookupLimit = 20

    static func guards(
        for source: String,
        lexicon: (_ prefix: String) -> [(word: String, frequency: Int)]
    ) -> [String] {
        let trimmed = source.trimmingCharacters(in: .whitespacesAndNewlines)
        let characters = Array(trimmed)
        guard characters.count >= 2, characters.allSatisfy(isCJK) else { return [] }
        var scored: [String: Int] = [:]
        for split in 0..<characters.count {
            let head = String(characters[..<split])
            let tail = String(characters[split...])
            for hit in lexicon(tail) where hit.word != tail {
                let candidate = head + hit.word
                guard candidate != trimmed else { continue }
                scored[candidate] = max(scored[candidate] ?? 0, hit.frequency)
            }
        }
        return scored
            .sorted { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }
            .prefix(maxGuards)
            .map { $0.key }
    }

    private static func isCJK(_ character: Character) -> Bool {
        guard let scalar = character.unicodeScalars.first else { return false }
        switch scalar.value {
        case 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF, 0x20000...0x2FA1F:
            return true
        default:
            return false
        }
    }
}

// MARK: - Drafts

struct RuleAssistantExample: Equatable {
    var text: String
    var context: String
    var expectedText: String

    /// Human-facing/legacy shape retained for review and conversation logs.
    func toJSON() -> [String: Any] {
        ["text": text, "context": context, "expected": expectedText]
    }

    /// Exact Worker schema (text, optional context, optional expectedText).
    func toWorkerJSON() -> [String: Any] {
        var json: [String: Any] = ["text": text]
        if !context.isEmpty { json["context"] = context }
        if !expectedText.isEmpty { json["expectedText"] = expectedText }
        return json
    }
}

struct RuleAssistantDraft: Equatable {
    var eventType: String
    var sourceText: String?
    var targetText: String?
    var sourcePattern: String?
    var familyId: String?
    var aliases: [String] = []
    var contextTokensAny: [String] = []
    var contextAliasesAny: [String] = []
    var policyId: String?
    var fromFamilyId: String?
    var toFamilyId: String?
    var reason: String?
    var disposition: String?
    var note: String?
    var positiveExamples: [RuleAssistantExample] = []
    var negativeExamples: [RuleAssistantExample] = []
    var makeAvailableNow = true
    var nonce: String = UUID().uuidString

    var isBroad: Bool { eventType == "replacementRule" || eventType == "replacementFamily" }

    /// Restructures existing families; never something to guess without Jason saying so.
    var isTransaction: Bool { RuleAssistantConstants.transactionTypes.contains(eventType) }

    func writeToolName() -> String? {
        RuleAssistantConstants.writeTools[eventType]
    }

    func isSafeForWrite() -> Bool {
        guard RuleAssistantConstants.writeTools[eventType] != nil else { return false }
        let longFields = [sourceText, targetText, sourcePattern, familyId, policyId, fromFamilyId, toFamilyId, reason, disposition, note]
        if longFields.compactMap({ $0 }).contains(where: { $0.count > 2_000 }) { return false }
        if aliases.count > 32 || aliases.contains(where: { $0.count > 200 }) { return false }
        if contextTokensAny.count > 32 || contextAliasesAny.count > 32 { return false }
        if (contextTokensAny + contextAliasesAny).contains(where: { $0.count > 200 }) { return false }
        if positiveExamples.count > 10 || negativeExamples.count > 10 { return false }
        if (positiveExamples + negativeExamples).contains(where: {
            $0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || $0.text.count > 2_000 || $0.context.count > 2_000 || $0.expectedText.count > 2_000
        }) { return false }

        func nonBlank(_ value: String?) -> Bool {
            guard let value else { return false }
            return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }

        switch eventType {
        case "correction":
            return nonBlank(sourceText) && nonBlank(targetText)
                && sourceText!.trimmingCharacters(in: .whitespacesAndNewlines) != targetText!.trimmingCharacters(in: .whitespacesAndNewlines)
        case "contextLockedRule":
            return nonBlank(sourcePattern) && nonBlank(targetText)
                && (!contextTokensAny.isEmpty || !contextAliasesAny.isEmpty)
        case "replacementRule":
            return nonBlank(sourcePattern) && nonBlank(targetText)
                && sourcePattern!.trimmingCharacters(in: .whitespacesAndNewlines) != targetText!.trimmingCharacters(in: .whitespacesAndNewlines)
        case "replacementFamily":
            return nonBlank(familyId) && nonBlank(targetText) && !aliases.isEmpty
                && !aliases.contains(where: {
                    $0.trimmingCharacters(in: .whitespacesAndNewlines) == targetText!.trimmingCharacters(in: .whitespacesAndNewlines)
                })
        case "tombstone":
            return (nonBlank(policyId) || (nonBlank(sourcePattern) && nonBlank(targetText)))
                && nonBlank(reason) && (disposition == "blocked" || disposition == "replaced")
        case "moveAliasToFamily":
            return nonBlank(toFamilyId) && (nonBlank(policyId) || nonBlank(sourcePattern))
                && toFamilyId!.trimmingCharacters(in: .whitespacesAndNewlines) != (fromFamilyId ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        case "mergeReplacementFamilies":
            return nonBlank(fromFamilyId) && nonBlank(toFamilyId)
                && fromFamilyId!.trimmingCharacters(in: .whitespacesAndNewlines) != toFamilyId!.trimmingCharacters(in: .whitespacesAndNewlines)
        default:
            return false
        }
    }

    /// Arguments for the Worker write tool, trimmed to that tool's exact input schema.
    func toMcpArguments(context: RuleAssistantContext) -> [String: Any] {
        var args = eventArguments(context: context)
        args["makeAvailableNow"] = makeAvailableNow
        return args
    }

    /// Same event as toMcpArguments, shaped for preview/duplicate tools (no publish flag).
    func toPreviewArguments(context: RuleAssistantContext) -> [String: Any] {
        var args = eventArguments(context: context)
        args["eventType"] = eventType
        return args
    }

    private func eventArguments(context: RuleAssistantContext) -> [String: Any] {
        var args: [String: Any] = [
            "actor": "voco-rule-assistant",
            "correctionSource": "voco",
            "correctionRow": context.correctionRow(),
        ]

        func putExamples() {
            if !positiveExamples.isEmpty { args["positiveExamples"] = positiveExamples.map { $0.toWorkerJSON() } }
            if !negativeExamples.isEmpty { args["negativeExamples"] = negativeExamples.map { $0.toWorkerJSON() } }
        }
        func putNote() {
            args["note"] = sourceNote(context: context)
        }

        switch eventType {
        case "correction":
            args["sourceText"] = sourceText
            args["targetText"] = targetText
            putNote()
        case "contextLockedRule":
            args["sourcePattern"] = sourcePattern
            args["targetText"] = targetText
            if let sourceText { args["sourceText"] = sourceText }
            if !contextTokensAny.isEmpty { args["contextTokensAny"] = contextTokensAny }
            if !contextAliasesAny.isEmpty { args["contextAliasesAny"] = contextAliasesAny }
            putExamples()
            putNote()
        case "replacementRule":
            args["sourcePattern"] = sourcePattern
            args["targetText"] = targetText
            if let sourceText { args["sourceText"] = sourceText }
            if let familyId { args["familyId"] = familyId }
            putExamples()
            putNote()
        case "replacementFamily":
            args["familyId"] = familyId
            args["targetText"] = targetText
            args["aliases"] = aliases
            putExamples()
            putNote()
        case "tombstone":
            if let policyId { args["policyId"] = policyId }
            if let sourcePattern { args["sourcePattern"] = sourcePattern }
            if let targetText { args["targetText"] = targetText }
            args["reason"] = reason
            args["disposition"] = disposition
        case "moveAliasToFamily":
            if let policyId { args["policyId"] = policyId }
            if let sourcePattern { args["sourcePattern"] = sourcePattern }
            if let fromFamilyId { args["fromFamilyId"] = fromFamilyId }
            args["toFamilyId"] = toFamilyId
            if let targetText { args["targetText"] = targetText }
            if let reason { args["reason"] = reason }
            putExamples()
            putNote()
        case "mergeReplacementFamilies":
            args["fromFamilyId"] = fromFamilyId
            args["toFamilyId"] = toFamilyId
            if let targetText { args["targetText"] = targetText }
            if let reason { args["reason"] = reason }
            putExamples()
            putNote()
        default:
            break
        }
        return args
    }

    private func sourceNote(context: RuleAssistantContext) -> String {
        let trimmed = note?.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = [trimmed.flatMap { $0.isEmpty ? nil : $0 }, "source=\(context.sourceNote())"].compactMap { $0 }
        return parts.joined(separator: "; ")
    }

    // MARK: Model-facing draft parsing

    /// Accepts the documented draft JSON, or an object wrapping it under "draft".
    static func parse(_ json: [String: Any]) -> RuleAssistantDraft? {
        var draft = json
        if json.keys.contains("draft") {
            guard let wrapped = json.raDict("draft") else { return nil }
            draft = wrapped
        }
        guard let type = draft.raNonBlankString("eventType") else { return nil }

        func text(_ key: String) -> String? {
            draft.raNonBlankString(key)
        }
        func strings(_ key: String) -> [String] {
            draft.raArray(key)?.compactMap { ($0 as? String).flatMap { $0.isEmpty ? nil : $0 } } ?? []
        }
        func examples(_ key: String) -> [RuleAssistantExample] {
            draft.raDictArray(key).compactMap { entry in
                func field(_ keys: String...) -> String {
                    for key in keys {
                        if let value = entry.raNonBlankString(key) { return value }
                    }
                    return ""
                }
                let text = field("text", "input", "source")
                if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return nil }
                return RuleAssistantExample(
                    text: text,
                    context: field("context"),
                    expectedText: field("expectedText", "expected", "target")
                )
            }
        }

        return RuleAssistantDraft(
            eventType: type,
            sourceText: text("sourceText"),
            targetText: text("targetText"),
            sourcePattern: text("sourcePattern"),
            familyId: text("familyId"),
            aliases: strings("aliases"),
            contextTokensAny: strings("contextTokensAny"),
            contextAliasesAny: strings("contextAliasesAny"),
            policyId: text("policyId"),
            fromFamilyId: text("fromFamilyId"),
            toFamilyId: text("toFamilyId"),
            reason: text("reason"),
            disposition: text("disposition"),
            note: text("note"),
            positiveExamples: examples("positiveExamples"),
            negativeExamples: examples("negativeExamples")
        )
    }

    /// Finds the first balanced JSON object in an answer; returns it and its character range.
    static func locateJSON(in answer: String) -> (json: [String: Any], range: Range<String.Index>)? {
        locateJSON(in: answer, startFrom: answer.startIndex)
    }

    /// Every top-level JSON object in the answer, in order and non-overlapping. The model emits one draft
    /// per plan line; a wrapper {"drafts":[...]} is expanded so each element becomes its own draft.
    static func locateAllJSON(in answer: String, max: Int = RuleAssistantConstants.maxDraftsPerAnswer) -> [(json: [String: Any], range: Range<String.Index>)] {
        var found: [(json: [String: Any], range: Range<String.Index>)] = []
        var from = answer.startIndex
        while found.count < max {
            guard let next = locateJSON(in: answer, startFrom: from) else { break }
            found.append(next)
            from = next.range.upperBound
        }
        return found
    }

    /// Parses every draft in the answer: plain objects, {"draft":{...}} wrappers and {"drafts":[...]} lists.
    static func parseAll(_ located: [(json: [String: Any], range: Range<String.Index>)], max: Int = RuleAssistantConstants.maxDraftsPerAnswer) -> [RuleAssistantDraft] {
        var drafts: [RuleAssistantDraft] = []
        for (json, _) in located {
            if let list = json.raArray("drafts") {
                for case let element as [String: Any] in list {
                    if let draft = parse(element) { drafts.append(draft) }
                }
            } else if let draft = parse(json) {
                drafts.append(draft)
            }
            if drafts.count >= max { break }
        }
        return Array(drafts.prefix(max))
    }

    private static func locateJSON(in answer: String, startFrom: String.Index) -> (json: [String: Any], range: Range<String.Index>)? {
        var searchFrom = startFrom
        while true {
            guard let start = answer[searchFrom...].firstIndex(of: "{") else { return nil }
            var depth = 0
            var end: String.Index?
            var quoted = false
            var escaped = false
            var index = start
            scan: while index < answer.endIndex {
                let character = answer[index]
                if quoted {
                    if escaped { escaped = false } else if character == "\\" { escaped = true } else if character == "\"" { quoted = false }
                    index = answer.index(after: index)
                    continue
                }
                if character == "\"" {
                    quoted = true
                } else if character == "{" {
                    depth += 1
                } else if character == "}" {
                    depth -= 1
                    if depth == 0 {
                        end = answer.index(after: index)
                        break scan
                    }
                }
                index = answer.index(after: index)
            }
            guard let end else { return nil }
            let candidate = String(answer[start..<end])
            if let json = RAJSON.parseObject(candidate) {
                return (json, start..<end)
            }
            searchFrom = answer.index(after: start)
        }
    }
}

// MARK: - SSE parsing

struct SseEvent: Equatable {
    var event: String?
    var data: String
    var id: String?
}

enum SseParserError: Error, Equatable {
    case limitExceeded(String)

    var message: String {
        switch self {
        case .limitExceeded(let message): return message
        }
    }
}

/// Incremental SSE parser. Data lines are joined with a newline as required by SSE.
final class SseEventParser {
    private let maxEventChars = 256_000
    // Bytes, not Characters: CR+LF is a single grapheme cluster in Swift, so Character-level
    // newline search never matches inside CRLF line endings. Array, not Data, so indices stay
    // zero-based across removeFirst.
    private var lineBuffer: [UInt8] = []
    private var data = ""
    private var event: String?
    private var id: String?
    private let onEvent: (SseEvent) -> Void

    init(onEvent: @escaping (SseEvent) -> Void) {
        self.onEvent = onEvent
    }

    func feed(_ chunk: String) throws {
        lineBuffer.append(contentsOf: chunk.utf8)
        guard lineBuffer.count <= maxEventChars else {
            throw SseParserError.limitExceeded("SSE line exceeded safety limit")
        }
        while true {
            guard let newlineIndex = lineBuffer.firstIndex(of: 0x0A) else { break }
            var line = String(decoding: lineBuffer[..<newlineIndex], as: UTF8.self)
            lineBuffer.removeFirst(newlineIndex + 1)
            if line.hasSuffix("\r") { line.removeLast() }
            try consumeLine(line)
        }
    }

    func finish() throws {
        if !lineBuffer.isEmpty {
            var tail = String(decoding: lineBuffer, as: UTF8.self)
            lineBuffer.removeAll()
            if tail.hasSuffix("\r") { tail.removeLast() }
            try consumeLine(tail)
        }
        if !data.isEmpty || event != nil || id != nil {
            dispatch()
        }
    }

    private func consumeLine(_ line: String) throws {
        if line.isEmpty {
            if !data.isEmpty || event != nil || id != nil {
                dispatch()
            }
            return
        }
        if line.hasPrefix(":") { return }
        let field: String
        let value: String
        if let colonIndex = line.firstIndex(of: ":") {
            field = String(line[line.startIndex..<colonIndex])
            var rest = String(line[line.index(after: colonIndex)...])
            if rest.hasPrefix(" ") { rest.removeFirst() }
            value = rest
        } else {
            field = line
            value = ""
        }
        switch field {
        case "event":
            event = value
        case "data":
            if !data.isEmpty { data.append("\n") }
            data.append(value)
            guard data.count <= maxEventChars else {
                throw SseParserError.limitExceeded("SSE event exceeded safety limit")
            }
        case "id":
            id = value
        default:
            break
        }
    }

    private func dispatch() {
        onEvent(SseEvent(event: event, data: data, id: id))
        data = ""
        event = nil
        id = nil
    }
}

// MARK: - Provider delta parsing

struct ToolCallFragment: Equatable {
    var index: Int
    var id: String?
    var name: String?
    var arguments: String?
}

struct OpenCodeDelta: Equatable {
    /// Thinking for the UI: provider reasoning plus any <think> text extracted from content.
    var reasoning: String = ""
    /// Visible answer for the UI: content with <think> blocks removed.
    var content: String = ""
    var toolCalls: [ToolCallFragment] = []
    var done = false
    var error: String?
    var finishReason: String?
    /// Exact provider reasoning_content/reasoning bytes, retained for the next assistant tool round-trip.
    var rawReasoningContent: String = ""
    /// Exact provider content bytes (including any <think> markup), retained for the wire history.
    var rawContent: String = ""
}

/// Parses OpenAI-compatible SSE deltas and keeps thinking separate from visible answer text.
final class OpenCodeDeltaParser {
    private var thinkMode = false
    private var carry = ""
    private let onDelta: (OpenCodeDelta) -> Void

    init(onDelta: @escaping (OpenCodeDelta) -> Void) {
        self.onDelta = onDelta
    }

    func accept(_ event: SseEvent) {
        if event.data == "[DONE]" {
            onDelta(OpenCodeDelta(done: true))
            return
        }
        guard let json = RAJSON.parseObject(event.data) else {
            onDelta(OpenCodeDelta(error: "Invalid provider event"))
            return
        }
        if let errorObject = json.raDict("error") {
            let message = errorObject.raString("message")
            let trimmed = String((message ?? "").prefix(300))
            onDelta(OpenCodeDelta(error: trimmed.isEmpty ? "Provider returned an error" : trimmed))
            return
        }
        if let errorString = json["error"] as? String {
            let trimmed = String(errorString.prefix(300))
            onDelta(OpenCodeDelta(error: trimmed.isEmpty ? "Provider returned an error" : trimmed))
            return
        }
        guard let choices = json.raArray("choices"), let choice = choices.first as? [String: Any] else {
            return
        }
        let delta = choice.raDict("delta") ?? [:]
        let rawReasoning: String
        if delta.keys.contains("reasoning_content"), !(delta["reasoning_content"] is NSNull) {
            rawReasoning = delta.raString("reasoning_content") ?? ""
        } else if delta.keys.contains("reasoning"), !(delta["reasoning"] is NSNull) {
            rawReasoning = delta.raString("reasoning") ?? ""
        } else {
            rawReasoning = ""
        }
        let rawContent: String
        if delta.keys.contains("content"), !(delta["content"] is NSNull) {
            rawContent = delta.raString("content") ?? ""
        } else {
            rawContent = ""
        }
        let split = splitThink(rawContent)
        var calls: [ToolCallFragment] = []
        for (position, item) in delta.raDictArray("tool_calls").enumerated() {
            let function = item.raDict("function")
            calls.append(ToolCallFragment(
                index: item.raInt64("index").map { Int($0) } ?? position,
                id: fragmentString(item, "id"),
                name: function.flatMap { fragmentString($0, "name") },
                arguments: function.flatMap { fragmentString($0, "arguments") }
            ))
        }
        let finishReason: String? = {
            guard choice.keys.contains("finish_reason"), !(choice["finish_reason"] is NSNull) else { return nil }
            return choice.raNonBlankString("finish_reason")
        }()
        onDelta(OpenCodeDelta(
            reasoning: rawReasoning + split.thinking,
            content: split.visible,
            toolCalls: calls,
            done: finishReason != nil,
            finishReason: finishReason,
            rawReasoningContent: rawReasoning,
            rawContent: rawContent
        ))
    }

    /// Flushes a trailing partial "<think>" prefix that never completed into a tag.
    func finish() {
        if carry.isEmpty { return }
        let tail = carry
        carry = ""
        // The raw bytes were already reported in rawContent when they arrived; only the UI split is late.
        onDelta(thinkMode ? OpenCodeDelta(reasoning: tail) : OpenCodeDelta(content: tail))
    }

    private func fragmentString(_ json: [String: Any], _ key: String) -> String? {
        guard json.keys.contains(key), !(json[key] is NSNull) else { return nil }
        return json.raString(key).flatMap { $0.isEmpty ? nil : $0 }
    }

    private func splitThink(_ raw: String) -> (thinking: String, visible: String) {
        if raw.isEmpty && carry.isEmpty { return ("", "") }
        var input = carry + raw
        carry = ""
        var thinking = ""
        var visible = ""
        while !input.isEmpty {
            if !thinkMode {
                if let openRange = input.range(of: "<think>") {
                    visible.append(String(input[input.startIndex..<openRange.lowerBound]))
                    input = String(input[openRange.upperBound...])
                    thinkMode = true
                } else {
                    let keep = tagPrefixLength(input, tag: "<think>")
                    if keep > 0 {
                        visible.append(String(input.dropLast(keep)))
                        carry = String(input.suffix(keep))
                    } else {
                        visible.append(input)
                    }
                    break
                }
            } else {
                if let closeRange = input.range(of: "</think>") {
                    thinking.append(String(input[input.startIndex..<closeRange.lowerBound]))
                    input = String(input[closeRange.upperBound...])
                    thinkMode = false
                } else {
                    let keep = tagPrefixLength(input, tag: "</think>")
                    if keep > 0 {
                        thinking.append(String(input.dropLast(keep)))
                        carry = String(input.suffix(keep))
                    } else {
                        thinking.append(input)
                    }
                    break
                }
            }
        }
        return (thinking, visible)
    }

    private func tagPrefixLength(_ value: String, tag: String) -> Int {
        let max = Swift.min(value.count, tag.count - 1)
        if max <= 0 { return 0 }
        for size in stride(from: max, through: 1, by: -1) {
            if tag.hasPrefix(String(value.suffix(size))) {
                return size
            }
        }
        return 0
    }
}

// MARK: - Tool call accumulation

/// Accumulates streamed tool-call fragments by index. A round is only executable when every call is complete.
final class ToolCallAccumulator {
    private struct Partial {
        var id = ""
        var name = ""
        var args = ""
        var oversized = false
    }

    private var parts: [Int: Partial] = [:]
    private var insertionOrder: [Int] = []
    private var tooMany = false

    static let maxArgumentChars = 32_000
    static let maxIdNameChars = 512
    static let maxToolCalls = 16

    func add(_ fragment: ToolCallFragment) {
        if parts[fragment.index] == nil {
            if parts.count >= Self.maxToolCalls {
                tooMany = true
                return
            }
            parts[fragment.index] = Partial()
            insertionOrder.append(fragment.index)
        }
        var part = parts[fragment.index]!
        if let id = fragment.id {
            if part.id.count + id.count <= Self.maxIdNameChars { part.id.append(id) } else { part.oversized = true }
        }
        if let name = fragment.name {
            if part.name.count + name.count <= Self.maxIdNameChars { part.name.append(name) } else { part.oversized = true }
        }
        if let arguments = fragment.arguments {
            if part.args.count + arguments.count > Self.maxArgumentChars { part.oversized = true } else { part.args.append(arguments) }
        }
        parts[fragment.index] = part
    }

    func isEmpty() -> Bool {
        parts.isEmpty && !tooMany
    }

    /// Reasons this round must not execute; empty when every call is complete and well-formed.
    func problems() -> [String] {
        var problems: [String] = []
        if tooMany {
            problems.append(String(localized: "More than \(Self.maxToolCalls) tool calls"))
        }
        for index in parts.keys.sorted() {
            let part = parts[index]!
            if part.oversized {
                problems.append(String(localized: "Tool call #\(index) exceeded the length limit"))
            } else if part.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                problems.append(String(localized: "Tool call #\(index) is missing an id"))
            } else if part.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                problems.append(String(localized: "Tool call #\(index) is missing a name"))
            } else if RAJSON.parseObject(part.args.isEmpty ? "{}" : part.args) == nil {
                problems.append(String(localized: "Tool call #\(index) (\(part.name)) has incomplete argument JSON"))
            }
        }
        return problems
    }

    /// Returns every call in index order, or throws when any call is incomplete. Partial rounds never execute.
    func complete() throws -> [[String: Any]] {
        let problems = problems()
        guard problems.isEmpty else {
            throw RuleAssistantFailure(problems.joined(separator: "; "))
        }
        return parts.keys.sorted().map { index in
            let part = parts[index]!
            let argsSource = part.args.isEmpty ? "{}" : part.args
            let args = RAJSON.parseObject(argsSource) ?? [:]
            return [
                "id": part.id,
                "type": "function",
                "function": [
                    "name": part.name,
                    "arguments": RAJSON.serialize(args),
                ] as [String: Any],
            ] as [String: Any]
        }
    }
}

// MARK: - Provider wire message

struct OpenCodeMessage: Equatable {
    var role: String
    var content: String?
    var reasoningContent: String?
    var toolCalls: [[String: Any]]
    var toolCallId: String?

    init(
        role: String,
        content: String? = nil,
        reasoningContent: String? = nil,
        toolCalls: [[String: Any]] = [],
        toolCallId: String? = nil
    ) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
    }

    static func == (lhs: OpenCodeMessage, rhs: OpenCodeMessage) -> Bool {
        lhs.role == rhs.role
            && lhs.content == rhs.content
            && lhs.reasoningContent == rhs.reasoningContent
            && lhs.toolCallId == rhs.toolCallId
            && NSDictionary(dictionary: lhs.toJSON()).isEqual(NSDictionary(dictionary: rhs.toJSON()))
    }

    func toJSON() -> [String: Any] {
        var json: [String: Any] = ["role": role]
        // Assistant tool rounds keep an explicit content field so the provider sees the exact wire shape it produced.
        if let content {
            json["content"] = content
        } else if role == "assistant" {
            json["content"] = ""
        }
        if let reasoningContent, !reasoningContent.isEmpty {
            json["reasoning_content"] = reasoningContent
        }
        if !toolCalls.isEmpty {
            json["tool_calls"] = toolCalls
        }
        if let toolCallId {
            json["tool_call_id"] = toolCallId
        }
        return json
    }
}

// MARK: - Errors

/// A user-facing flow failure (its message is safe to show).
struct RuleAssistantFailure: Error, Equatable, LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}
