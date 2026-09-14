import Foundation

struct TriggerSnapshot {
    let appBundleIds: Set<String>
    let websites: Set<String>
    let templateIds: Set<String>

    init(appConfigs: [AppConfig], websiteConfigs: [URLConfig], triggerGroups: [ModeTriggerGroup], cleanURL: (String) -> String) {
        appBundleIds = Set(appConfigs.map(\.bundleIdentifier) + triggerGroups.flatMap { $0.appConfigs.map(\.bundleIdentifier) })
        websites = Set(websiteConfigs.map { cleanURL($0.url) } + triggerGroups.flatMap { $0.urlConfigs.map { cleanURL($0.url) } })
        templateIds = Set(triggerGroups.compactMap(\.templateId))
    }
}

extension ModeTriggerGroup {
    var summaryText: String {
        let appCount = appConfigs.count
        let websiteCount = urlConfigs.count

        switch (appCount, websiteCount) {
        case (0, 0):
            return String(localized: "No triggers")
        case (0, _):
            return countText(websiteCount, single: "1 website", many: "\(websiteCount) websites")
        case (_, 0):
            return countText(appCount, single: "1 app", many: "\(appCount) apps")
        default:
            let apps = countText(appCount, single: "1 app", many: "\(appCount) apps")
            let websites = countText(websiteCount, single: "1 website", many: "\(websiteCount) websites")
            return "\(apps) · \(websites)"
        }
    }

    private func countText(_ count: Int, single: LocalizedStringResource, many: LocalizedStringResource) -> String {
        String(localized: count == 1 ? single : many)
    }
}
