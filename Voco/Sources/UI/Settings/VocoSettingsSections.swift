import SwiftUI

struct VocoSettingsSections: View {
    var body: some View {
        Group {
            ChinesePostProcessingSettingsView()
            AutoApplyModelSettingsView()
            RuleAssistantSettingsView()
            ASRContextBiasSettingsView()
        }
    }
}
