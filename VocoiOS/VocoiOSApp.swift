// VocoiOSApp.swift
// iOS host app for Voco keyboard extension
// [AI-Claude: 2026-03-02]

import SwiftUI

@main
struct VocoiOSApp: App {
    var body: some Scene {
        WindowGroup {
            VocoiOSContentView()
        }
    }
}

struct VocoiOSContentView: View {
    var body: some View {
        TabView {
            ModelManagementView()
                .tabItem {
                    Label("Models", systemImage: "cpu")
                }

            SetupGuideView()
                .tabItem {
                    Label("Setup", systemImage: "keyboard")
                }

            LLMSettingsView()
                .tabItem {
                    Label("AI", systemImage: "sparkles")
                }
        }
    }
}
