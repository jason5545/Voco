//
//  WhisperStateUITests.swift
//  VoiceInkTests
//

import XCTest
import SwiftData
@testable import VoiceInk

@MainActor
final class WhisperStateUITests: XCTestCase {

    var modelContainer: ModelContainer!
    var modelContext: ModelContext!
    var whisperState: WhisperState!

    override func setUpWithError() throws {
        // Set up in-memory ModelContainer for testing
        let schema = Schema([Transcription.self])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
        modelContainer = try ModelContainer(for: schema, configurations: [modelConfiguration])
        modelContext = modelContainer.mainContext
        
        whisperState = WhisperState(modelContext: modelContext, enhancementService: nil)
    }

    override func tearDownWithError() throws {
        whisperState = nil
        modelContext = nil
        modelContainer = nil
    }

    func testToggleMiniRecorderDoesNotBlockMainThread() async throws {
        // Arrange: Simulate "Electron" mode where accessibility focused element is unavailable.
        let cache = EditModeCacheService.shared
        // Note: For a true unit test you'd want a mock for EditModeCacheService
        // But here we can inject the required state or assume toggleMiniRecorder returns quickly.
        
        let startTime = Date()
        
        // Act
        // Call toggleMiniRecorder. This should return almost immediately now because the slow
        // SelectedTextService.fetchSelectedText() call is pushed to a background task.
        await whisperState.toggleMiniRecorder()
        
        let duration = Date().timeIntervalSince(startTime)
        
        // Assert
        // We expect toggleMiniRecorder to return very quickly (< 50ms usually, but we set upper bound to 100ms
        // to be safe on slower CI machines).
        // Before the fix, the await menuAction would take 100-200ms.
        XCTAssertLessThan(duration, 0.1, "toggleMiniRecorder should return quickly and not block the main thread waiting for menuAction.")
        
        // Clean up: Wait a bit to let any lingering tasks finish
        try await Task.sleep(nanoseconds: 200_000_000)
    }
}
