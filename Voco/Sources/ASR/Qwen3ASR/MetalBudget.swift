// MetalBudget.swift
// GPU memory budget utilities for MLX inference stability
// Adapted from ivan-digital/qwen3-asr-swift

import Cmlx
import Foundation
import MLX
import os

/// Metal GPU memory budget utilities.
/// Pins GPU memory to prevent paging under macOS memory pressure,
/// improving MLX inference stability.
enum MetalBudget {

    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "MetalBudget")

    // Cache hardware constants that never change at runtime
    private static let deviceInfo = GPU.deviceInfo()

    /// Total device memory in bytes.
    static var totalMemory: Int { deviceInfo.memorySize }

    /// Maximum recommended working set size in bytes.
    static var maxRecommendedWorkingSet: Int { Int(deviceInfo.maxRecommendedWorkingSetSize) }

    /// Query real Metal headroom: recommended working set minus active allocations.
    /// Returns nil if Metal device info is unavailable.
    static var availableBytes: Int? {
        guard maxRecommendedWorkingSet > 0 else { return nil }
        let overhead = 256 * 1024 * 1024  // 256 MB safety margin
        return max(0, maxRecommendedWorkingSet - Memory.activeMemory - overhead)
    }

    /// Pin GPU memory to prevent paging under pressure.
    /// Uses 90% of recommended working set by default.
    @discardableResult
    static func pinMemory(fraction: Double = 0.9) -> Int {
        let limit = Int(Double(maxRecommendedWorkingSet) * fraction)
        var previous: size_t = 0
        mlx_set_wired_limit(&previous, size_t(limit))
        let limitMB = limit / (1024 * 1024)
        let activeMB = Memory.activeMemory / (1024 * 1024)
        logger.info("GPU memory pinned: limit=\(limitMB)MB, active=\(activeMB)MB, previous=\(Int(previous) / (1024 * 1024))MB")
        return Int(previous)
    }

    /// Unpin GPU memory (set wired limit to 0).
    @discardableResult
    static func unpinMemory() -> Int {
        var previous: size_t = 0
        mlx_set_wired_limit(&previous, 0)
        logger.info("GPU memory unpinned (previous limit: \(Int(previous) / (1024 * 1024))MB)")
        return Int(previous)
    }

    /// Check if a model of the given size (bytes) can fit in available GPU memory.
    static func canFit(modelBytes: Int) -> Bool {
        guard let available = availableBytes else { return true }
        return modelBytes <= available
    }
}
