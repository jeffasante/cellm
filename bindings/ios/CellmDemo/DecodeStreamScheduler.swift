import Foundation

final class DecodeStreamScheduler {
    struct Profile {
        let minChunkChars: Int
        let maxChunkChars: Int
        let softFlushInterval: TimeInterval
        let maxFlushInterval: TimeInterval
        let immediatePieces: Int

        var label: String {
            let soft = String(format: "%.2f", softFlushInterval)
            let hard = String(format: "%.2f", maxFlushInterval)
            return "min=\(minChunkChars),max=\(maxChunkChars),soft=\(soft),max=\(hard),burst=\(immediatePieces)"
        }
    }

    private let profile: Profile
    private var lastFlushAt: Date
    private var remainingImmediatePieces: Int

    init(activeBackend: String, thermalLevel: CellmThermalLevel, prefillCacheHit: Bool, startAt: Date = Date()) {
        self.profile = Self.makeProfile(activeBackend: activeBackend, thermalLevel: thermalLevel, prefillCacheHit: prefillCacheHit)
        self.lastFlushAt = startAt
        self.remainingImmediatePieces = self.profile.immediatePieces
    }

    var profileLabel: String { profile.label }

    func shouldFlush(afterAppending piece: String, pendingChars: Int, now: Date) -> Bool {
        if pendingChars <= 0 { return false }
        if Self.isHardBoundary(piece: piece) { return true }

        if remainingImmediatePieces > 0 {
            remainingImmediatePieces -= 1
            return true
        }

        let elapsed = now.timeIntervalSince(lastFlushAt)
        if pendingChars >= profile.maxChunkChars { return true }
        if elapsed >= profile.maxFlushInterval { return true }
        if pendingChars >= profile.minChunkChars && elapsed >= profile.softFlushInterval { return true }
        return false
    }

    func markFlushed(at now: Date) {
        lastFlushAt = now
    }

    private static func isHardBoundary(piece: String) -> Bool {
        if piece.contains("\n") { return true }
        if piece.contains(".") || piece.contains("!") || piece.contains("?") { return true }
        if piece.hasSuffix(":") || piece.hasSuffix(";") { return true }
        return false
    }

    private static func makeProfile(activeBackend: String, thermalLevel: CellmThermalLevel, prefillCacheHit: Bool) -> Profile {
        let backend = activeBackend.lowercased()
        var minChunkChars = 12
        var maxChunkChars = 32
        var softFlush: TimeInterval = 0.06
        var maxFlush: TimeInterval = 0.16
        var immediate = 2

        if backend == "metal" {
            minChunkChars = 10
            maxChunkChars = 28
            softFlush = 0.05
            maxFlush = 0.14
            immediate = 3
        }

        if prefillCacheHit {
            softFlush *= 0.80
            maxFlush *= 0.80
            immediate += 1
        }

        switch thermalLevel {
        case .nominal:
            break
        case .elevated:
            minChunkChars += 4
            softFlush *= 1.2
        case .critical:
            minChunkChars += 8
            maxChunkChars += 8
            softFlush *= 1.4
            maxFlush *= 1.4
            immediate = max(1, immediate - 1)
        case .emergency:
            minChunkChars += 12
            maxChunkChars += 16
            softFlush *= 1.6
            maxFlush *= 1.6
            immediate = 1
        }

        return Profile(
            minChunkChars: minChunkChars,
            maxChunkChars: maxChunkChars,
            softFlushInterval: softFlush,
            maxFlushInterval: maxFlush,
            immediatePieces: immediate
        )
    }
}
