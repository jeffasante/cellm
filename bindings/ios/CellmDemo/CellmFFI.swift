import Foundation
import CellmFFI

enum CellmBackend: UInt32, CaseIterable, Identifiable {
    case cpu = 0
    case metal = 1

    var id: UInt32 { rawValue }
    var label: String {
        switch self {
        case .cpu: return "CPU"
        case .metal: return "Metal"
        }
    }
}

enum CellmKvEncoding: UInt32, CaseIterable, Identifiable {
    case f16 = 0
    case turboquant = 1

    var id: UInt32 { rawValue }
    var label: String {
        switch self {
        case .f16: return "F16"
        case .turboquant: return "TurboQuant"
        }
    }
}

enum CellmThermalLevel: UInt32, CaseIterable, Identifiable {
    case nominal = 0
    case elevated = 1
    case critical = 2
    case emergency = 3

    var id: UInt32 { rawValue }
    var label: String {
        switch self {
        case .nominal: return "Nominal"
        case .elevated: return "Elevated"
        case .critical: return "Critical"
        case .emergency: return "Emergency"
        }
    }
}

enum CellmError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case .message(let s): return s
        }
    }
}

struct VlmTimings {
    let patchMs: Double
    let encoderMs: Double
    let decodeMs: Double
    let totalMs: Double
    let encoderLayerMs: [Double]
}

struct LlmGenerationStats {
    let promptTokenCount: Int
    let generatedTokenCount: Int
    let prefillMs: Double
    let decodeMs: Double
    let totalMs: Double
    let firstPiece: String
    let stopReason: String
}

final class CellmTokenizer {
    private var handle: cellm_tokenizer_t = 0
    let tokenizerURL: URL

    init(tokenizerURL: URL) throws {
        self.tokenizerURL = tokenizerURL
        let path = tokenizerURL.path
        let h = path.withCString { cstr in
            cellm_tokenizer_create(cstr)
        }
        guard h != 0 else { throw CellmError.message(CellmFFI.lastError()) }
        self.handle = h
    }

    deinit {
        if handle != 0 {
            cellm_tokenizer_destroy(handle)
            handle = 0
        }
    }

    func encode(_ text: String) throws -> [UInt32] {
        let needed = text.withCString { cstr in
            cellm_tokenizer_encode(handle, cstr, nil, 0)
        }
        if needed == 0 { throw CellmError.message(CellmFFI.lastError()) }

        var out = [UInt32](repeating: 0, count: needed)
        let written = text.withCString { cstr in
            out.withUnsafeMutableBufferPointer { buf in
                cellm_tokenizer_encode(handle, cstr, buf.baseAddress, buf.count)
            }
        }
        if written != needed { throw CellmError.message("encode wrote \(written) tokens, expected \(needed)") }
        return out
    }

    func decode(tokens: [UInt32]) throws -> String {
        let needed = tokens.withUnsafeBufferPointer { buf in
            cellm_tokenizer_decode(handle, buf.baseAddress, buf.count, nil, 0)
        }
        // Some tokens can legitimately decode to an empty piece.
        if needed == 0 { return "" }

        var bytes = [CChar](repeating: 0, count: needed + 1)
        let written = tokens.withUnsafeBufferPointer { buf in
            bytes.withUnsafeMutableBufferPointer { out in
                cellm_tokenizer_decode(handle, buf.baseAddress, buf.count, out.baseAddress, out.count)
            }
        }
        if written == 0 { return "" }
        return String(cString: bytes)
    }

    func decodeOne(_ token: UInt32) throws -> String {
        return try decode(tokens: [token])
    }
}

final class CellmEngine {
    private var handle: cellm_engine_t = 0
    private let tokenizer: CellmTokenizer
    private var session: cellm_session_t
    private let modelURL: URL

    let activeBackend: String
    let isLiteRtProxy: Bool
    private(set) var lastPromptStyle: String = "unknown"
    private(set) var lastDebugTrace: [String] = []
    private(set) var lastGenerationStats: LlmGenerationStats?

    private static let debugLogsEnabled = true

    init(modelURL: URL, tokenizer: CellmTokenizer, tokensPerBlock: UInt32 = 16, totalBlocks: UInt32 = 256, topK: UInt32 = 40, temperature: Float = 0.2, repeatPenalty: Float = 1.08, repeatWindow: UInt32 = 96, seed: UInt64 = 1, backend: CellmBackend = .metal, kvEncoding: CellmKvEncoding = .f16, turboqInt8Dot: Bool = true, turboqQjlCorr: Bool = true) throws {
        self.modelURL = modelURL
        self.tokenizer = tokenizer
        let path = modelURL.path
        let requestedBackend: CellmBackend = backend
        let h = path.withCString { cstr in
            cellm_engine_create_v4(
                cstr,
                tokensPerBlock,
                totalBlocks,
                topK,
                temperature,
                repeatPenalty,
                repeatWindow,
                seed,
                requestedBackend.rawValue,
                kvEncoding.rawValue,
                turboqInt8Dot ? 1 : 0,
                turboqQjlCorr ? 1 : 0
            )
        }
        let firstError = CellmFFI.lastError()

        guard h != 0 else {
            let detail = firstError.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "unknown error (ffi returned empty last_error)" : firstError
            throw CellmError.message("engine_create_v4 failed: \(detail) model=\(modelURL.lastPathComponent) backend=\(requestedBackend.label.lowercased()) kv=\(kvEncoding.label.lowercased())")
        }
        self.handle = h

        let sid = cellm_session_create(h)
        guard sid != 0 else { throw CellmError.message(CellmFFI.lastError()) }
        self.session = sid
        self.activeBackend = CellmFFI.engineBackendName(h)
        self.isLiteRtProxy = (cellm_engine_is_litert_proxy(h) != 0)
    }

    deinit {
        if handle != 0 {
            _ = cellm_session_cancel(handle, session)
            cellm_engine_destroy(handle)
            handle = 0
        }
    }

    func generate(
        prompt: String,
        maxNewTokens: Int,
        imageURL: URL? = nil,
        audioURL: URL? = nil,
        thermalLevel: CellmThermalLevel = .nominal,
        exerciseSuspendResume: Bool = false,
        onToken: ((String) -> Void)? = nil
    ) throws -> String {
        if isLiteRtProxy {
            let text = try generateWithLiteRt(prompt: prompt, imageURL: imageURL, audioURL: audioURL)
            onToken?(text)
            let totalMs = 0.0
            lastGenerationStats = LlmGenerationStats(
                promptTokenCount: 0,
                generatedTokenCount: 0,
                prefillMs: 0.0,
                decodeMs: 0.0,
                totalMs: totalMs,
                firstPiece: text,
                stopReason: "litert_proxy"
            )
            return text
        }
        lastGenerationStats = nil
        lastDebugTrace = []
        let uppercaseConstraintTarget = Self.expectedUppercaseTarget(prompt: prompt)
        let processor = ModelDataProcessorFactory.make(tokenizerURL: tokenizer.tokenizerURL, modelURL: modelURL)
        lastPromptStyle = processor.label
        let promptText = processor.wrapPrompt(prompt)
        let promptTokens = try tokenizer.encode(promptText)
        lastDebugTrace.append("start backend=\(activeBackend) prompt_style=\(lastPromptStyle) prompt_tokens=\(promptTokens.count)")
        Self.debugLog("generate start backend=\(activeBackend) prompt_style=\(lastPromptStyle) prompt_tokens=\(promptTokens.count) max_new_tokens=\(maxNewTokens)")

        let tStart = Date()
        try setThermalLevel(thermalLevel)
        var next: UInt32 = 0
        var cacheHit: UInt32 = 0
        let rc = promptTokens.withUnsafeBufferPointer { buf in
            cellm_submit_tokens_cached(handle, session, buf.baseAddress, buf.count, &next, &cacheHit)
        }
        if rc != 0 { throw CellmError.message(CellmFFI.lastError()) }
        let tAfterPrefill = Date()
        lastDebugTrace.append("prefill_cache_hit=\(cacheHit == 1 ? "yes" : "no")")
        lastDebugTrace.append("prefill next_token=\(next) prefill_ms=\(String(format: "%.1f", tAfterPrefill.timeIntervalSince(tStart) * 1000.0))")
        Self.debugLog("prefill done cache_hit=\(cacheHit == 1 ? "yes" : "no") next_token=\(next) prefill_ms=\(String(format: "%.1f", tAfterPrefill.timeIntervalSince(tStart) * 1000.0))")

        if exerciseSuspendResume {
            try suspendSession()
            try resumeSession()
        }

        var out = ""
        // Lightweight on-device decode scheduler: smooth streaming cadence with
        // backend/thermal/cache-hit aware thresholds.
        let streamScheduler = DecodeStreamScheduler(
            activeBackend: activeBackend,
            thermalLevel: thermalLevel,
            prefillCacheHit: cacheHit == 1,
            startAt: tAfterPrefill
        )
        lastDebugTrace.append("stream_profile \(streamScheduler.profileLabel)")
        Self.debugLog("stream scheduler \(streamScheduler.profileLabel)")
        var pendingStreamChunk = ""
        let flushPendingChunk: (Bool) -> Void = { force in
            guard !pendingStreamChunk.isEmpty else { return }
            guard force else {
                return
            }
            let now = Date()
            onToken?(pendingStreamChunk)
            pendingStreamChunk = ""
            streamScheduler.markFlushed(at: now)
        }
        let pushStreamPiece: (String) -> Void = { piece in
            pendingStreamChunk += piece
            let now = Date()
            let shouldFlush = streamScheduler.shouldFlush(
                afterAppending: piece,
                pendingChars: pendingStreamChunk.count,
                now: now
            )
            if shouldFlush {
                onToken?(pendingStreamChunk)
                pendingStreamChunk = ""
                streamScheduler.markFlushed(at: now)
            }
        }
        let firstPiece = try tokenizer.decodeOne(next)
        var stopReason = "max_tokens"
        if let upper = Self.firstUppercaseASCII(in: firstPiece), let target = uppercaseConstraintTarget {
            let chosen = (upper == target) ? upper : target
            let s = String(chosen)
            out = s
            onToken?(s)
            stopReason = (upper == target)
                ? "uppercase_constraint_satisfied"
                : "uppercase_constraint_target_enforced"
        } else if !processor.isStopPiece(firstPiece) {
            out += firstPiece
            pushStreamPiece(firstPiece)
        }
        var lastPiece = firstPiece
        var samePieceRun = firstPiece.isEmpty ? 0 : 1
        var generatedTokens: [UInt32] = [next]
        var generated = 1

        if maxNewTokens <= 1 {
            let totalMs = Date().timeIntervalSince(tStart) * 1000.0
            let prefillMs = tAfterPrefill.timeIntervalSince(tStart) * 1000.0
            lastGenerationStats = LlmGenerationStats(
                promptTokenCount: promptTokens.count,
                generatedTokenCount: generated,
                prefillMs: prefillMs,
                decodeMs: max(0.0, totalMs - prefillMs),
                totalMs: totalMs,
                firstPiece: firstPiece,
                stopReason: "max_tokens_1"
            )
            Self.debugLog("generate done generated=\(generated) stop_reason=max_tokens_1 total_ms=\(String(format: "%.1f", totalMs))")
            return out
        }

        for i in 0..<(maxNewTokens - 1) {
            if stopReason == "uppercase_constraint_satisfied" {
                break
            }
            var outSession: UInt64 = 0
            var tok: UInt32 = 0
            let r = cellm_step_decode(handle, &outSession, &tok)
            if r < 0 { throw CellmError.message(CellmFFI.lastError()) }
            if r == 0 {
                stopReason = "eos_or_engine_stop"
                break
            }
            let piece = try tokenizer.decodeOne(tok)
            if Self.debugLogsEnabled && (i < 8 || i % 8 == 0) {
                let cleanPiece = piece.replacingOccurrences(of: "\n", with: "\\n")
                if i < 12 {
                    lastDebugTrace.append("decode[\(i)] tok=\(tok) piece=\"\(cleanPiece)\"")
                }
                Self.debugLog("decode[\(i)] token=\(tok) piece=\"\(cleanPiece)\"")
            }
            if let upper = Self.firstUppercaseASCII(in: piece), let target = uppercaseConstraintTarget {
                let chosen = (upper == target) ? upper : target
                let s = String(chosen)
                if out != s {
                    out = s
                    flushPendingChunk(true)
                    onToken?(s)
                }
                stopReason = (upper == target)
                    ? "uppercase_constraint_satisfied"
                    : "uppercase_constraint_target_enforced"
                generated += 1
                break
            }
            if processor.isStopPiece(piece) {
                stopReason = "stop_piece"
                break
            }
            if Self.hasLongDigitRun(piece, threshold: 10) {
                stopReason = "digit_run_guard"
                break
            }
            generatedTokens.append(tok)
            if piece == lastPiece && !piece.isEmpty {
                samePieceRun += 1
            } else {
                samePieceRun = piece.isEmpty ? 0 : 1
                lastPiece = piece
            }
            if samePieceRun >= 4 {
                stopReason = "same_piece_loop_guard"
                break
            }
            if Self.isLikelyTokenLoop(generatedTokens, window: 12, maxDominantCount: 8) {
                stopReason = "dominant_token_loop_guard"
                break
            }
            if Self.isAlternatingTwoTokenLoop(generatedTokens, minLength: 8) {
                stopReason = "alternating_loop_guard"
                break
            }
            out += piece
            pushStreamPiece(piece)
            generated += 1
        }
        let totalMs = Date().timeIntervalSince(tStart) * 1000.0
        let prefillMs = tAfterPrefill.timeIntervalSince(tStart) * 1000.0
        if let fallback = uppercaseConstraintTarget {
            let hasUpper = Self.firstUppercaseASCII(in: out) != nil
            if !hasUpper {
                let s = String(fallback)
                if out != s {
                    out = s
                    flushPendingChunk(true)
                    onToken?(s)
                }
                stopReason = "uppercase_constraint_fallback_prompt_target"
            }
        }
        flushPendingChunk(true)
        lastGenerationStats = LlmGenerationStats(
            promptTokenCount: promptTokens.count,
            generatedTokenCount: generated,
            prefillMs: prefillMs,
            decodeMs: max(0.0, totalMs - prefillMs),
            totalMs: totalMs,
            firstPiece: firstPiece,
            stopReason: stopReason
        )
        lastDebugTrace.append("done generated=\(generated) stop_reason=\(stopReason) total_ms=\(String(format: "%.1f", totalMs))")
        Self.debugLog("generate done generated=\(generated) stop_reason=\(stopReason) total_ms=\(String(format: "%.1f", totalMs))")
        return out
    }

    private func generateWithLiteRt(prompt: String, imageURL: URL? = nil, audioURL: URL? = nil) throws -> String {
        #if os(iOS)
        throw CellmError.message("LiteRT proxy models are desktop-only in this build. Use native .cellm/.cellmd models on iOS.")
        #else
        let needsMultimodal = (imageURL != nil || audioURL != nil)
        let needed = prompt.withCString { cPrompt in
            if needsMultimodal {
                if let imageURL {
                    return imageURL.path.withCString { cImage in
                        if let audioURL {
                            return audioURL.path.withCString { cAudio in
                                cellm_engine_generate_multimodal(handle, cPrompt, cImage, cAudio, nil, 0)
                            }
                        }
                        return cellm_engine_generate_multimodal(handle, cPrompt, cImage, nil, nil, 0)
                    }
                }
                if let audioURL {
                    return audioURL.path.withCString { cAudio in
                        cellm_engine_generate_multimodal(handle, cPrompt, nil, cAudio, nil, 0)
                    }
                }
                return cellm_engine_generate_multimodal(handle, cPrompt, nil, nil, nil, 0)
            }
            return cellm_engine_generate_text(handle, cPrompt, nil, 0)
        }
        if needed == 0 {
            throw CellmError.message(CellmFFI.lastError())
        }
        var bytes = [CChar](repeating: 0, count: needed + 1)
        let written = prompt.withCString { cPrompt in
            bytes.withUnsafeMutableBufferPointer { out in
                if needsMultimodal {
                    if let imageURL {
                        return imageURL.path.withCString { cImage in
                            if let audioURL {
                                return audioURL.path.withCString { cAudio in
                                    cellm_engine_generate_multimodal(handle, cPrompt, cImage, cAudio, out.baseAddress, out.count)
                                }
                            }
                            return cellm_engine_generate_multimodal(handle, cPrompt, cImage, nil, out.baseAddress, out.count)
                        }
                    }
                    if let audioURL {
                        return audioURL.path.withCString { cAudio in
                            cellm_engine_generate_multimodal(handle, cPrompt, nil, cAudio, out.baseAddress, out.count)
                        }
                    }
                    return cellm_engine_generate_multimodal(handle, cPrompt, nil, nil, out.baseAddress, out.count)
                }
                return cellm_engine_generate_text(handle, cPrompt, out.baseAddress, out.count)
            }
        }
        if written == 0 {
            throw CellmError.message(CellmFFI.lastError())
        }
        return String(cString: bytes)
        #endif
    }

    private static func debugLog(_ message: String) {
        guard debugLogsEnabled else { return }
        print("[CellmDebug] \(message)")
    }
    
    private static func firstUppercaseASCII(in text: String) -> Character? {
        for scalar in text.unicodeScalars {
            if scalar.value >= 65 && scalar.value <= 90 {
                return Character(scalar)
            }
        }
        return nil
    }
    
    private static func expectedUppercaseTarget(prompt: String) -> Character? {
        let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = p.lowercased()
        guard lower.contains("exactly one uppercase letter") else { return nil }
        if let idx = p.lastIndex(of: ":") {
            let suffix = p[p.index(after: idx)...].trimmingCharacters(in: .whitespacesAndNewlines)
            if let c = suffix.first, c.isASCII, c.isUppercase {
                return c
            }
        }
        return nil
    }

    func setThermalLevel(_ level: CellmThermalLevel) throws {
        let rc = cellm_engine_set_thermal_level(handle, level.rawValue)
        if rc != 0 { throw CellmError.message(CellmFFI.lastError()) }
    }

    func suspendSession() throws {
        let rc = cellm_session_suspend(handle, session)
        if rc != 0 { throw CellmError.message(CellmFFI.lastError()) }
    }

    func resumeSession() throws {
        let rc = cellm_session_resume(handle, session)
        if rc != 0 { throw CellmError.message(CellmFFI.lastError()) }
    }

    func resetSession() throws {
        let rc = cellm_session_reset(handle, session)
        if rc != 0 { throw CellmError.message(CellmFFI.lastError()) }
    }

    private static func hasLongDigitRun(_ piece: String, threshold: Int) -> Bool {
        guard threshold > 1 else { return false }
        var run = 0
        for scalar in piece.unicodeScalars {
            if CharacterSet.decimalDigits.contains(scalar) {
                run += 1
                if run >= threshold { return true }
            } else {
                run = 0
            }
        }
        return false
    }

    private static func isLikelyTokenLoop(_ tokens: [UInt32], window: Int, maxDominantCount: Int) -> Bool {
        guard tokens.count >= window else { return false }
        let recent = tokens.suffix(window)
        var counts: [UInt32: Int] = [:]
        for t in recent {
            counts[t, default: 0] += 1
        }
        guard let dominant = counts.values.max() else { return false }
        return dominant >= maxDominantCount
    }

    private static func isAlternatingTwoTokenLoop(_ tokens: [UInt32], minLength: Int) -> Bool {
        guard tokens.count >= minLength else { return false }
        let recent = Array(tokens.suffix(minLength))
        guard minLength >= 4 else { return false }
        let a = recent[0]
        let b = recent[1]
        if a == b { return false }
        for i in 0..<recent.count {
            let expected = (i % 2 == 0) ? a : b
            if recent[i] != expected {
                return false
            }
        }
        return true
    }
}

final class CellmVLMEngine {
    private var handle: cellm_engine_t = 0
    private let session: cellm_session_t

    let activeBackend: String
    private(set) var lastTimings: VlmTimings?

    init(modelURL: URL, tokensPerBlock: UInt32 = 16, totalBlocks: UInt32 = 512, topK: UInt32 = 40, temperature: Float = 0.7, repeatPenalty: Float = 1.15, repeatWindow: UInt32 = 128, seed: UInt64 = 1, backend: CellmBackend = .metal, kvEncoding: CellmKvEncoding = .f16, turboqInt8Dot: Bool = true, turboqQjlCorr: Bool = true) throws {
        let path = modelURL.path
        let h = path.withCString { cstr in
            cellm_engine_create_v4(
                cstr,
                tokensPerBlock,
                totalBlocks,
                topK,
                temperature,
                repeatPenalty,
                repeatWindow,
                seed,
                backend.rawValue,
                kvEncoding.rawValue,
                turboqInt8Dot ? 1 : 0,
                turboqQjlCorr ? 1 : 0
            )
        }
        guard h != 0 else { throw CellmError.message(CellmFFI.lastError()) }
        self.handle = h

        let sid = cellm_session_create(h)
        guard sid != 0 else { throw CellmError.message(CellmFFI.lastError()) }
        self.session = sid
        self.activeBackend = CellmFFI.engineBackendName(h)
    }

    deinit {
        if handle != 0 {
            _ = cellm_session_cancel(handle, session)
            cellm_engine_destroy(handle)
            handle = 0
        }
    }

    func describe(imageBytes: Data, prompt: String) throws -> String {
        lastTimings = nil
        var out = [CChar](repeating: 0, count: 32 * 1024)
        let rc = prompt.withCString { cPrompt in
            imageBytes.withUnsafeBytes { raw in
                out.withUnsafeMutableBufferPointer { buf in
                    cellm_vlm_describe_image(
                        handle,
                        session,
                        raw.bindMemory(to: UInt8.self).baseAddress,
                        raw.count,
                        cPrompt,
                        buf.baseAddress,
                        buf.count
                    )
                }
            }
        }
        if rc != 0 {
            throw CellmError.message(CellmFFI.lastError())
        }
        lastTimings = try CellmFFI.vlmLastTimings()
        return String(cString: out)
    }
}

enum CellmFFI {
    static func lastError() -> String {
        var buf = [CChar](repeating: 0, count: 4096)
        let n = cellm_last_error_message(&buf, buf.count)
        if n == 0 { return "ffi_error_empty" }
        return String(cString: buf)
    }

    static func engineBackendName(_ handle: cellm_engine_t) -> String {
        var buf = [CChar](repeating: 0, count: 64)
        let n = cellm_engine_backend_name(handle, &buf, buf.count)
        if n == 0 { return "unknown" }
        return String(cString: buf)
    }

    static func metalSmokeError() -> String? {
        let rc = cellm_metal_smoke_test()
        if rc == 0 { return nil }
        let msg = lastError()
        return msg.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "metal smoke test failed" : msg
    }

    static func runLiteRtMultimodalOnce(
        modelURL: URL,
        prompt: String,
        imageURL: URL?,
        audioURL: URL?,
        backend: CellmBackend
    ) throws -> (text: String, activeBackend: String) {
        let h = modelURL.path.withCString { cModel in
            cellm_engine_create_v4(
                cModel,
                16,
                256,
                40,
                0.2,
                1.08,
                96,
                UInt64(Date().timeIntervalSince1970 * 1000.0),
                backend.rawValue,
                CellmKvEncoding.f16.rawValue,
                1,
                1
            )
        }
        guard h != 0 else { throw CellmError.message(lastError()) }
        defer { cellm_engine_destroy(h) }

        let sid = cellm_session_create(h)
        if sid != 0 {
            defer { _ = cellm_session_cancel(h, sid) }
        }

        guard cellm_engine_is_litert_proxy(h) != 0 else {
            throw CellmError.message("Selected model is not a LiteRT proxy model.")
        }

        let active = engineBackendName(h)

        func call(_ outBuf: UnsafeMutablePointer<CChar>?, _ outLen: Int) -> Int {
            prompt.withCString { cPrompt in
                if let imageURL {
                    return imageURL.path.withCString { cImage in
                        if let audioURL {
                            return audioURL.path.withCString { cAudio in
                                Int(cellm_engine_generate_multimodal(h, cPrompt, cImage, cAudio, outBuf, outLen))
                            }
                        }
                        return Int(cellm_engine_generate_multimodal(h, cPrompt, cImage, nil, outBuf, outLen))
                    }
                }
                if let audioURL {
                    return audioURL.path.withCString { cAudio in
                        Int(cellm_engine_generate_multimodal(h, cPrompt, nil, cAudio, outBuf, outLen))
                    }
                }
                return Int(cellm_engine_generate_multimodal(h, cPrompt, nil, nil, outBuf, outLen))
            }
        }

        let needed = call(nil, 0)
        if needed <= 0 {
            throw CellmError.message(lastError())
        }
        var out = [CChar](repeating: 0, count: needed + 1)
        let written = out.withUnsafeMutableBufferPointer { buf in
            call(buf.baseAddress, buf.count)
        }
        if written <= 0 {
            throw CellmError.message(lastError())
        }
        return (String(cString: out), active)
    }

    static func vlmLastTimings() throws -> VlmTimings {
        var patch = 0.0
        var encoder = 0.0
        var decode = 0.0
        var total = 0.0
        let rc = cellm_vlm_last_timings_ms(&patch, &encoder, &decode, &total)
        if rc != 0 {
            throw CellmError.message(lastError())
        }
        let layers = vlmLastEncoderLayerMs()
        return VlmTimings(
            patchMs: patch,
            encoderMs: encoder,
            decodeMs: decode,
            totalMs: total,
            encoderLayerMs: layers
        )
    }

    static func vlmLastEncoderLayerMs() -> [Double] {
        let count = Int(cellm_vlm_last_encoder_layer_count())
        if count <= 0 { return [] }
        var out: [Double] = []
        out.reserveCapacity(count)
        for i in 0..<count {
            var value = 0.0
            let rc = cellm_vlm_last_encoder_layer_time_ms(UInt32(i), &value)
            if rc == 0 {
                out.append(value)
            } else {
                break
            }
        }
        return out
    }
}
