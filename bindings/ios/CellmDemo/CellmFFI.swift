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
        if needed == 0 { throw CellmError.message(CellmFFI.lastError()) }

        var bytes = [CChar](repeating: 0, count: needed + 1)
        let written = tokens.withUnsafeBufferPointer { buf in
            bytes.withUnsafeMutableBufferPointer { out in
                cellm_tokenizer_decode(handle, buf.baseAddress, buf.count, out.baseAddress, out.count)
            }
        }
        if written == 0 { throw CellmError.message(CellmFFI.lastError()) }
        return String(cString: bytes)
    }

    func decodeOne(_ token: UInt32) throws -> String {
        return try decode(tokens: [token])
    }
}

final class CellmEngine {
    private var handle: cellm_engine_t = 0
    private let tokenizer: CellmTokenizer
    private let session: cellm_session_t

    let activeBackend: String
    private(set) var lastGenerationStats: LlmGenerationStats?

    init(modelURL: URL, tokenizer: CellmTokenizer, tokensPerBlock: UInt32 = 16, totalBlocks: UInt32 = 512, topK: UInt32 = 40, temperature: Float = 0.2, repeatPenalty: Float = 1.08, repeatWindow: UInt32 = 96, seed: UInt64 = 1, backend: CellmBackend = .metal) throws {
        self.tokenizer = tokenizer
        let path = modelURL.path
        let h = path.withCString { cstr in
            cellm_engine_create_v3(cstr, tokensPerBlock, totalBlocks, topK, temperature, repeatPenalty, repeatWindow, seed, backend.rawValue)
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

    func generate(prompt: String, maxNewTokens: Int) throws -> String {
        lastGenerationStats = nil
        let promptText = Self.wrapPrompt(prompt, tokenizerURL: tokenizer.tokenizerURL)
        let promptTokens = try tokenizer.encode(promptText)

        let tStart = Date()
        var next: UInt32 = 0
        let rc = promptTokens.withUnsafeBufferPointer { buf in
            cellm_submit_tokens(handle, session, buf.baseAddress, buf.count, &next)
        }
        if rc != 0 { throw CellmError.message(CellmFFI.lastError()) }
        let tAfterPrefill = Date()

        var out = ""
        let firstPiece = try tokenizer.decodeOne(next)
        if !Self.isStopPiece(firstPiece) {
            out += firstPiece
        }
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
                firstPiece: firstPiece
            )
            return out
        }

        for _ in 0..<(maxNewTokens - 1) {
            var outSession: UInt64 = 0
            var tok: UInt32 = 0
            let r = cellm_step_decode(handle, &outSession, &tok)
            if r < 0 { throw CellmError.message(CellmFFI.lastError()) }
            if r == 0 { break }
            let piece = try tokenizer.decodeOne(tok)
            if Self.isStopPiece(piece) { break }
            if Self.hasLongDigitRun(piece, threshold: 10) { break }
            out += piece
            generated += 1
        }
        let totalMs = Date().timeIntervalSince(tStart) * 1000.0
        let prefillMs = tAfterPrefill.timeIntervalSince(tStart) * 1000.0
        lastGenerationStats = LlmGenerationStats(
            promptTokenCount: promptTokens.count,
            generatedTokenCount: generated,
            prefillMs: prefillMs,
            decodeMs: max(0.0, totalMs - prefillMs),
            totalMs: totalMs,
            firstPiece: firstPiece
        )
        return out
    }

    private static func wrapPrompt(_ prompt: String, tokenizerURL: URL) -> String {
        let cleanPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        switch promptStyle(tokenizerURL: tokenizerURL) {
        case .smolChat:
            return "<|im_start|>User: \(cleanPrompt)<end_of_utterance>\nAssistant:"
        case .chatML(let includeThinkPrefill):
            var s = "<|im_start|>user\n\(cleanPrompt)<|im_end|>\n<|im_start|>assistant\n"
            if includeThinkPrefill {
                s += "<think>\n\n</think>\n\n"
            }
            return s
        case .plain:
            return "User: \(cleanPrompt)\nAssistant:"
        }
    }

    private enum PromptStyle {
        case smolChat
        case chatML(includeThinkPrefill: Bool)
        case plain
    }

    private static func promptStyle(tokenizerURL: URL) -> PromptStyle {
        let cfgURL = tokenizerURL.deletingLastPathComponent().appendingPathComponent("tokenizer_config.json")
        guard
            let data = try? Data(contentsOf: cfgURL),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let tpl = obj["chat_template"] as? String
        else {
            return .plain
        }
        if tpl.contains("<end_of_utterance>") && tpl.contains("Assistant:") {
            return .smolChat
        }
        if tpl.contains("<|im_start|>") && tpl.contains("<|im_end|>") {
            let includeThinkPrefill = tpl.contains("<think>") && tpl.contains("</think>")
            return .chatML(includeThinkPrefill: includeThinkPrefill)
        }
        return .plain
    }

    private static func isStopPiece(_ piece: String) -> Bool {
        let p = piece.trimmingCharacters(in: .whitespacesAndNewlines)
        return p == "<|im_end|>" || p == "<end_of_utterance>" || p == "<|endoftext|>"
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
}

final class CellmVLMEngine {
    private var handle: cellm_engine_t = 0
    private let session: cellm_session_t

    let activeBackend: String
    private(set) var lastTimings: VlmTimings?

    init(modelURL: URL, tokensPerBlock: UInt32 = 16, totalBlocks: UInt32 = 512, topK: UInt32 = 40, temperature: Float = 0.7, repeatPenalty: Float = 1.15, repeatWindow: UInt32 = 128, seed: UInt64 = 1, backend: CellmBackend = .metal) throws {
        let path = modelURL.path
        let h = path.withCString { cstr in
            cellm_engine_create_v3(cstr, tokensPerBlock, totalBlocks, topK, temperature, repeatPenalty, repeatWindow, seed, backend.rawValue)
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
        if n == 0 { return "unknown error" }
        return String(cString: buf)
    }

    static func engineBackendName(_ handle: cellm_engine_t) -> String {
        var buf = [CChar](repeating: 0, count: 64)
        let n = cellm_engine_backend_name(handle, &buf, buf.count)
        if n == 0 { return "unknown" }
        return String(cString: buf)
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
