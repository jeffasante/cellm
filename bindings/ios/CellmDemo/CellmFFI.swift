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

final class CellmTokenizer {
    private var handle: cellm_tokenizer_t = 0

    init(tokenizerURL: URL) throws {
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

    init(modelURL: URL, tokenizer: CellmTokenizer, tokensPerBlock: UInt32 = 16, totalBlocks: UInt32 = 512, topK: UInt32 = 40, temperature: Float = 0.7, repeatPenalty: Float = 1.05, repeatWindow: UInt32 = 64, seed: UInt64 = 1, backend: CellmBackend = .metal) throws {
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
        let promptText = "User: \(prompt)\nAssistant:"
        let promptTokens = try tokenizer.encode(promptText)

        var next: UInt32 = 0
        let rc = promptTokens.withUnsafeBufferPointer { buf in
            cellm_submit_tokens(handle, session, buf.baseAddress, buf.count, &next)
        }
        if rc != 0 { throw CellmError.message(CellmFFI.lastError()) }

        var out = ""
        out += try tokenizer.decodeOne(next)

        if maxNewTokens <= 1 {
            return out
        }

        for _ in 0..<(maxNewTokens - 1) {
            var outSession: UInt64 = 0
            var tok: UInt32 = 0
            let r = cellm_step_decode(handle, &outSession, &tok)
            if r < 0 { throw CellmError.message(CellmFFI.lastError()) }
            if r == 0 { break }
            out += try tokenizer.decodeOne(tok)
        }
        return out
    }
}

final class CellmVLMEngine {
    private var handle: cellm_engine_t = 0
    private let session: cellm_session_t

    let activeBackend: String

    init(modelURL: URL, tokensPerBlock: UInt32 = 16, totalBlocks: UInt32 = 512, topK: UInt32 = 40, temperature: Float = 0.0, repeatPenalty: Float = 1.05, repeatWindow: UInt32 = 64, seed: UInt64 = 1, backend: CellmBackend = .metal) throws {
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
}
