import Foundation
import CellmSDK

enum SmokeError: Error, CustomStringConvertible {
    case message(String)
    var description: String {
        switch self {
        case .message(let m): return m
        }
    }
}

func ffiLastError() -> String {
    var buf = [CChar](repeating: 0, count: 2048)
    _ = cellm_last_error_message(&buf, buf.count)
    let s = String(cString: buf).trimmingCharacters(in: .whitespacesAndNewlines)
    return s.isEmpty ? "unknown ffi error" : s
}

func parseFlag(_ name: String, in args: [String], default value: String? = nil) -> String? {
    if let i = args.firstIndex(of: name), i + 1 < args.count {
        return args[i + 1]
    }
    return value
}

func decodeOne(tok: cellm_tokenizer_t, token: UInt32) -> String {
    var t = token
    let needed = withUnsafePointer(to: &t) { ptr in
        cellm_tokenizer_decode(tok, ptr, 1, nil, 0)
    }
    if needed == 0 { return "" }
    var buf = [CChar](repeating: 0, count: needed + 1)
    let written = withUnsafePointer(to: &t) { ptr in
        cellm_tokenizer_decode(tok, ptr, 1, &buf, buf.count)
    }
    if written == 0 { return "" }
    return String(cString: buf)
}

func encode(tok: cellm_tokenizer_t, text: String) throws -> [UInt32] {
    let needed = text.withCString { cstr in
        cellm_tokenizer_encode(tok, cstr, nil, 0)
    }
    if needed == 0 {
        throw SmokeError.message("tokenizer_encode(size) failed: \(ffiLastError())")
    }
    var out = [UInt32](repeating: 0, count: needed)
    let written = text.withCString { cstr in
        out.withUnsafeMutableBufferPointer { bp in
            cellm_tokenizer_encode(tok, cstr, bp.baseAddress, bp.count)
        }
    }
    if written != needed {
        throw SmokeError.message("tokenizer_encode wrote \(written), expected \(needed)")
    }
    return out
}

func wrapGemmaPrompt(_ prompt: String) -> String {
    let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    return "<start_of_turn>user\n\(p)<end_of_turn>\n<start_of_turn>model\n"
}

func backendRaw(_ s: String) -> UInt32 {
    s.lowercased() == "metal" ? UInt32(CELLM_BACKEND_METAL.rawValue) : UInt32(CELLM_BACKEND_CPU.rawValue)
}

let args = CommandLine.arguments
guard let model = parseFlag("--model", in: args),
      let tokenizerPath = parseFlag("--tokenizer", in: args),
      let prompt = parseFlag("--prompt", in: args)
else {
    fputs("""
Usage:
  swift run CellmSmoke --model /path/model.cellm[d] --tokenizer /path/tokenizer.json --prompt "text" [--backend cpu|metal] [--gen 32]
""", stderr)
    exit(2)
}

let backend = parseFlag("--backend", in: args, default: "cpu") ?? "cpu"
let gen = Int(parseFlag("--gen", in: args, default: "32") ?? "32") ?? 32

if backend.lowercased() == "metal" {
    let smoke = cellm_metal_smoke_test()
    if smoke != 0 {
        throw SmokeError.message("metal smoke failed: \(ffiLastError())")
    }
}

let tok = tokenizerPath.withCString { cstr in
    cellm_tokenizer_create(cstr)
}
if tok == 0 {
    throw SmokeError.message("tokenizer_create failed: \(ffiLastError())")
}
defer { cellm_tokenizer_destroy(tok) }

let engine = model.withCString { cstr in
    cellm_engine_create_v4(
        cstr,
        8,
        48,
        40,
        0.0,
        1.08,
        96,
        1,
        backendRaw(backend),
        UInt32(CELLM_KV_ENCODING_F16.rawValue),
        1,
        1
    )
}
if engine == 0 {
    throw SmokeError.message("engine_create_v4 failed: \(ffiLastError())")
}
defer { cellm_engine_destroy(engine) }

let session = cellm_session_create(engine)
if session == 0 {
    throw SmokeError.message("session_create failed: \(ffiLastError())")
}
defer { _ = cellm_session_cancel(engine, session) }

var backendBuf = [CChar](repeating: 0, count: 32)
_ = cellm_engine_backend_name(engine, &backendBuf, backendBuf.count)
print("active_backend=\(String(cString: backendBuf))")

let wrapped = wrapGemmaPrompt(prompt)
let promptTokens = try encode(tok: tok, text: wrapped)
print("prompt_tokens=\(promptTokens.count)")

var next: UInt32 = 0
let submitRc = promptTokens.withUnsafeBufferPointer { bp in
    cellm_submit_tokens(engine, session, bp.baseAddress, bp.count, &next)
}
if submitRc != 0 {
    throw SmokeError.message("submit_tokens failed: \(ffiLastError())")
}

var textOut = ""
for i in 0..<gen {
    let piece = decodeOne(tok: tok, token: next)
    let sanitized = piece.replacingOccurrences(of: "\n", with: "\\n")
    print("gen[\(i)] token=\(next) piece=\"\(sanitized)\"")
    textOut += piece

    var sid: cellm_session_t = 0
    var tokNext: UInt32 = 0
    let rc = cellm_step_decode(engine, &sid, &tokNext)
    if rc < 0 {
        throw SmokeError.message("step_decode failed: \(ffiLastError())")
    }
    if rc == 0 {
        break
    }
    next = tokNext
}

print("----")
print(textOut.isEmpty ? "<empty>" : textOut)
