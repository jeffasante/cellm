import Foundation

protocol ModelDataProcessor {
    var label: String { get }
    func wrapPrompt(_ prompt: String, system: String?) -> String
    func isStopPiece(_ piece: String) -> Bool
}

enum ModelDataProcessorFactory {
    static func make(tokenizerURL: URL, modelURL: URL?) -> any ModelDataProcessor {
        let tokenizerName = tokenizerURL.lastPathComponent.lowercased()
        let modelName = modelURL?.lastPathComponent.lowercased() ?? ""
        let cfgURL = tokenizerURL.deletingLastPathComponent().appendingPathComponent("tokenizer_config.json")

        if
            let data = try? Data(contentsOf: cfgURL),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let tpl = obj["chat_template"] as? String
        {
            if tpl.contains("<end_of_utterance>") && tpl.contains("Assistant:") {
                return SmolChatProcessor()
            }
            if tpl.contains("<|im_start|>") && tpl.contains("<|im_end|>") {
                // Qwen chat templates may include think prefill; keep disabled for now
                // to avoid degenerate output on mobile.
                return ChatMLProcessor(includeThinkPrefill: false)
            }
            if tpl.contains("<|turn>") && tpl.contains("<turn|>") {
                return Gemma4TurnProcessor()
            }
            if tpl.contains("<start_of_turn>") && tpl.contains("<end_of_turn>") {
                return GemmaTurnProcessor()
            }
        }

        if tokenizerName.contains("gemma-4") || modelName.contains("gemma-4") {
            return Gemma4TurnProcessor()
        }
        if tokenizerName.contains("gemma") || modelName.contains("gemma") {
            return GemmaTurnProcessor()
        }
        if tokenizerName.contains("qwen") || modelName.contains("qwen") {
            return ChatMLProcessor(includeThinkPrefill: false)
        }
        if tokenizerName.contains("smollm") || modelName.contains("smollm") {
            return SmolChatProcessor()
        }
        return PlainProcessor()
    }
}

private struct SmolChatProcessor: ModelDataProcessor {
    let label = "smol_chat"

    func wrapPrompt(_ prompt: String, system: String?) -> String {
        let cleanPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let sys = system ?? "You are a helpful assistant."
        return "<|im_start|>system\n\(sys)<|im_end|>\n<|im_start|>user\n\(cleanPrompt)<|im_end|>\n<|im_start|>assistant\n"
    }

    func isStopPiece(_ piece: String) -> Bool {
        let p = piece.trimmingCharacters(in: .whitespacesAndNewlines)
        return p == "<end_of_utterance>" || p == "<|endoftext|>"
    }
}

private struct ChatMLProcessor: ModelDataProcessor {
    let includeThinkPrefill: Bool
    let label = "chatml"

    func wrapPrompt(_ prompt: String, system: String?) -> String {
        let cleanPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        // Qwen and SmolLM models can heavily hallucinate conversational formatting
        // (like "Human:" and "assistant:") if the system prompt is completely omitted.
        let sys = system ?? "You are a helpful AI assistant."
        var s = "<|im_start|>system\n\(sys)<|im_end|>\n"
        s += "<|im_start|>user\n\(cleanPrompt)<|im_end|>\n<|im_start|>assistant\n"
        if includeThinkPrefill {
            s += "<think>\n\n</think>\n\n"
        }
        return s
    }

    func isStopPiece(_ piece: String) -> Bool {
        let p = piece.trimmingCharacters(in: .whitespacesAndNewlines)
        return p == "<|im_end|>" || p == "<|endoftext|>" || p.hasSuffix("<|im_end|>")
    }
}

private struct GemmaTurnProcessor: ModelDataProcessor {
    let label = "gemma_turn"

    func wrapPrompt(_ prompt: String, system: String?) -> String {
        let cleanPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let userText = if let sys = system, !sys.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            "\(sys)\n\n\(cleanPrompt)"
        } else {
            cleanPrompt
        }
        // Do not prepend BOS as raw text. If BOS is needed it should be
        // injected as an id-level special token, not literal bytes.
        return "<start_of_turn>user\n\(userText)<end_of_turn>\n<start_of_turn>model\n"
    }

    func isStopPiece(_ piece: String) -> Bool {
        let p = piece.trimmingCharacters(in: .whitespacesAndNewlines)
        return p == "<end_of_turn>" || p == "<|endoftext|>"
    }
}

private struct Gemma4TurnProcessor: ModelDataProcessor {
    let label = "gemma4_turn"

    func wrapPrompt(_ prompt: String, system: String?) -> String {
        let cleanPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let userText = if let sys = system, !sys.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            "\(sys)\n\n\(cleanPrompt)"
        } else {
            cleanPrompt
        }
        return "<|turn>user\n\(userText)<turn|>\n<|turn>model\n"
    }

    func isStopPiece(_ piece: String) -> Bool {
        let p = piece.trimmingCharacters(in: .whitespacesAndNewlines)
        return p == "<turn|>" || p == "<|endoftext|>"
    }
}

private struct PlainProcessor: ModelDataProcessor {
    let label = "plain"

    func wrapPrompt(_ prompt: String, system: String?) -> String {
        let cleanPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if let sys = system, !sys.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "System: \(sys)\nUser: \(cleanPrompt)\nAssistant:"
        }
        return "User: \(cleanPrompt)\nAssistant:"
    }

    func isStopPiece(_ piece: String) -> Bool {
        let p = piece.trimmingCharacters(in: .whitespacesAndNewlines)
        return p == "<|endoftext|>"
    }
}
