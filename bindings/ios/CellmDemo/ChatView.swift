import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

struct ChatView: View {
    private enum SharedSelection {
        static let llmModelPath = "cellm.shared.llm.model.path"
        static let llmTokenizerPath = "cellm.shared.llm.tokenizer.path"
    }

    private enum PickerTarget: String, Identifiable {
        case llmModel
        case tokenizer
        case vlmModel
        case audioFile

        var id: String { rawValue }
        var allowedTypes: [UTType] {
            switch self {
            case .tokenizer: return [.json]
            case .audioFile: return [.audio]
            case .llmModel, .vlmModel: return [.item]
            }
        }
    }

    private struct ChatMessage: Identifiable {
        let id = UUID()
        let role: String
        let text: String
        let imageData: Data?
        let audioFileName: String?
    }

    @State private var messages: [ChatMessage] = []
    @State private var inputText: String = ""

    @State private var llmModelURL: URL?
    @State private var llmTokenizerURL: URL?
    @State private var vlmModelURL: URL?

    @State private var pickerTarget: PickerTarget?

    @State private var selectedImageItem: PhotosPickerItem?
    @State private var pendingImageData: Data?
    @State private var pendingImage: Image?
    @State private var pendingAudioURL: URL?

    @State private var selectedBackend: CellmBackend = .metal
    @State private var activeBackend: String = ""
    @State private var isRunning = false
    @State private var errorText: String?
    @State private var infoText: String = ""
    @State private var selectedSampleLabel: String = ""
    @State private var runDiagnostics: String = ""
    @FocusState private var isComposerFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            configBar
            if !runDiagnostics.isEmpty {
                diagnosticsBar
            }
            Divider()
            messagesView
            composer
        }
        .background(Color(.systemGroupedBackground).ignoresSafeArea())
        .contentShape(Rectangle())
        .onTapGesture {
            dismissKeyboard()
        }
        .sheet(item: $pickerTarget) { target in
            DocumentPicker(allowed: target.allowedTypes) { url in
                switch target {
                case .llmModel:
                    llmModelURL = persistPickedFile(url, subdir: "picked/chat/llm")
                case .tokenizer:
                    llmTokenizerURL = persistPickedFile(url, subdir: "picked/chat/tokenizer")
                case .vlmModel:
                    vlmModelURL = persistPickedFile(url, subdir: "picked/chat/vlm")
                case .audioFile:
                    pendingAudioURL = persistPickedFile(url, subdir: "picked/chat/audio")
                }
            }
        }
        .onChange(of: selectedImageItem) { item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let ui = UIImage(data: data) {
                    await MainActor.run {
                        pendingImageData = normalizedJPEGData(from: ui) ?? data
                        pendingImage = Image(uiImage: ui)
                    }
                }
            }
        }
        .onAppear {
            restoreDefaults()
        }
        .onChange(of: llmModelURL) { _ in persistSharedSelection() }
        .onChange(of: llmTokenizerURL) { _ in persistSharedSelection() }
    }

    private var diagnosticsBar: some View {
        Text(runDiagnostics)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.bottom, 10)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("cellm Chat")
                .font(.system(size: 34, weight: .bold))
            Text("Multimodal chat with optional image per turn")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            if !infoText.isEmpty {
                Text(infoText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            if !selectedSampleLabel.isEmpty {
                Text("Selected: \(selectedSampleLabel)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            if let errorText {
                Text(errorText)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var configBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                Button(llmModelURL == nil ? "Pick LLM .cellm" : "LLM: \(llmModelURL!.lastPathComponent)") { pickerTarget = .llmModel }
                    .buttonStyle(.bordered)
                Button(llmTokenizerURL == nil ? "Pick tokenizer.json" : "Tok: \(llmTokenizerURL!.lastPathComponent)") { pickerTarget = .tokenizer }
                    .buttonStyle(.bordered)
                Button(vlmModelURL == nil ? "Pick VLM .cellm" : "VLM: \(vlmModelURL!.lastPathComponent)") { pickerTarget = .vlmModel }
                    .buttonStyle(.bordered)
                Picker("Backend", selection: $selectedBackend) {
                    ForEach(CellmBackend.allCases) { backend in
                        Text(backend.label).tag(backend)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 180)
                if !activeBackend.isEmpty {
                    Text("active: \(activeBackend)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Button("Use Gemma") { selectGemmaPreset() }
                    .buttonStyle(.borderedProminent)
                Button("Use Qwen") { selectQwenPreset() }
                    .buttonStyle(.borderedProminent)
                Button("Use SmolLM") { selectSmolPreset() }
                    .buttonStyle(.borderedProminent)
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 10)
        }
    }

    private var messagesView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(messages) { msg in
                        messageBubble(msg)
                    }
                }
                .padding(12)
            }
            .onChange(of: messages.count) { _ in
                if let last = messages.last {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            .scrollDismissesKeyboard(.interactively)
        }
    }

    private var composer: some View {
        VStack(spacing: 10) {
            if let imagePreview = pendingImage {
                HStack(spacing: 10) {
                    imagePreview
                        .resizable()
                        .scaledToFit()
                        .frame(width: 84, height: 84)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    Text("Image attached")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Remove") {
                        pendingImage = nil
                        pendingImageData = nil
                        selectedImageItem = nil
                    }
                    .buttonStyle(.bordered)
                }
                .padding(10)
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            if let audioURL = pendingAudioURL {
                HStack(spacing: 10) {
                    Image(systemName: "waveform")
                        .foregroundStyle(.blue)
                    Text("Audio attached: \(audioURL.lastPathComponent)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Remove") {
                        pendingAudioURL = nil
                    }
                    .buttonStyle(.bordered)
                }
                .padding(10)
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }

            HStack(alignment: .bottom, spacing: 8) {
                PhotosPicker(selection: $selectedImageItem, matching: .images) {
                    Image(systemName: "photo")
                        .font(.title3)
                        .padding(10)
                        .background(Color(.secondarySystemGroupedBackground))
                        .clipShape(Circle())
                }
                Button {
                    pickerTarget = .audioFile
                } label: {
                    Image(systemName: "waveform")
                        .font(.title3)
                        .padding(10)
                        .background(Color(.secondarySystemGroupedBackground))
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
                TextField("Message", text: $inputText, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...5)
                    .focused($isComposerFocused)
                Button(isRunning ? "..." : "Send") {
                    sendMessage()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isRunning || inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(12)
        .background(Color(.systemBackground))
    }

    private func messageBubble(_ msg: ChatMessage) -> some View {
        VStack(alignment: msg.role == "User" ? .trailing : .leading, spacing: 6) {
            Text(msg.role)
                .font(.caption)
                .foregroundStyle(.secondary)
            if let data = msg.imageData, let ui = UIImage(data: data) {
                Image(uiImage: ui)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 220)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
            if let audioFileName = msg.audioFileName {
                Label("Audio: \(audioFileName)", systemImage: "waveform")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            Text(msg.text)
                .padding(10)
                .background(msg.role == "User" ? Color.blue.opacity(0.12) : Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: msg.role == "User" ? .trailing : .leading)
    }

    private func sendMessage() {
        dismissKeyboard()
        errorText = nil
        runDiagnostics = ""
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        let attachedImage = pendingImageData
        let attachedAudio = pendingAudioURL
        messages.append(
            ChatMessage(
                role: "User",
                text: text,
                imageData: attachedImage,
                audioFileName: attachedAudio?.lastPathComponent
            )
        )
        inputText = ""
        pendingImage = nil
        pendingImageData = nil
        selectedImageItem = nil
        pendingAudioURL = nil

        isRunning = true
        messages.append(ChatMessage(role: "Assistant", text: "", imageData: nil, audioFileName: nil))
        let assistantIndex = messages.count - 1

        let backend = selectedBackend
        let textTranscript = conversationPrompt(maxTurns: 8, includeAssistantPlaceholder: true)
        let imagePrompt = vlmPrompt(userText: text)
        let currentVLMModelURL = vlmModelURL
        let currentLLMModelURL = llmModelURL
        let currentLLMTokenizerURL = llmTokenizerURL
        let currentAudioURL = attachedAudio

        Task.detached(priority: .userInitiated) {
            do {
                let initStart = Date()
                var diag: [String] = []
                guard let llmModelURL = currentLLMModelURL,
                      let llmTokenizerURL = currentLLMTokenizerURL else {
                    throw CellmError.message("Pick LLM model + tokenizer for chat.")
                }
                let initTokenizerStart = Date()
                let tok = try CellmTokenizer(tokenizerURL: llmTokenizerURL)
                let initTokenizerMs = Date().timeIntervalSince(initTokenizerStart) * 1000.0
                let initEngineStart = Date()
                let eng = try CellmEngine(
                    modelURL: llmModelURL,
                    tokenizer: tok,
                    topK: 20,
                    temperature: 0.2,
                    repeatPenalty: 1.08,
                    repeatWindow: 96,
                    backend: backend
                )
                let initEngineMs = Date().timeIntervalSince(initEngineStart) * 1000.0
                let initTotalMs = Date().timeIntervalSince(initStart) * 1000.0
                diag.append(String(format: "init_tokenizer=%.1fms init_engine=%.1fms init_total=%.1fms", initTokenizerMs, initEngineMs, initTotalMs))
                diag.append("requested_backend=\(backend.label.lowercased()) active_backend=\(eng.activeBackend)")

                if eng.isLiteRtProxy, (attachedImage != nil || currentAudioURL != nil) {
                    let tempImageURL = attachedImage.flatMap { writeTempImageForLiteRt($0) }
                    let reply = try eng.generate(
                        prompt: text,
                        maxNewTokens: 128,
                        imageURL: tempImageURL,
                        audioURL: currentAudioURL
                    )
                    await MainActor.run {
                        activeBackend = eng.activeBackend
                        runDiagnostics = (diag + ["mode=litert_multimodal"]).joined(separator: "\n")
                        messages[assistantIndex] = ChatMessage(role: "Assistant", text: prettyOutput(reply), imageData: nil, audioFileName: nil)
                        isRunning = false
                    }
                    return
                }

                if let imageBytes = attachedImage {
                    guard let vlmModelURL = currentVLMModelURL else {
                        throw CellmError.message("Attach a VLM model to use image chat.")
                    }
                    let vlmEng = try CellmVLMEngine(
                        modelURL: vlmModelURL,
                        topK: 20,
                        temperature: 0.2,
                        repeatPenalty: 1.15,
                        repeatWindow: 64,
                        backend: backend
                    )
                    let reply = try vlmEng.describe(imageBytes: imageBytes, prompt: imagePrompt)
                    await MainActor.run {
                        activeBackend = vlmEng.activeBackend
                        runDiagnostics = (diag + ["mode=vlm_image"]).joined(separator: "\n")
                        messages[assistantIndex] = ChatMessage(role: "Assistant", text: sanitizeVlmOutput(reply), imageData: nil, audioFileName: nil)
                        isRunning = false
                    }
                    return
                }

                let reply = try eng.generate(
                    prompt: textTranscript,
                    maxNewTokens: 128,
                    thermalLevel: .nominal,
                    exerciseSuspendResume: false,
                    onToken: { piece in
                        Task { @MainActor in
                            let existing = messages[assistantIndex].text
                            messages[assistantIndex] = ChatMessage(role: "Assistant", text: existing + piece, imageData: nil, audioFileName: nil)
                        }
                    }
                )
                await MainActor.run {
                    activeBackend = eng.activeBackend
                    if let stats = eng.lastGenerationStats {
                        diag.append(
                            String(
                                format: "prompt_tokens=%d generated_tokens=%d prefill=%.1fms decode=%.1fms total=%.1fms stop_reason=%@",
                                stats.promptTokenCount,
                                stats.generatedTokenCount,
                                finiteMs(stats.prefillMs),
                                finiteMs(stats.decodeMs),
                                finiteMs(stats.totalMs),
                                stats.stopReason
                            )
                        )
                    }
                    runDiagnostics = diag.joined(separator: "\n")
                    messages[assistantIndex] = ChatMessage(role: "Assistant", text: prettyOutput(reply), imageData: nil, audioFileName: nil)
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    errorText = String(describing: error)
                    if assistantIndex < messages.count {
                        messages[assistantIndex] = ChatMessage(role: "Assistant", text: "", imageData: nil, audioFileName: nil)
                    }
                    isRunning = false
                }
            }
        }
    }

    private func conversationPrompt(maxTurns: Int, includeAssistantPlaceholder: Bool) -> String {
        let recent = messages.suffix(maxTurns)
        var lines: [String] = [
            "You are a helpful assistant. Keep answers concise and accurate."
        ]
        for turn in recent {
            if turn.role == "User" {
                if turn.imageData != nil {
                    lines.append("User (with image): \(turn.text)")
                } else {
                    lines.append("User: \(turn.text)")
                }
            } else if !turn.text.isEmpty {
                lines.append("Assistant: \(turn.text)")
            }
        }
        if includeAssistantPlaceholder {
            lines.append("Assistant:")
        }
        return lines.joined(separator: "\n")
    }

    private func vlmPrompt(userText: String) -> String {
        let q = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        return """
        Describe only what is clearly visible in the image.
        Keep the response short (1-2 sentences), concrete, and avoid repeating phrases.
        User question: \(q)
        """
    }

    private func prettyOutput(_ text: String) -> String {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        let lines = normalized.components(separatedBy: "\n")
        var out: [String] = []
        var previousBlank = false
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty {
                if !previousBlank {
                    out.append("")
                    previousBlank = true
                }
                continue
            }
            previousBlank = false
            out.append(trimmed)
        }
        return out.joined(separator: "\n")
    }

    private func sanitizeVlmOutput(_ text: String) -> String {
        let pretty = prettyOutput(text)
        if pretty.isEmpty { return pretty }
        let lines = pretty.components(separatedBy: "\n")
        var seen = Set<String>()
        var cleaned: [String] = []
        for raw in lines {
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty {
                if cleaned.last != "" {
                    cleaned.append("")
                }
                continue
            }
            if seen.contains(line) { continue }
            seen.insert(line)
            cleaned.append(line)
            if cleaned.count >= 4 { break }
        }
        return cleaned.joined(separator: "\n")
    }

    private func finiteMs(_ value: Double) -> Double {
        value.isFinite ? value : 0.0
    }

    private func dismissKeyboard() {
        isComposerFocused = false
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }

    private func normalizedJPEGData(from image: UIImage) -> Data? {
        let size = image.size
        guard size.width > 0, size.height > 0 else { return nil }
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1.0
        let rendered = UIGraphicsImageRenderer(size: size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
        return rendered.jpegData(compressionQuality: 0.95)
    }

    private func writeTempImageForLiteRt(_ data: Data) -> URL? {
        do {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("cellm_chat_litert_\(UUID().uuidString)")
                .appendingPathExtension("jpg")
            try data.write(to: url, options: .atomic)
            return url
        } catch {
            return nil
        }
    }

    private func restoreDefaults() {
        if let modelPath = UserDefaults.standard.string(forKey: SharedSelection.llmModelPath),
           let tokPath = UserDefaults.standard.string(forKey: SharedSelection.llmTokenizerPath) {
            let model = URL(fileURLWithPath: modelPath)
            let tok = URL(fileURLWithPath: tokPath)
            if FileManager.default.fileExists(atPath: model.path),
               FileManager.default.fileExists(atPath: tok.path) {
                llmModelURL = model
                llmTokenizerURL = tok
                selectedSampleLabel = "Shared from LLM"
            }
        }

        if llmModelURL == nil || llmTokenizerURL == nil {
            selectGemmaPreset(silentOnMissing: true)
            if llmModelURL == nil || llmTokenizerURL == nil {
                selectQwenPreset(silentOnMissing: true)
            }
            if llmModelURL == nil || llmTokenizerURL == nil {
                selectSmolPreset(silentOnMissing: true)
            }
        }
        if vlmModelURL == nil {
            vlmModelURL = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmFileName)
        }
        if messages.isEmpty {
            messages = [
                ChatMessage(role: "Assistant", text: "Hi. You can chat with text, attach an image, or attach audio (LiteRT Gemma).", imageData: nil, audioFileName: nil)
            ]
        }
        infoText = "Text uses LLM model/tokenizer. Image uses LiteRT (if selected) or VLM. Audio uses LiteRT."
    }

    private func persistSharedSelection() {
        if let model = llmModelURL?.path {
            UserDefaults.standard.set(model, forKey: SharedSelection.llmModelPath)
        }
        if let tok = llmTokenizerURL?.path {
            UserDefaults.standard.set(tok, forKey: SharedSelection.llmTokenizerPath)
        }
    }

    private func selectGemmaPreset(silentOnMissing: Bool = false) {
        let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma42p3bFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma3FileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "gemma-3-1b-it-int8.cellmd")
        let tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma42p3bTokenizerFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma3TokenizerFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "tokenizer-gemma-3-1b-it.json")
        if let model, let tok {
            llmModelURL = model
            llmTokenizerURL = tok
            selectedSampleLabel = model.lastPathComponent.contains("gemma-4-2p3b-it")
                ? "Gemma-4-2P3B-IT (LiteRT)"
                : "Gemma3-1B-IT"
            errorText = nil
        } else if !silentOnMissing {
            errorText = "Gemma files not found in Documents/samples."
        }
    }

    private func selectQwenPreset(silentOnMissing: Bool = false) {
        let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35StableFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35CompactFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "qwen3.5-0.8b.cellm")
            ?? RemoteAssets.existingDocumentsFile(fileName: "qwen3.5-0.8b-int8.cellm")
            ?? RemoteAssets.existingDocumentsFile(fileName: "qwen3.5-0.8b-int4-textonly.cellm")
        let tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35TokenizerFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "tokenizer-qwen3.5-0.8b.json")
        if let model, let tok {
            llmModelURL = model
            llmTokenizerURL = tok
            selectedSampleLabel = "Qwen3.5"
            errorText = nil
        } else if !silentOnMissing {
            errorText = "Qwen files not found in Documents/samples."
        }
    }

    private func selectSmolPreset(silentOnMissing: Bool = false) {
        let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2FileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "smollm2-135m-int8.cellm")
            ?? RemoteAssets.existingDocumentsFile(fileName: "smollm2-135m.cellm")
        let tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "tokenizer-smollm2-135m.json")
        if let model, let tok {
            llmModelURL = model
            llmTokenizerURL = tok
            selectedSampleLabel = "SmolLM2"
            errorText = nil
        } else if !silentOnMissing {
            errorText = "SmolLM files not found in Documents/samples."
        }
    }

    private func persistPickedFile(_ sourceURL: URL, subdir: String) -> URL? {
        do {
            let name = sourceURL.lastPathComponent
            let dest = RemoteAssets.documentsURL(fileName: "\(subdir)/\(name)")
            let dir = dest.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: nil)
            if FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.removeItem(at: dest)
            }
            try FileManager.default.copyItem(at: sourceURL, to: dest)
            return dest
        } catch {
            errorText = "Failed to import selected file: \(error.localizedDescription)"
            return nil
        }
    }
}
