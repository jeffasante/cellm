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
    @State private var showSettings = false
    @State private var temperature: Double = 0.2
    @State private var maxNewTokens: Int = 200
    @State private var isInitializing = false
    @FocusState private var isComposerFocused: Bool
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.scenePhase) private var scenePhase
    @State private var generationTask: Task<Void, Never>?
    @State private var initTask: Task<Void, Never>?

    // Persistent engine + tokenizer to avoid reloading model weights on every message.
    @State private var cachedEngine: CellmEngine?
    @State private var cachedTokenizer: CellmTokenizer?
    @State private var cachedEngineModelURL: URL?
    @State private var cachedEngineBackend: CellmBackend?

    var body: some View {
        ZStack {
            Color(.systemBackground).ignoresSafeArea()
            
            VStack(spacing: 0) {
                premiumHeader
                
                if messages.isEmpty && !isRunning {
                    Spacer()
                    emptyStateHero
                    Spacer()
                } else {
                    messagesView
                }
                
                premiumComposer
            }
        }
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
            initializeEngine()
        }
        .onChange(of: llmModelURL) { _ in 
            persistSharedSelection()
            invalidateCachedEngine()
            initializeEngine()
        }
        .onChange(of: selectedBackend) { _ in
            invalidateCachedEngine()
        }
        .onChange(of: llmTokenizerURL) { _ in persistSharedSelection() }
        .onChange(of: scenePhase) { phase in
            guard phase != .active else { return }
            generationTask?.cancel()
            generationTask = nil
            initTask?.cancel()
            initTask = nil
            if isRunning || isInitializing {
                isRunning = false
                isInitializing = false
                errorText = "Generation stopped: app moved to background/inactive. Reopen app and send again."
            }
        }
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

    private var premiumHeader: some View {
        VStack(spacing: 12) {
            // Navigation Row
            HStack {
                Button {
                    // Back action
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
                
                Spacer()
                
                HStack(spacing: 8) {
                    Image(systemName: "bubble.left.and.exclamationmark.bubble.right.fill")
                        .foregroundStyle(.blue)
                    Text("AI Chat")
                        .font(.headline)
                        .foregroundStyle(.blue)
                }
                
                Spacer()
                
                HStack(spacing: 16) {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }
                    .sheet(isPresented: $showSettings) {
                        GenerationSettingsSheet(
                            temperature: $temperature,
                            maxNewTokens: $maxNewTokens,
                            selectedBackend: $selectedBackend
                        )
                    }
                    Button {
                        // New Chat action
                    } label: {
                        Image(systemName: "plus.circle")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)
            
            // Model Selector Pill
            Menu {
                Section("Smart Presets") {
                    Button("Gemma 3 (Stable)") { selectGemmaPreset() }
                    Button("Qwen 3.5 (Stable)") { selectQwenPreset() }
                    Button("SmolLM 2 (Fast)") { selectSmolPreset() }
                    Button("Bonsai 1.7B (1-Bit)") { selectBonsaiPreset() }
                }
                Section("Advanced Overrides") {
                    Button("Pick Custom LLM...") { pickerTarget = .llmModel }
                    Button("Pick Custom Tokenizer...") { pickerTarget = .tokenizer }
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.down.circle.fill")
                        .font(.caption)
                    Text(selectedSampleLabel.isEmpty ? "Select Model" : selectedSampleLabel)
                        .font(.subheadline.bold())
                    Image(systemName: "chevron.down")
                        .font(.caption2)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color(.systemGray6).opacity(0.1))
                .foregroundStyle(.primary)
                .clipShape(Capsule())
                .overlay(Capsule().stroke(Color(.separator).opacity(0.35), lineWidth: 0.5))
            }
            
            // Backend Badge
            if !activeBackend.isEmpty {
                HStack(spacing: 4) {
                    Image(systemName: activeBackend == "metal" ? "bolt.fill" : "cpu")
                        .font(.caption2)
                    Text(activeBackend.uppercased())
                        .font(.caption2.bold())
                }
                .foregroundStyle(activeBackend == "metal" ? .green : .secondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(
                    (activeBackend == "metal" ? Color.green : Color.gray)
                        .opacity(0.12)
                )
                .clipShape(Capsule())
            }
            
            // Initializing Status Pill
            if isInitializing || isRunning {
                HStack(spacing: 8) {
                    ProgressView()
                        .tint(.cyan)
                        .scaleEffect(0.8)
                    Text(isInitializing ? "Initializing model..." : "Generating...")
                        .font(.footnote.bold())
                        .foregroundStyle(.cyan)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(Color.cyan.opacity(0.15))
                .clipShape(Capsule())
                .transition(.opacity.combined(with: .scale))
            }

            if let errorText, !errorText.isEmpty {
                Text(errorText)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.red.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .padding(.horizontal, 16)
            }
        }
        .padding(.bottom, 16)
    }

    private var emptyStateHero: some View {
        VStack(spacing: 12) {
            Text("AI Chat")
                .font(.system(size: 40, weight: .medium, design: .rounded))
                .foregroundStyle(.primary)
            
            Text("Chat with an on-device large\nlanguage model")
                .font(.title3)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
    }

    private var premiumComposer: some View {
        VStack(spacing: 12) {
            // Attachments Preview
            if pendingImage != nil || pendingAudioURL != nil {
                HStack {
                    if let image = pendingImage {
                        image.resizable()
                            .scaledToFill()
                            .frame(width: 50, height: 50)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    if let audio = pendingAudioURL {
                        Image(systemName: "waveform")
                            .foregroundStyle(.blue)
                        Text(audio.lastPathComponent)
                            .font(.caption)
                    }
                    Spacer()
                    Button {
                        pendingImage = nil
                        pendingImageData = nil
                        pendingAudioURL = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(8)
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .padding(.horizontal, 16)
            }
            
            // Input Area
            HStack(alignment: .bottom, spacing: 12) {
                Button {
                    // Logic for adding attachments
                    pickerTarget = .audioFile
                } label: {
                    Image(systemName: "plus")
                        .font(.title3.bold())
                        .padding(10)
                        .background(Color(.secondarySystemGroupedBackground))
                        .foregroundStyle(.primary)
                        .clipShape(Circle())
                }
                
                ZStack(alignment: .leading) {
                    if inputText.isEmpty {
                        Text("Type prompt...")
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 14)
                    }
                    TextEditor(text: $inputText)
                        .scrollContentBackground(.hidden)
                        .background(Color.clear)
                        .foregroundStyle(.primary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 8)
                        .frame(minHeight: 44, maxHeight: 96)
                        .focused($isComposerFocused)
                }
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 24))
                .overlay(RoundedRectangle(cornerRadius: 24).stroke(Color(.separator).opacity(0.35), lineWidth: 1))
                
                Button {
                    sendMessage()
                } label: {
                    Image(systemName: "paperplane.fill")
                        .font(.title3)
                        .padding(12)
                        .background(inputText.isEmpty || isRunning ? Color(.secondarySystemGroupedBackground) : Color.accentColor)
                        .foregroundStyle(inputText.isEmpty || isRunning ? Color.secondary : Color.white)
                        .clipShape(Circle())
                }
                .disabled(inputText.isEmpty || isRunning)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 24)
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
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(msg.role == "User" ? Color.accentColor.opacity(colorScheme == .dark ? 0.35 : 0.18) : Color(.secondarySystemGroupedBackground))
                .foregroundStyle(.primary)
                .clipShape(RoundedRectangle(cornerRadius: 18))
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: msg.role == "User" ? .trailing : .leading)
    }

    private func sendMessage() {
        guard scenePhase == .active else {
            errorText = "Bring app to foreground before running Metal generation."
            return
        }
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
        // Let model-specific prompt formatting in the engine handle role wrapping.
        // Passing a raw transcript here can degrade quality for gemma_turn/chat templates.
        let textPrompt = text
        let imagePrompt = vlmPrompt(userText: text)
        let currentVLMModelURL = vlmModelURL
        let currentLLMModelURL = llmModelURL
        let currentLLMTokenizerURL = llmTokenizerURL
        let currentAudioURL = attachedAudio

        generationTask?.cancel()
        generationTask = Task.detached(priority: .userInitiated) {
            do {
                let initStart = Date()
                var diag: [String] = []
                guard let llmModelURL = currentLLMModelURL,
                      let llmTokenizerURL = currentLLMTokenizerURL else {
                    throw CellmError.message("Pick LLM model + tokenizer for chat.")
                }

                // Reuse cached engine if model and backend haven't changed.
                let eng: CellmEngine
                let engineWasCached: Bool
                if let cached = await MainActor.run(body: { cachedEngine }),
                   await MainActor.run(body: { cachedEngineModelURL }) == llmModelURL,
                   await MainActor.run(body: { cachedEngineBackend }) == backend {
                    eng = cached
                    engineWasCached = true
                    try eng.resetSession()
                    diag.append("engine=cached (reused)")
                } else {
                    await MainActor.run { isInitializing = true }
                    let tok = try CellmTokenizer(tokenizerURL: llmTokenizerURL)
                    let capturedTemp = await MainActor.run { temperature }
                    let topK: UInt32 = capturedTemp < 0.05 ? 1 : 40
                    let newEng = try CellmEngine(
                        modelURL: llmModelURL,
                        tokenizer: tok,
                        topK: topK,
                        temperature: Float(capturedTemp),
                        repeatPenalty: 1.12,
                        repeatWindow: 96,
                        backend: backend
                    )
                    eng = newEng
                    engineWasCached = false
                    await MainActor.run {
                        cachedEngine = newEng
                        cachedTokenizer = tok
                        cachedEngineModelURL = llmModelURL
                        cachedEngineBackend = backend
                        isInitializing = false
                    }
                    let initMs = Date().timeIntervalSince(initStart) * 1000.0
                    diag.append(String(format: "engine=fresh init=%.1fms", initMs))
                }
                let capturedMaxToks = await MainActor.run { maxNewTokens }
                diag.append("requested_backend=\(backend.label.lowercased()) active_backend=\(eng.activeBackend) cached=\(engineWasCached)")

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

                // Throttle UI updates to avoid CoreGraphics NaN from rapid SwiftUI relayout.
                // Accumulate tokens and flush to the view at most every ~80ms.
                let streamBuffer = StreamBuffer()
                let systemPrompt = llmModelURL.lastPathComponent.contains("Bonsai")
                    ? "I am a 1-bit model developed by PrismML. I was created by the team at Caltech and is based in Pasadena, California."
                    : nil

                let reply = try eng.generate(
                    prompt: textPrompt,
                    system: systemPrompt,
                    maxNewTokens: capturedMaxToks,
                    thermalLevel: .nominal,
                    exerciseSuspendResume: false,
                    onToken: { piece in
                        streamBuffer.append(piece)
                        let pending = streamBuffer.flushIfReady()
                        guard let pending, !pending.isEmpty else { return }
                        Task { @MainActor in
                            guard assistantIndex < messages.count else { return }
                            let existing = messages[assistantIndex].text
                            messages[assistantIndex] = ChatMessage(role: "Assistant", text: existing + pending, imageData: nil, audioFileName: nil)
                        }
                    }
                )
                // Flush any remaining buffered tokens after generation completes.
                let remaining = streamBuffer.flushAll()
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
                    generationTask = nil
                }
            } catch {
                await MainActor.run {
                    isInitializing = false
                    errorText = String(describing: error)
                    if assistantIndex < messages.count {
                        messages[assistantIndex] = ChatMessage(role: "Assistant", text: "", imageData: nil, audioFileName: nil)
                    }
                    isRunning = false
                    generationTask = nil
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

    private func restoreDefaults() {
        if let modelPath = UserDefaults.standard.string(forKey: SharedSelection.llmModelPath),
           let tokPath = UserDefaults.standard.string(forKey: SharedSelection.llmTokenizerPath) {
            let model = URL(fileURLWithPath: modelPath)
            let tok = URL(fileURLWithPath: tokPath)
            let isSupportedModel = model.pathExtension.lowercased() == "cellm" || model.pathExtension.lowercased() == "cellmd"
            if isSupportedModel,
               FileManager.default.fileExists(atPath: model.path),
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
                ChatMessage(role: "Assistant", text: "Hi. You can chat with text or attach an image.", imageData: nil, audioFileName: nil)
            ]
        }
        infoText = "Text uses LLM model/tokenizer. Image uses VLM."
    }

    private func persistSharedSelection() {
        if let model = llmModelURL?.path {
            UserDefaults.standard.set(model, forKey: SharedSelection.llmModelPath)
        }
        if let tok = llmTokenizerURL?.path {
            UserDefaults.standard.set(tok, forKey: SharedSelection.llmTokenizerPath)
        }
    }

    private func selectGemma4E4BPreset() {
        // Kept for compatibility with older UI hooks; maps to the native Gemma 4 preset.
        selectGemmaPreset()
    }

    private func selectGemmaPreset(silentOnMissing: Bool = false) {
        let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma4E2BFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "gemma-4-E2B-it-int4-aggr-v5.cellmd")
        let tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma4E2BTokenizerFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "tokenizer.json")
        if let model, let tok {

            llmModelURL = model
            llmTokenizerURL = tok
            selectedSampleLabel = "Gemma-4-E2B-IT (int4 aggr v5)"
            errorText = nil
        } else if !silentOnMissing {
            errorText = "Gemma 4 files not found in Documents/samples. Download Gemma 4 model + tokenizer from LLM tab."
        }
    }

    private func selectQwenPreset(silentOnMissing: Bool = false) {
        let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen25FileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "qwen2.5-0.5b-int8-v1.cellm")
        let tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen25TokenizerFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "tokenizer.json")
        if let model, let tok {
            llmModelURL = model
            llmTokenizerURL = tok
            selectedSampleLabel = "Qwen2.5-0.5B (int8 v1)"
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

    private func selectBonsaiPreset(silentOnMissing: Bool = false) {
        let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.bonsai1B1BitFileName)
        let tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.bonsai1B1BitTokenizerFileName)
        if let model, let tok {
            llmModelURL = model
            llmTokenizerURL = tok
            selectedSampleLabel = "Bonsai 1.7B (1-Bit)"
            errorText = nil
        } else if !silentOnMissing {
            errorText = "Bonsai files not found in Documents/samples. Download Bonsai from Model Hub."
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

    private func initializeEngine() {
        guard let modelURL = llmModelURL, let tokURL = llmTokenizerURL else { return }

        // Skip if we already have a cached engine for this model + backend.
        if cachedEngine != nil,
           cachedEngineModelURL == modelURL,
           cachedEngineBackend == selectedBackend {
            return
        }

        initTask?.cancel()
        initTask = Task {
            await MainActor.run { isInitializing = true }
            do {
                let backend = await MainActor.run { scenePhase == .active ? selectedBackend : .cpu }
                let tok = try CellmTokenizer(tokenizerURL: tokURL)
                let eng = try CellmEngine(
                    modelURL: modelURL,
                    tokenizer: tok,
                    backend: backend
                )
                await MainActor.run {
                    cachedEngine = eng
                    cachedTokenizer = tok
                    cachedEngineModelURL = modelURL
                    cachedEngineBackend = backend
                    errorText = nil
                }
            } catch {
                await MainActor.run {
                    errorText = "Model init failed: \(String(describing: error))"
                }
            }
            await MainActor.run {
                isInitializing = false
                initTask = nil
            }
        }
    }

    private func invalidateCachedEngine() {
        cachedEngine = nil
        cachedTokenizer = nil
        cachedEngineModelURL = nil
        cachedEngineBackend = nil
    }
}

// MARK: - Generation Settings Sheet

struct GenerationSettingsSheet: View {
    @Binding var temperature: Double
    @Binding var maxNewTokens: Int
    @Binding var selectedBackend: CellmBackend
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Backend", selection: $selectedBackend) {
                        ForEach(CellmBackend.allCases) { b in
                            Text(b.label).tag(b)
                        }
                    }
                    .pickerStyle(.segmented)
                } header: {
                    Text("Inference Backend")
                } footer: {
                    Text("Metal uses the GPU for faster inference. CPU is a fallback.")
                }

                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("Temperature", systemImage: "thermometer.medium")
                            Spacer()
                            Text(String(format: "%.2f", temperature))
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $temperature, in: 0...2, step: 0.05)
                            .tint(.orange)
                        Text(temperatureHint)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text("Sampling")
                }

                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("Max Tokens", systemImage: "textformat.123")
                            Spacer()
                            Text("\(maxNewTokens)")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        Slider(
                            value: Binding(
                                get: { Double(maxNewTokens) },
                                set: { maxNewTokens = Int($0.rounded()) }
                            ),
                            in: 50...512,
                            step: 10
                        )
                        .tint(.blue)
                        Text("Maximum tokens the model will generate per response.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text("Generation Length")
                }

                Section {
                    Button("Reset to Defaults") {
                        withAnimation {
                            temperature = 0.2
                            maxNewTokens = 200
                            selectedBackend = .metal
                        }
                    }
                    .foregroundStyle(.red)
                }
            }
            .navigationTitle("Generation Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium])
        .presentationDragIndicator(.visible)
    }

    private var temperatureHint: String {
        switch temperature {
        case 0..<0.05:   return "Greedy — deterministic, most accurate"
        case 0.05..<0.4: return "Focused — low creativity, factual"
        case 0.4..<0.8:  return "Balanced (default)"
        case 0.8..<1.2:  return "Creative — more variety"
        default:         return "Very creative — may produce unexpected output"
        }
    }
}

// MARK: - Stream Buffer (throttles UI updates)

/// Accumulates streaming token pieces and flushes them to the UI at most
/// once per `interval` seconds. This prevents the rapid-fire SwiftUI state
/// mutations that cause CoreGraphics NaN layout errors on iPhone.
private final class StreamBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer: String = ""
    private var lastFlush: Date = .distantPast
    private let interval: TimeInterval = 0.08 // 80ms

    func append(_ piece: String) {
        lock.lock()
        buffer += piece
        lock.unlock()
    }

    /// Returns accumulated text if enough time has passed since the last flush.
    /// Returns nil if the throttle window hasn't elapsed yet.
    func flushIfReady() -> String? {
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        guard now.timeIntervalSince(lastFlush) >= interval else { return nil }
        let result = buffer
        buffer = ""
        lastFlush = now
        return result
    }

    /// Returns any remaining buffered text (call at end of generation).
    func flushAll() -> String {
        lock.lock()
        defer { lock.unlock() }
        let result = buffer
        buffer = ""
        lastFlush = Date()
        return result
    }
}
