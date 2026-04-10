import SwiftUI
import UniformTypeIdentifiers
import Dispatch

struct LLMView: View {
    private enum SharedSelection {
        static let llmModelPath = "cellm.shared.llm.model.path"
        static let llmTokenizerPath = "cellm.shared.llm.tokenizer.path"
    }

    private struct ChatTurn: Identifiable {
        let id = UUID()
        let role: String
        let text: String
    }

    private final class LoadedRuntime {
        let key: String
        let tokenizer: CellmTokenizer
        let engine: CellmEngine
        let initLines: [String]

        init(key: String, tokenizer: CellmTokenizer, engine: CellmEngine, initLines: [String]) {
            self.key = key
            self.tokenizer = tokenizer
            self.engine = engine
            self.initLines = initLines
        }
    }

    private final class RuntimeCache {
        static let shared = RuntimeCache()
        private let lock = NSLock()
        private var loaded: LoadedRuntime?
        private init() {}

        func clear() {
            lock.lock()
            loaded = nil
            lock.unlock()
        }

        func getOrCreate(
            key: String,
            build: () throws -> LoadedRuntime
        ) throws -> (runtime: LoadedRuntime, created: Bool) {
            lock.lock()
            if let loaded, loaded.key == key {
                lock.unlock()
                return (loaded, false)
            }
            lock.unlock()

            let newRuntime = try build()

            lock.lock()
            loaded = newRuntime
            lock.unlock()
            return (newRuntime, true)
        }
    }

    private final class StreamTelemetry {
        struct Snapshot {
            let flushes: Int
            let chars: Int
            let elapsedMs: Double
        }

        private let lock = NSLock()
        private let startedAt = Date()
        private var flushes = 0
        private var chars = 0

        func record(piece: String) -> Snapshot {
            lock.lock()
            flushes += 1
            chars += piece.count
            let snapshot = Snapshot(
                flushes: flushes,
                chars: chars,
                elapsedMs: Date().timeIntervalSince(startedAt) * 1000.0
            )
            lock.unlock()
            return snapshot
        }
    }

    private enum GenerationPreset: String, CaseIterable, Identifiable {
        case chat = "Chat"
        case balanced = "Balanced"
        case strict = "Strict"

        var id: String { rawValue }

        var temperature: Float {
            switch self {
            case .chat: return 0.1
            case .balanced: return 0.35
            case .strict: return 0.0
            }
        }

        var topK: UInt32 {
            switch self {
            case .chat: return 20
            case .balanced: return 40
            case .strict: return 1
            }
        }

        var repeatPenalty: Float {
            switch self {
            case .chat: return 1.14
            case .balanced: return 1.08
            case .strict: return 1.0
            }
        }

        var repeatWindow: UInt32 {
            switch self {
            case .chat: return 128
            case .balanced: return 64
            case .strict: return 0
            }
        }

        var maxTokens: Int {
            switch self {
            case .chat: return 96
            case .balanced: return 128
            case .strict: return 64
            }
        }
    }

    @State private var modelURL: URL?
    @State private var tokenizerURL: URL?
    @State private var prompt: String = "Return exactly one uppercase letter: R"
    @State private var output: String = ""
    @State private var runDiagnostics: String = ""
    @State private var streamFlushes: Int = 0
    @State private var streamChars: Int = 0
    @State private var streamElapsedMs: Double = 0
    @State private var isRunning: Bool = false
    @State private var errorText: String?
    @State private var selectedBackend: CellmBackend = .metal
    @State private var selectedThermalLevel: CellmThermalLevel = .nominal
    @State private var activeBackend: String = ""
    @State private var downloadStatus: String = ""
    @State private var downloadProgress: Double = 0.0
    @State private var currentDownloadFile: String = ""
    @State private var currentDownloadSizeText: String = ""
    @State private var isDownloading: Bool = false
    @State private var backendWarning: String?
    @State private var selectedPreset: GenerationPreset = .chat
    @State private var selectedSampleLabel: String = ""
    @State private var chatInput: String = ""
    @State private var chatTurns: [ChatTurn] = []
    @State private var runtimeStatus: String = "Runtime: not loaded"
    @State private var showSettings: Bool = false
    @State private var showAdvancedDownloads: Bool = false

    @State private var showModelPicker = false
    @State private var showTokenizerPicker = false

    private enum QwenVariant {
        case none
        case stable
        case experimental
    }

    private var isQwenSelected: Bool {
        (modelURL?.lastPathComponent.lowercased().contains("qwen") ?? false) ||
        (tokenizerURL?.lastPathComponent.lowercased().contains("qwen") ?? false)
    }
    
    private var isGemmaSelected: Bool {
        (modelURL?.lastPathComponent.lowercased().contains("gemma") ?? false) ||
        (tokenizerURL?.lastPathComponent.lowercased().contains("gemma") ?? false)
    }

    private var qwenVariant: QwenVariant {
        guard isQwenSelected else { return .none }
        let modelName = modelURL?.lastPathComponent.lowercased() ?? ""
        if modelName == "qwen3.5-0.8b.cellm" || modelName == "qwen3.5-0.8b-int8.cellm" {
            return .stable
        }
        if modelName.contains("qwen") {
            return .experimental
        }
        return .experimental
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                filesCard
                storageCard
                promptCard
                conversationCard
                settingsSection
                generateButton
                if let errorText { errorCard(errorText) }
                outputCard
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .padding(.bottom, 24)
            .contentShape(Rectangle())
            .onTapGesture {
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Color(.systemGroupedBackground).ignoresSafeArea())
        .sheet(isPresented: $showModelPicker) {
            DocumentPicker(allowed: [.data]) { pickedURL in
                if let saved = persistPickedFile(pickedURL, subdir: "picked/models") {
                    modelURL = saved
                    selectedSampleLabel = "Custom"
                    downloadStatus = "Loaded local model: \(saved.lastPathComponent)"
                }
            }
        }
        .sheet(isPresented: $showTokenizerPicker) {
            DocumentPicker(allowed: [.json]) { pickedURL in
                if let saved = persistPickedFile(pickedURL, subdir: "picked/tokenizers") {
                    tokenizerURL = saved
                    selectedSampleLabel = "Custom"
                    downloadStatus = "Loaded local tokenizer: \(saved.lastPathComponent)"
                }
            }
        }
        .onAppear {
            restoreAssets()
        }
        .onChange(of: modelURL) { _ in persistSharedSelection() }
        .onChange(of: tokenizerURL) { _ in persistSharedSelection() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("cellm LLM")
                .font(.system(size: 38, weight: .bold))
            Text("Run local text generation on-device")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 4)
    }

    private var filesCard: some View {
        card("Model Management") {
            Text("Manage on-device LLMs and download recommended presets.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Text("Recommended models")
                .font(.subheadline)
                .fontWeight(.semibold)

            modelDownloadCard(
                title: "Gemma3-1B-IT (int8)",
                subtitle: "Best on-device",
                detail: "Native cellm model for iOS/Android. No LiteRT proxy runtime required.",
                sizeText: "~1.2 GB",
                isInstalled: hasGemma3Sample,
                isSelected: selectedSampleLabel == "Gemma3-1B-IT (int8)",
                onDownload: { downloadGemmaSampleAssets() },
                onUseInstalled: {
                    selectInstalledSample(
                        modelFile: DemoAssetLinks.gemma3FileName,
                        tokenizerFile: DemoAssetLinks.gemma3TokenizerFileName,
                        label: "Gemma3-1B-IT (int8)"
                    )
                }
            )

            modelDownloadCard(
                title: "Gemma-4-2P3B-IT (LiteRT)",
                subtitle: "Proxy bundle (desktop)",
                detail: "This build is Python-free on iOS. LiteRT proxy bundles are not runnable in-app; use native .cellm/.cellmd models.",
                sizeText: "~2.45 GB",
                isInstalled: hasGemma4LiteRtSample,
                isSelected: selectedSampleLabel == "Gemma-4-2P3B-IT (LiteRT)",
                onDownload: { downloadGemma4LiteRtAssets() },
                onUseInstalled: {
                    selectInstalledSample(
                        modelFile: DemoAssetLinks.gemma42p3bFileName,
                        tokenizerFile: DemoAssetLinks.gemma42p3bTokenizerFileName,
                        label: "Gemma-4-2P3B-IT (LiteRT)"
                    )
                }
            )

            modelDownloadCard(
                title: "Qwen3.5-0.8B",
                subtitle: "Fast baseline",
                detail: "Stable small model with good speed/quality tradeoff.",
                sizeText: "~1.6 GB",
                isInstalled: hasQwenStableSample,
                isSelected: selectedSampleLabel == "Qwen3.5 (stable)",
                onDownload: { downloadQwenSampleAssets() },
                onUseInstalled: {
                    selectInstalledSample(
                        modelFile: DemoAssetLinks.qwen35StableFileName,
                        tokenizerFile: DemoAssetLinks.qwen35TokenizerFileName,
                        label: "Qwen3.5 (stable)"
                    )
                }
            )

            modelDownloadCard(
                title: "SmolLM2-135M",
                subtitle: "Smallest",
                detail: "Tiny model for quick local tests and integration checks.",
                sizeText: "~300 MB",
                isInstalled: hasSmolLMSample,
                isSelected: selectedSampleLabel == "SmolLM2",
                onDownload: { downloadSmolLMSampleAssets() },
                onUseInstalled: {
                    selectInstalledSample(
                        modelFile: DemoAssetLinks.smollm2FileName,
                        tokenizerFile: DemoAssetLinks.smollm2TokenizerFileName,
                        label: "SmolLM2"
                    )
                }
            )

            if !downloadStatus.isEmpty {
                Text(downloadStatus).font(.footnote).foregroundStyle(.secondary)
            }
            if isDownloading {
                ProgressView(value: safeUnitProgress(downloadProgress))
                    .progressViewStyle(.linear)
                Text("\(Int((safeUnitProgress(downloadProgress) * 100.0).rounded()))%")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                if !currentDownloadFile.isEmpty {
                    Text("File: \(currentDownloadFile)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if !currentDownloadSizeText.isEmpty {
                    Text(currentDownloadSizeText)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            if !selectedSampleLabel.isEmpty {
                Text("Selected sample: \(selectedSampleLabel)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    showAdvancedDownloads.toggle()
                }
            } label: {
                HStack {
                    Text(showAdvancedDownloads ? "Hide Advanced Actions" : "Show Advanced Actions")
                        .font(.footnote)
                        .fontWeight(.semibold)
                    Spacer()
                    Image(systemName: showAdvancedDownloads ? "chevron.up" : "chevron.down")
                        .font(.footnote.bold())
                }
                .padding(10)
                .background(Color(.systemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 10))
            }
            .buttonStyle(.plain)

            if showAdvancedDownloads {
                actionButton(modelURL == nil ? "Pick .cellm model" : modelURL!.lastPathComponent, icon: "externaldrive") { showModelPicker = true }
                actionButton(tokenizerURL == nil ? "Pick tokenizer.json" : tokenizerURL!.lastPathComponent, icon: "doc.text") { showTokenizerPicker = true }
                actionButton(isDownloading ? "Downloading sample files…" : "Download Qwen stable model only (~1.6 GB)", icon: "shippingbox", disabled: isDownloading || isRunning) {
                    downloadQwenModelOnly()
                }
                actionButton(isDownloading ? "Downloading sample files…" : "Download Qwen compact model only (~361 MB, experimental)", icon: "shippingbox", disabled: isDownloading || isRunning) {
                    downloadQwenCompactModelOnly()
                }
                actionButton(isDownloading ? "Downloading sample files…" : "Download Qwen tokenizer JSON only", icon: "arrow.down.doc", disabled: isDownloading || isRunning) {
                    downloadQwenTokenizerOnly()
                }
                actionButton(isDownloading ? "Downloading sample files…" : "Download SmolLM model only", icon: "shippingbox", disabled: isDownloading || isRunning) {
                    downloadSmolLMModelOnly()
                }
                actionButton(isDownloading ? "Downloading sample files…" : "Download SmolLM tokenizer JSON only", icon: "arrow.down.doc", disabled: isDownloading || isRunning) {
                    downloadSmolLMTokenizerOnly()
                }
                actionButton("Run Qwen Smoke Test", icon: "bolt.circle", disabled: isRunning || modelURL == nil || tokenizerURL == nil) {
                    runQwenSmokeTest()
                }
                actionButton("Run Scheduler Smoke (Suspend/Resume)", icon: "pause.circle", disabled: isRunning || modelURL == nil || tokenizerURL == nil) {
                    runSchedulerSmokeTest()
                }
            }
        }
    }

    private var hasGemma3Sample: Bool {
        hasSampleFiles(
            modelFile: DemoAssetLinks.gemma3FileName,
            tokenizerFile: DemoAssetLinks.gemma3TokenizerFileName,
            tokenizerConfigFile: DemoAssetLinks.gemma3TokenizerConfigFileName
        )
    }

    private var hasGemma4LiteRtSample: Bool {
        hasSampleFiles(
            modelFile: DemoAssetLinks.gemma42p3bFileName,
            tokenizerFile: DemoAssetLinks.gemma42p3bTokenizerFileName,
            tokenizerConfigFile: DemoAssetLinks.gemma42p3bTokenizerConfigFileName
        )
    }

    private var hasQwenStableSample: Bool {
        hasSampleFiles(
            modelFile: DemoAssetLinks.qwen35StableFileName,
            tokenizerFile: DemoAssetLinks.qwen35TokenizerFileName,
            tokenizerConfigFile: DemoAssetLinks.qwen35TokenizerConfigFileName
        )
    }

    private var hasSmolLMSample: Bool {
        hasSampleFiles(
            modelFile: DemoAssetLinks.smollm2FileName,
            tokenizerFile: DemoAssetLinks.smollm2TokenizerFileName,
            tokenizerConfigFile: DemoAssetLinks.smollm2TokenizerConfigFileName
        )
    }

    private func hasSampleFiles(modelFile: String, tokenizerFile: String, tokenizerConfigFile: String) -> Bool {
        RemoteAssets.existingDocumentsFile(fileName: modelFile) != nil &&
        RemoteAssets.existingDocumentsFile(fileName: tokenizerFile) != nil &&
        RemoteAssets.existingDocumentsFile(fileName: tokenizerConfigFile) != nil
    }

    private func selectInstalledSample(modelFile: String, tokenizerFile: String, label: String) {
        guard let model = RemoteAssets.existingDocumentsFile(fileName: modelFile),
              let tokenizer = RemoteAssets.existingDocumentsFile(fileName: tokenizerFile) else {
            errorText = "Sample files are not installed yet."
            return
        }
        modelURL = model
        tokenizerURL = tokenizer
        selectedSampleLabel = label
        downloadStatus = "Using installed files: \(model.lastPathComponent), \(tokenizer.lastPathComponent)"
    }

    private func modelDownloadCard(
        title: String,
        subtitle: String,
        detail: String,
        sizeText: String,
        isInstalled: Bool,
        isSelected: Bool,
        onDownload: @escaping () -> Void,
        onUseInstalled: @escaping () -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(subtitle.uppercased())
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundStyle(.orange)
                    Text(title)
                        .font(.headline)
                }
                Spacer()
                Text(isInstalled ? "Installed" : "Not installed")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(isInstalled ? .green : .secondary)
            }
            Text(sizeText)
                .font(.footnote)
                .foregroundStyle(.secondary)
            Text(detail)
                .font(.footnote)
                .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                Button(isDownloading ? "Downloading…" : "Download") { onDownload() }
                    .buttonStyle(.borderedProminent)
                    .disabled(isDownloading || isRunning)
                Button("Use Installed") { onUseInstalled() }
                    .buttonStyle(.bordered)
                    .disabled(!isInstalled || isRunning || isDownloading)
            }
            if isSelected {
                Text("Currently selected")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(.blue)
            }
        }
        .padding(12)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var promptCard: some View {
        card("Prompt") {
            TextEditor(text: $prompt)
                .frame(minHeight: 110)
                .padding(8)
                .background(Color(.systemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private var conversationCard: some View {
        card("Conversation") {
            HStack(spacing: 8) {
                Button("Load Runtime") { preloadRuntime() }
                    .buttonStyle(.bordered)
                    .disabled(isRunning || modelURL == nil || tokenizerURL == nil)
                Button("Unload Runtime") { unloadRuntime() }
                    .buttonStyle(.bordered)
                    .disabled(isRunning)
                Button("Clear Chat") { chatTurns.removeAll() }
                    .buttonStyle(.bordered)
                    .disabled(isRunning || chatTurns.isEmpty)
            }
            Text(runtimeStatus)
                .font(.footnote)
                .foregroundStyle(.secondary)
            if chatTurns.isEmpty {
                Text("No conversation yet.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(chatTurns) { turn in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(turn.role)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text(turn.text)
                            .font(.body)
                    }
                    .padding(10)
                    .background(Color(.systemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }
            TextField("Type a message…", text: $chatInput)
                .textFieldStyle(.roundedBorder)
            Button(isRunning ? "Sending…" : "Send Message") { sendConversationMessage() }
                .buttonStyle(.borderedProminent)
                .disabled(
                    isRunning ||
                    modelURL == nil ||
                    tokenizerURL == nil ||
                    chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                )
        }
    }

    private var storageCard: some View {
        card("Storage") {
            if modelURL == nil && tokenizerURL == nil {
                Text("No local sample files found.")
                    .foregroundStyle(.secondary)
                    .font(.footnote)
            } else {
                if let modelSize = RemoteAssets.fileSizeString(url: modelURL) {
                    Text("Model: \(modelSize)").font(.footnote).foregroundStyle(.secondary)
                }
                if let tokSize = RemoteAssets.fileSizeString(url: tokenizerURL) {
                    Text("Tokenizer: \(tokSize)").font(.footnote).foregroundStyle(.secondary)
                }
            }

            HStack(spacing: 10) {
                Button("Re-download") { forceRedownload() }
                    .buttonStyle(.bordered)
                    .disabled(isDownloading || isRunning)
                Button("Delete local files") { clearLocalFiles() }
                    .buttonStyle(.bordered)
                    .disabled(isDownloading || isRunning || (modelURL == nil && tokenizerURL == nil))
            }
        }
    }

    private var backendCard: some View {
        card("Backend") {
            Picker("Requested", selection: $selectedBackend) {
                ForEach(CellmBackend.allCases) { backend in
                    Text(backend.label).tag(backend)
                }
            }
            .pickerStyle(.segmented)
            Picker("Thermal", selection: $selectedThermalLevel) {
                ForEach(CellmThermalLevel.allCases) { level in
                    Text(level.label).tag(level)
                }
            }
            .pickerStyle(.segmented)
            if !activeBackend.isEmpty {
                Text("Active: \(activeBackend)").font(.footnote).foregroundStyle(.secondary)
            }
            if let backendWarning {
                Text(backendWarning).font(.footnote).foregroundStyle(.orange)
            }
            Text("Note: Backend selection is strict in this build (no automatic CPU fallback).")
                .font(.footnote)
                .foregroundStyle(.secondary)
            if isQwenSelected {
                switch qwenVariant {
                case .stable:
                    Text("Qwen parity-fixed model selected. Recommended: qwen3.5-0.8b.cellm or qwen3.5-0.8b-int8.cellm.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                case .experimental:
                    Text("This Qwen model is experimental on current mobile runner (for example int8/int4) and may produce degraded output. Use qwen3.5-0.8b.cellm for best quality.")
                        .font(.footnote)
                        .foregroundStyle(.orange)
                case .none:
                    EmptyView()
                }
            }
        }
    }

    private var presetCard: some View {
        card("Generation") {
            Picker("Preset", selection: $selectedPreset) {
                ForEach(GenerationPreset.allCases) { preset in
                    Text(preset.rawValue).tag(preset)
                }
            }
            .pickerStyle(.segmented)
            Text("Preset controls temperature/repetition to improve quality.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private var settingsSection: some View {
        VStack(spacing: 12) {
            Button {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                    showSettings.toggle()
                }
            } label: {
                HStack {
                    Label(showSettings ? "Hide Settings" : "Generation & Backend Settings", systemImage: "slider.horizontal.3")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Image(systemName: showSettings ? "chevron.up" : "chevron.down")
                        .font(.caption.bold())
                        .foregroundStyle(.secondary)
                }
                .padding(14)
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 16))
            }
            .buttonStyle(.plain)

            if showSettings {
                VStack(spacing: 16) {
                    presetCard
                    backendCard
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }

    private var generateButton: some View {
        Button(isRunning ? "Generating…" : "Generate") { run() }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .frame(maxWidth: .infinity)
            .disabled(isRunning || modelURL == nil || tokenizerURL == nil || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    private func errorCard(_ text: String) -> some View {
        card("Error") {
            Text(text).foregroundStyle(.red)
        }
    }

    private var outputCard: some View {
        card("Output") {
            if streamFlushes > 0 || isRunning {
                streamHealthView
            }
            if !runDiagnostics.isEmpty {
                Text(runDiagnostics)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(.bottom, 6)
            }
            Text(output.isEmpty ? "No output yet." : output)
                .font(.body)
                .lineSpacing(4)
                .foregroundStyle(output.isEmpty ? .secondary : .primary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(Color(.systemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private var streamHealthView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Stream Health")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)

            let seconds = max(0.001, streamElapsedMs / 1000.0)
            let flushRate = Double(streamFlushes) / seconds
            let charRate = Double(streamChars) / seconds

            HStack(spacing: 8) {
                metricBadge("flushes", "\(streamFlushes)")
                metricBadge("flush/s", String(format: "%.2f", flushRate))
                metricBadge("chars/s", String(format: "%.1f", charRate))
                metricBadge("elapsed", String(format: "%.1fs", seconds))
            }
        }
        .padding(10)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func metricBadge(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title.uppercased())
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.primary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func card<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.caption)
                .foregroundStyle(.secondary)
                .fontWeight(.semibold)
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private func actionButton(_ title: String, icon: String, disabled: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: icon).foregroundStyle(.blue)
                Text(title)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(12)
            .background(Color(.systemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .opacity(disabled ? 0.6 : 1.0)
    }

    private func run(
        exerciseSuspendResume: Bool = false,
        maxTokensOverride: Int? = nil,
        promptOverride: String? = nil,
        onFinalText: ((String) -> Void)? = nil
    ) {
        // Dismiss keyboard
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)

        errorText = nil
        output = ""
        runDiagnostics = ""
        streamFlushes = 0
        streamChars = 0
        streamElapsedMs = 0
        backendWarning = nil
        resolveStaleSelectionsIfNeeded()
        guard var effectiveModelURL = modelURL, var effectiveTokenizerURL = tokenizerURL else { return }

        guard FileManager.default.fileExists(atPath: effectiveModelURL.path) else {
            errorText = "Selected model file is missing. Re-pick the model or download sample assets."
            return
        }
        guard FileManager.default.fileExists(atPath: effectiveTokenizerURL.path) else {
            errorText = "Selected tokenizer file is missing. Re-pick tokenizer.json or download tokenizer assets."
            return
        }
        let promptText = promptOverride ?? prompt
        let backend = selectedBackend
        let thermal = selectedThermalLevel
        let shouldForceStrictForExperimentalQwen = (qwenVariant == .experimental && selectedPreset != .strict)
        let useGemmaMetalStabilitySampling = (selectedBackend == .metal && isGemmaSelected)
        let preset = shouldForceStrictForExperimentalQwen ? GenerationPreset.strict : selectedPreset
        let isExperimentalQwen = (qwenVariant == .experimental)
        let requestedMaxTokens = maxTokensOverride ?? preset.maxTokens
        let effectiveMaxTokens = isExperimentalQwen ? min(requestedMaxTokens, 4) : requestedMaxTokens
        let tokensPerBlock: UInt32 = 8
        let totalBlocks: UInt32 = isExperimentalQwen ? 24 : 48
        let runtimeTopK: UInt32 = useGemmaMetalStabilitySampling ? max(preset.topK, 40) : preset.topK
        let runtimeTemperature: Float = useGemmaMetalStabilitySampling ? max(preset.temperature, 0.35) : preset.temperature
        let runtimeRepeatPenalty: Float = useGemmaMetalStabilitySampling ? max(preset.repeatPenalty, 1.18) : preset.repeatPenalty
        let runtimeRepeatWindow: UInt32 = useGemmaMetalStabilitySampling ? max(preset.repeatWindow, 128) : preset.repeatWindow

        isRunning = true
        Task.detached(priority: .userInitiated) {
            do {
                var initLogLines: [String] = []
                let streamTelemetry = StreamTelemetry()
                if backend == .metal, let metalErr = CellmFFI.metalSmokeError() {
                    throw CellmError.message("Metal requested but unavailable: \(metalErr)")
                }
                let runtimeKey = Self.runtimeCacheKey(
                    modelURL: effectiveModelURL,
                    tokenizerURL: effectiveTokenizerURL,
                    backend: backend,
                    tokensPerBlock: tokensPerBlock,
                    totalBlocks: totalBlocks,
                    topK: runtimeTopK,
                    temperature: runtimeTemperature,
                    repeatPenalty: runtimeRepeatPenalty,
                    repeatWindow: runtimeRepeatWindow
                )
                let runtimeResult = try RuntimeCache.shared.getOrCreate(key: runtimeKey) {
                    let initStart = Date()
                    var lines: [String] = []

                    let afterPreflight = Date()
                    lines.append(
                        String(
                            format: "init_preflight=%.1fms",
                            afterPreflight.timeIntervalSince(initStart) * 1000.0
                        )
                    )

                    let tokStart = Date()
                    let tok = try CellmTokenizer(tokenizerURL: effectiveTokenizerURL)
                    let tokEnd = Date()
                    lines.append(
                        String(
                            format: "init_tokenizer=%.1fms",
                            tokEnd.timeIntervalSince(tokStart) * 1000.0
                        )
                    )

                    let engineStart = Date()
                    let eng = try CellmEngine(
                        modelURL: effectiveModelURL,
                        tokenizer: tok,
                        tokensPerBlock: tokensPerBlock,
                        totalBlocks: totalBlocks,
                        topK: runtimeTopK,
                        temperature: runtimeTemperature,
                        repeatPenalty: runtimeRepeatPenalty,
                        repeatWindow: runtimeRepeatWindow,
                        seed: UInt64(Date().timeIntervalSince1970 * 1000.0),
                        backend: backend
                    )
                    let engineEnd = Date()
                    lines.append(
                        String(
                            format: "init_engine=%.1fms",
                            engineEnd.timeIntervalSince(engineStart) * 1000.0
                        )
                    )
                    lines.append(
                        String(
                            format: "init_total=%.1fms",
                            engineEnd.timeIntervalSince(initStart) * 1000.0
                        )
                    )
                    lines.append("requested_backend=\(backend.label.lowercased()) active_backend=\(eng.activeBackend)")
                    return LoadedRuntime(key: runtimeKey, tokenizer: tok, engine: eng, initLines: lines)
                }

                let eng = runtimeResult.runtime.engine
                if runtimeResult.created {
                    initLogLines.append(contentsOf: runtimeResult.runtime.initLines)
                    print("[CellmDebug] runtime cache: created key=\(runtimeKey)")
                } else {
                    initLogLines.append("init_runtime=reused")
                    initLogLines.append("requested_backend=\(backend.label.lowercased()) active_backend=\(eng.activeBackend)")
                    print("[CellmDebug] runtime cache: reused key=\(runtimeKey)")
                }
                try eng.resetSession()
                initLogLines.append("session=reset")

                let text = try eng.generate(
                    prompt: promptText,
                    maxNewTokens: effectiveMaxTokens,
                    thermalLevel: thermal,
                    exerciseSuspendResume: exerciseSuspendResume,
                    onToken: { piece in
                        let snapshot = streamTelemetry.record(piece: piece)
                        Task { @MainActor in
                            self.output += piece
                            self.streamFlushes = snapshot.flushes
                            self.streamChars = snapshot.chars
                            self.streamElapsedMs = snapshot.elapsedMs
                            self.runDiagnostics = formatLiveDiagnostics(
                                lines: initLogLines,
                                flushes: snapshot.flushes,
                                chars: snapshot.chars,
                                elapsedMs: snapshot.elapsedMs
                            )
                        }
                    }
                )
                await MainActor.run {
                    self.output = prettyOutput(text)
                    onFinalText?(text)
                    var finalInitLogLines = initLogLines
                    if let stats = eng.lastGenerationStats {
                        finalInitLogLines.append("prompt_style=\(eng.lastPromptStyle)")
                        if !eng.lastDebugTrace.isEmpty {
                            finalInitLogLines.append(contentsOf: eng.lastDebugTrace.map { "trace \($0)" })
                        }
                        self.runDiagnostics = formatInitDiagnostics(lines: finalInitLogLines, stats: stats)
                    }
                    self.activeBackend = eng.activeBackend
                    var warnings: [String] = []
                    if backend == .metal && !eng.activeBackend.lowercased().contains("metal") {
                        warnings.append("Metal requested but active backend is \(eng.activeBackend).")
                    }
                    if shouldForceStrictForExperimentalQwen {
                        warnings.append("Strict preset auto-applied because selected Qwen variant is experimental.")
                    }
                    if useGemmaMetalStabilitySampling {
                        warnings.append("Gemma Metal stability sampling applied (temp/top-k/repetition tuned) to reduce decode loops.")
                    }
                    if isExperimentalQwen && effectiveMaxTokens < requestedMaxTokens {
                        warnings.append("Experimental Qwen token cap applied (\(effectiveMaxTokens) max) to avoid long looped runs.")
                    }
                    if backend == .metal && isGemmaSelected {
                        warnings.append("Gemma Metal is enabled, but output quality parity is still being tuned in this build.")
                    }
                    warnings.append("Mobile KV cache tuned for latency (block=\(tokensPerBlock), total=\(totalBlocks)).")
                    self.backendWarning = warnings.isEmpty ? nil : warnings.joined(separator: " ")
                    self.runtimeStatus = runtimeResult.created
                        ? "Runtime: loaded and ready (\(eng.activeBackend))"
                        : "Runtime: reused (\(eng.activeBackend))"
                    self.isRunning = false
                }
            } catch {
                await MainActor.run {
                    let modelExists = FileManager.default.fileExists(atPath: effectiveModelURL.path)
                    let tokExists = FileManager.default.fileExists(atPath: effectiveTokenizerURL.path)
                    let raw = String(describing: error)
                    self.errorText = """
                    \(raw)
                    model=\(effectiveModelURL.lastPathComponent) exists=\(modelExists)
                    tokenizer=\(effectiveTokenizerURL.lastPathComponent) exists=\(tokExists)
                    requested_backend=\(backend.label.lowercased()) active_backend=\(self.activeBackend.isEmpty ? "n/a" : self.activeBackend)
                    preset=\(preset.rawValue)
                    max_tokens=\(effectiveMaxTokens)
                    """
                    self.runtimeStatus = "Runtime: error"
                    self.isRunning = false
                }
            }
        }
    }

    nonisolated private static func runtimeCacheKey(
        modelURL: URL,
        tokenizerURL: URL,
        backend: CellmBackend,
        tokensPerBlock: UInt32,
        totalBlocks: UInt32,
        topK: UInt32,
        temperature: Float,
        repeatPenalty: Float,
        repeatWindow: UInt32
    ) -> String {
        "\(modelURL.path)|\(tokenizerURL.path)|\(backend.rawValue)|\(tokensPerBlock)|\(totalBlocks)|\(topK)|\(temperature)|\(repeatPenalty)|\(repeatWindow)"
    }

    private func preloadRuntime() {
        runtimeStatus = "Runtime: loading..."
        run(maxTokensOverride: 1, promptOverride: "Return exactly one uppercase letter: R")
    }

    private func unloadRuntime() {
        RuntimeCache.shared.clear()
        runtimeStatus = "Runtime: unloaded"
    }

    private func sendConversationMessage() {
        let msg = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !msg.isEmpty else { return }
        chatTurns.append(ChatTurn(role: "User", text: msg))
        chatInput = ""
        let transcript = conversationPrompt(from: chatTurns, maxTurns: 6, maxChars: 1200)
        run(maxTokensOverride: 48, promptOverride: transcript) { finalText in
            let assistantText = prettyOutput(finalText).trimmingCharacters(in: .whitespacesAndNewlines)
            if !assistantText.isEmpty {
                chatTurns.append(ChatTurn(role: "Assistant", text: assistantText))
            }
        }
    }

    private func conversationPrompt(from turns: [ChatTurn], maxTurns: Int, maxChars: Int) -> String {
        let recentTurns = Array(turns.suffix(maxTurns))
        var lines: [String] = []
        var total = 0
        for turn in recentTurns.reversed() {
            let line = "\(turn.role): \(turn.text)"
            total += line.count + 1
            if total > maxChars { break }
            lines.append(line)
        }
        let convo = lines.reversed().joined(separator: "\n")
        return """
You are a concise helpful assistant.
\(convo)
Assistant:
"""
    }

    private func downloadQwenSampleAssets() {
        errorText = nil
        selectedSampleLabel = "Qwen3.5 (stable)"
        if let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35StableFileName),
           let tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35TokenizerFileName),
           RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35TokenizerConfigFileName) != nil {
            modelURL = model
            tokenizerURL = tok
            downloadStatus = "Using existing files in Documents."
            downloadProgress = 0
            return
        }

        downloadStatus = "Downloading Qwen sample files..."
        downloadProgress = 0
        currentDownloadFile = ""
        currentDownloadSizeText = ""
        isDownloading = true
        Task {
            do {
                let totalFiles = 3.0
                let modelPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.qwen35Stable,
                    fileName: DemoAssetLinks.qwen35StableFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 0.0, progress: p, totalFiles: totalFiles, label: "Qwen stable", fileName: DemoAssetLinks.qwen35StableFileName)
                    }
                )
                let tokPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.qwen35Tokenizer,
                    fileName: DemoAssetLinks.qwen35TokenizerFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 1.0, progress: p, totalFiles: totalFiles, label: "Qwen", fileName: DemoAssetLinks.qwen35TokenizerFileName)
                    }
                )
                _ = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.qwen35TokenizerConfig,
                    fileName: DemoAssetLinks.qwen35TokenizerConfigFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 2.0, progress: p, totalFiles: totalFiles, label: "Qwen", fileName: DemoAssetLinks.qwen35TokenizerConfigFileName)
                    }
                )
                await MainActor.run {
                    self.modelURL = modelPath
                    self.tokenizerURL = tokPath
                    self.downloadProgress = 1.0
                    self.downloadStatus = "Saved: \(modelPath.lastPathComponent), \(tokPath.lastPathComponent)"
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            } catch {
                await MainActor.run {
                    self.errorText = String(describing: error)
                    self.downloadStatus = ""
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            }
        }
    }

    private func downloadSmolLMSampleAssets() {
        errorText = nil
        selectedSampleLabel = "SmolLM2"
        if let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2FileName),
           let tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerFileName),
           RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerConfigFileName) != nil {
            modelURL = model
            tokenizerURL = tok
            downloadStatus = "Using existing files in Documents."
            downloadProgress = 0
            return
        }

        downloadStatus = "Downloading SmolLM sample files..."
        downloadProgress = 0
        currentDownloadFile = ""
        currentDownloadSizeText = ""
        isDownloading = true
        Task {
            do {
                let totalFiles = 3.0
                let modelPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.smollm2Int8,
                    fileName: DemoAssetLinks.smollm2FileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 0.0, progress: p, totalFiles: totalFiles, label: "SmolLM", fileName: DemoAssetLinks.smollm2FileName)
                    }
                )
                let tokPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.smollm2Tokenizer,
                    fileName: DemoAssetLinks.smollm2TokenizerFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 1.0, progress: p, totalFiles: totalFiles, label: "SmolLM", fileName: DemoAssetLinks.smollm2TokenizerFileName)
                    }
                )
                _ = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.smollm2TokenizerConfig,
                    fileName: DemoAssetLinks.smollm2TokenizerConfigFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 2.0, progress: p, totalFiles: totalFiles, label: "SmolLM", fileName: DemoAssetLinks.smollm2TokenizerConfigFileName)
                    }
                )
                await MainActor.run {
                    self.modelURL = modelPath
                    self.tokenizerURL = tokPath
                    self.downloadProgress = 1.0
                    self.downloadStatus = "Saved: \(modelPath.lastPathComponent), \(tokPath.lastPathComponent)"
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            } catch {
                await MainActor.run {
                    self.errorText = String(describing: error)
                    self.downloadStatus = ""
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            }
        }
    }

    private func downloadGemmaSampleAssets() {
        errorText = nil
        selectedSampleLabel = "Gemma3-1B-IT (int8)"
        if let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma3FileName),
           let tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma3TokenizerFileName),
           RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma3TokenizerConfigFileName) != nil {
            modelURL = model
            tokenizerURL = tok
            downloadStatus = "Using existing files in Documents."
            downloadProgress = 0
            return
        }

        downloadStatus = "Downloading Gemma sample files..."
        downloadProgress = 0
        currentDownloadFile = ""
        currentDownloadSizeText = ""
        isDownloading = true
        Task {
            do {
                let totalFiles = 3.0
                let modelPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.gemma3Int8,
                    fileName: DemoAssetLinks.gemma3FileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 0.0, progress: p, totalFiles: totalFiles, label: "Gemma3", fileName: DemoAssetLinks.gemma3FileName)
                    }
                )
                let tokPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.gemma3Tokenizer,
                    fileName: DemoAssetLinks.gemma3TokenizerFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 1.0, progress: p, totalFiles: totalFiles, label: "Gemma3 tokenizer", fileName: DemoAssetLinks.gemma3TokenizerFileName)
                    }
                )
                _ = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.gemma3TokenizerConfig,
                    fileName: DemoAssetLinks.gemma3TokenizerConfigFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 2.0, progress: p, totalFiles: totalFiles, label: "Gemma3 tokenizer", fileName: DemoAssetLinks.gemma3TokenizerConfigFileName)
                    }
                )
                await MainActor.run {
                    self.modelURL = modelPath
                    self.tokenizerURL = tokPath
                    self.downloadProgress = 1.0
                    self.downloadStatus = "Saved: \(modelPath.lastPathComponent), \(tokPath.lastPathComponent)"
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            } catch {
                await MainActor.run {
                    self.errorText = String(describing: error)
                    self.downloadStatus = ""
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            }
        }
    }

    private func downloadGemma4LiteRtAssets() {
        errorText = nil
        selectedSampleLabel = "Gemma-4-2P3B-IT (LiteRT)"
        if let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma42p3bFileName),
           let tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma42p3bTokenizerFileName) {
            modelURL = model
            tokenizerURL = tok
            downloadStatus = "Using existing files in Documents."
            downloadProgress = 0
            return
        }
        downloadStatus = ""
        downloadProgress = 0
        currentDownloadFile = ""
        currentDownloadSizeText = ""
        isDownloading = false
        errorText = "Please download the Gemma-4 LiteRT bundle from the Models Hub tab first."
    }

    private func downloadQwenModelOnly() {
        errorText = nil
        selectedSampleLabel = "Qwen3.5 (stable)"
        downloadStatus = "Downloading Qwen stable model only..."
        downloadProgress = 0
        currentDownloadFile = ""
        currentDownloadSizeText = ""
        isDownloading = true
        Task {
            do {
                let modelPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.qwen35Stable,
                    fileName: DemoAssetLinks.qwen35StableFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 0.0, progress: p, totalFiles: 1.0, label: "Qwen stable", fileName: DemoAssetLinks.qwen35StableFileName)
                    }
                )
                await MainActor.run {
                    self.modelURL = modelPath
                    self.downloadProgress = 1.0
                    self.downloadStatus = "Saved model: \(modelPath.lastPathComponent)"
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            } catch {
                await MainActor.run {
                    self.errorText = String(describing: error)
                    self.downloadStatus = ""
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            }
        }
    }

    private func downloadQwenTokenizerOnly() {
        errorText = nil
        selectedSampleLabel = "Qwen3.5"
        downloadStatus = "Downloading Qwen tokenizer JSONs..."
        downloadProgress = 0
        currentDownloadFile = ""
        currentDownloadSizeText = ""
        isDownloading = true
        Task {
            do {
                let totalFiles = 2.0
                let tokPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.qwen35Tokenizer,
                    fileName: DemoAssetLinks.qwen35TokenizerFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 0.0, progress: p, totalFiles: totalFiles, label: "Qwen tokenizer", fileName: DemoAssetLinks.qwen35TokenizerFileName)
                    }
                )
                _ = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.qwen35TokenizerConfig,
                    fileName: DemoAssetLinks.qwen35TokenizerConfigFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 1.0, progress: p, totalFiles: totalFiles, label: "Qwen tokenizer", fileName: DemoAssetLinks.qwen35TokenizerConfigFileName)
                    }
                )
                await MainActor.run {
                    self.tokenizerURL = tokPath
                    self.downloadProgress = 1.0
                    self.downloadStatus = "Saved tokenizer JSONs for Qwen."
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            } catch {
                await MainActor.run {
                    self.errorText = String(describing: error)
                    self.downloadStatus = ""
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            }
        }
    }

    private func downloadQwenCompactModelOnly() {
        errorText = nil
        selectedSampleLabel = "Qwen3.5 (compact/experimental)"
        downloadStatus = "Downloading Qwen compact model only..."
        downloadProgress = 0
        currentDownloadFile = ""
        currentDownloadSizeText = ""
        isDownloading = true
        Task {
            do {
                let modelPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.qwen35CompactInt4TextOnly,
                    fileName: DemoAssetLinks.qwen35CompactFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 0.0, progress: p, totalFiles: 1.0, label: "Qwen compact", fileName: DemoAssetLinks.qwen35CompactFileName)
                    }
                )
                await MainActor.run {
                    self.modelURL = modelPath
                    self.downloadProgress = 1.0
                    self.downloadStatus = "Saved compact model: \(modelPath.lastPathComponent)"
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            } catch {
                await MainActor.run {
                    self.errorText = String(describing: error)
                    self.downloadStatus = ""
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            }
        }
    }

    private func downloadSmolLMModelOnly() {
        errorText = nil
        selectedSampleLabel = "SmolLM2"
        downloadStatus = "Downloading SmolLM model only..."
        downloadProgress = 0
        currentDownloadFile = ""
        currentDownloadSizeText = ""
        isDownloading = true
        Task {
            do {
                let modelPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.smollm2Int8,
                    fileName: DemoAssetLinks.smollm2FileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 0.0, progress: p, totalFiles: 1.0, label: "SmolLM", fileName: DemoAssetLinks.smollm2FileName)
                    }
                )
                await MainActor.run {
                    self.modelURL = modelPath
                    self.downloadProgress = 1.0
                    self.downloadStatus = "Saved model: \(modelPath.lastPathComponent)"
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            } catch {
                await MainActor.run {
                    self.errorText = String(describing: error)
                    self.downloadStatus = ""
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            }
        }
    }

    private func downloadSmolLMTokenizerOnly() {
        errorText = nil
        selectedSampleLabel = "SmolLM2"
        downloadStatus = "Downloading SmolLM tokenizer JSONs..."
        downloadProgress = 0
        currentDownloadFile = ""
        currentDownloadSizeText = ""
        isDownloading = true
        Task {
            do {
                let totalFiles = 2.0
                let tokPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.smollm2Tokenizer,
                    fileName: DemoAssetLinks.smollm2TokenizerFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 0.0, progress: p, totalFiles: totalFiles, label: "SmolLM tokenizer", fileName: DemoAssetLinks.smollm2TokenizerFileName)
                    }
                )
                _ = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.smollm2TokenizerConfig,
                    fileName: DemoAssetLinks.smollm2TokenizerConfigFileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 1.0, progress: p, totalFiles: totalFiles, label: "SmolLM tokenizer", fileName: DemoAssetLinks.smollm2TokenizerConfigFileName)
                    }
                )
                await MainActor.run {
                    self.tokenizerURL = tokPath
                    self.downloadProgress = 1.0
                    self.downloadStatus = "Saved tokenizer JSONs for SmolLM."
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            } catch {
                await MainActor.run {
                    self.errorText = String(describing: error)
                    self.downloadStatus = ""
                    self.currentDownloadFile = ""
                    self.currentDownloadSizeText = ""
                    self.isDownloading = false
                }
            }
        }
    }

    private func restoreAssets() {
        let gemmaModel = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma3FileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "gemma-3-1b-it-int8.cellmd")
        let gemmaTok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma3TokenizerFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "tokenizer-gemma-3-1b-it.json")
        let gemmaCfg = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma3TokenizerConfigFileName) != nil
            || RemoteAssets.existingDocumentsFile(fileName: "tokenizer_config.json") != nil
        if let gemmaModel, let gemmaTok, gemmaCfg {
            modelURL = gemmaModel
            tokenizerURL = gemmaTok
            selectedSampleLabel = "Gemma3-1B-IT (int8)"
            if downloadStatus.isEmpty { downloadStatus = "Loaded local sample files." }
            return
        }

        let gemma4Model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma42p3bFileName)
        let gemma4Tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma42p3bTokenizerFileName)
        let gemma4Cfg = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.gemma42p3bTokenizerConfigFileName) != nil
        if let gemma4Model, let gemma4Tok, gemma4Cfg {
            modelURL = gemma4Model
            tokenizerURL = gemma4Tok
            selectedSampleLabel = "Gemma-4-2P3B-IT (LiteRT)"
            if downloadStatus.isEmpty { downloadStatus = "Loaded local sample files." }
            return
        }

        let qwenModel = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35StableFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35CompactFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "qwen3.5-0.8b-int4-textonly.cellm")
            ?? RemoteAssets.existingDocumentsFile(fileName: "qwen3.5-0.8b.cellm")
        let qwenTok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35TokenizerFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "tokenizer-qwen3.5-0.8b.json")
        let qwenCfg = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35TokenizerConfigFileName) != nil
            || RemoteAssets.existingDocumentsFile(fileName: "tokenizer_config.json") != nil
        if let qwenModel, let qwenTok, qwenCfg {
            modelURL = qwenModel
            tokenizerURL = qwenTok
            let n = qwenModel.lastPathComponent.lowercased()
            selectedSampleLabel = (n == "qwen3.5-0.8b.cellm" || n == "qwen3.5-0.8b-int8.cellm")
                ? "Qwen3.5 (parity-fixed)"
                : "Qwen3.5 (experimental)"
            if downloadStatus.isEmpty { downloadStatus = "Loaded local sample files." }
            return
        }

        let smolModel = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2FileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "smollm2-135m-int8.cellm")
        let smolTok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "tokenizer-smollm2-135m.json")
        let smolCfg = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerConfigFileName) != nil
            || RemoteAssets.existingDocumentsFile(fileName: "tokenizer_config.json") != nil
        if let smolModel, let smolTok, smolCfg {
            modelURL = smolModel
            tokenizerURL = smolTok
            selectedSampleLabel = "SmolLM2"
            if downloadStatus.isEmpty { downloadStatus = "Loaded local sample files." }
            return
        }
    }

    private func clearLocalFiles() {
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.gemma42p3bFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.gemma42p3bTokenizerFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.gemma42p3bTokenizerConfigFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.gemma3FileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.gemma3TokenizerFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.gemma3TokenizerConfigFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.qwen35StableFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.qwen35CompactFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.qwen35TokenizerFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.qwen35TokenizerConfigFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.smollm2FileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerConfigFileName)
        RemoteAssets.removeDocumentsFile(fileName: "qwen3.5-0.8b-int4-textonly.cellm")
        RemoteAssets.removeDocumentsFile(fileName: "tokenizer-qwen3.5-0.8b.json")
        RemoteAssets.removeDocumentsFile(fileName: "smollm2-135m-int8.cellm")
        RemoteAssets.removeDocumentsFile(fileName: "tokenizer-smollm2-135m.json")
        RemoteAssets.removeDocumentsFile(fileName: "tokenizer_config.json")
        RemoteAssets.removeDocumentsFile(fileName: "picked")
        modelURL = nil
        tokenizerURL = nil
        selectedSampleLabel = ""
        downloadStatus = "Local sample files deleted."
        downloadProgress = 0
        currentDownloadFile = ""
        currentDownloadSizeText = ""
    }

    private func forceRedownload() {
        let priorSelection = selectedSampleLabel
        clearLocalFiles()
        if priorSelection == "SmolLM2" {
            downloadSmolLMSampleAssets()
        } else if priorSelection == "Gemma-4-2P3B-IT (LiteRT)" {
            downloadGemma4LiteRtAssets()
        } else if priorSelection == "Gemma3-1B-IT (int8)" {
            downloadGemmaSampleAssets()
        } else {
            downloadQwenSampleAssets()
        }
    }

    private func resolveStaleSelectionsIfNeeded() {
        let fm = FileManager.default
        if let modelURL, !fm.fileExists(atPath: modelURL.path) {
            self.modelURL = nil
        }
        if let tokenizerURL, !fm.fileExists(atPath: tokenizerURL.path) {
            self.tokenizerURL = nil
        }
        if self.modelURL == nil || self.tokenizerURL == nil {
            restoreAssets()
        }
    }

    private func persistPickedFile(_ sourceURL: URL, subdir: String) -> URL? {
        let accessing = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if accessing {
                sourceURL.stopAccessingSecurityScopedResource()
            }
        }

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
            self.errorText = "Failed to import selected file: \(error.localizedDescription)"
            return nil
        }
    }

    private func persistSharedSelection() {
        if let model = modelURL?.path {
            UserDefaults.standard.set(model, forKey: SharedSelection.llmModelPath)
        }
        if let tok = tokenizerURL?.path {
            UserDefaults.standard.set(tok, forKey: SharedSelection.llmTokenizerPath)
        }
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

    private func runQwenSmokeTest() {
        guard
            let modelName = modelURL?.lastPathComponent.lowercased(),
            let tokName = tokenizerURL?.lastPathComponent.lowercased(),
            modelName.contains("qwen"),
            tokName.contains("qwen")
        else {
            errorText = "Qwen smoke test requires both a Qwen model and Qwen tokenizer. Download Qwen sample assets first."
            return
        }
        prompt = "Return exactly one uppercase letter: R"
        selectedPreset = .strict
        run(maxTokensOverride: 1)
    }

    private func runSchedulerSmokeTest() {
        prompt = "Return exactly one uppercase letter: R"
        selectedPreset = .strict
        run(exerciseSuspendResume: true)
    }

    private func formatDiagnostics(stats: LlmGenerationStats) -> String {
        String(
            format: "prompt_tokens=%d generated_tokens=%d stop_reason=%@ first_piece=%@ prefill=%.1fms decode=%.1fms total=%.1fms",
            stats.promptTokenCount,
            stats.generatedTokenCount,
            stats.stopReason,
            stats.firstPiece,
            stats.prefillMs,
            stats.decodeMs,
            stats.totalMs
        )
    }

    private func formatInitDiagnostics(lines: [String], stats: LlmGenerationStats) -> String {
        let initLine = lines.joined(separator: " ")
        let genLine = formatDiagnostics(stats: stats)
        return "\(initLine)\n\(genLine)"
    }

    private func formatLiveDiagnostics(lines: [String], flushes: Int, chars: Int, elapsedMs: Double) -> String {
        let initLine = lines.joined(separator: " ")
        let seconds = max(0.001, elapsedMs / 1000.0)
        let flushRate = Double(flushes) / seconds
        let charRate = Double(chars) / seconds
        let liveLine = String(
            format: "stream_live flushes=%d chars=%d elapsed=%.1fms flushes_per_s=%.2f chars_per_s=%.2f",
            flushes,
            chars,
            elapsedMs,
            flushRate,
            charRate
        )
        return "\(initLine)\n\(liveLine)"
    }

    private func setDownloadProgress(completedFiles: Double, progress: RemoteAssets.DownloadProgress, totalFiles: Double, label: String, fileName: String) {
        let clamped = safeUnitProgress(progress.fraction)
        let safeTotal = (totalFiles.isFinite && totalFiles > 0.0) ? totalFiles : 1.0
        let overall = safeUnitProgress((completedFiles + clamped) / safeTotal)
        DispatchQueue.main.async {
            self.downloadProgress = overall
            self.downloadStatus = "Downloading \(label) sample files... \(Int((overall * 100).rounded()))%"
            self.currentDownloadFile = URL(fileURLWithPath: fileName).lastPathComponent
            self.currentDownloadSizeText = self.formatSizeProgress(received: progress.bytesReceived, expected: progress.bytesExpected)
        }
    }

    private func safeUnitProgress(_ x: Double) -> Double {
        guard x.isFinite else { return 0.0 }
        if x < 0.0 { return 0.0 }
        if x > 1.0 { return 1.0 }
        return x
    }

    private func formatSizeProgress(received: Int64, expected: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        let receivedText = formatter.string(fromByteCount: max(0, received))
        if expected > 0 {
            let expectedText = formatter.string(fromByteCount: expected)
            return "\(receivedText) / \(expectedText)"
        }
        return "\(receivedText) downloaded"
    }
}

struct DocumentPicker: UIViewControllerRepresentable {
    let allowed: [UTType]
    let onPick: (URL) -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: allowed, asCopy: true)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

    final class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }
        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let u = urls.first else { return }
            onPick(u)
        }
    }
}
