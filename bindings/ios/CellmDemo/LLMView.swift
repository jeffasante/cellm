import SwiftUI
import UniformTypeIdentifiers
import Dispatch

struct LLMView: View {
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
            case .chat: return 32
            case .balanced: return 64
            case .strict: return 32
            }
        }
    }

    @State private var modelURL: URL?
    @State private var tokenizerURL: URL?
    @State private var prompt: String = "Return exactly one uppercase letter: R"
    @State private var output: String = ""
    @State private var runDiagnostics: String = ""
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

    @State private var showModelPicker = false
    @State private var showTokenizerPicker = false

    private var isQwenSelected: Bool {
        (modelURL?.lastPathComponent.lowercased().contains("qwen") ?? false) ||
        (tokenizerURL?.lastPathComponent.lowercased().contains("qwen") ?? false)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                filesCard
                storageCard
                promptCard
                presetCard
                backendCard
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
            DocumentPicker(allowed: [.data]) { modelURL = $0 }
        }
        .sheet(isPresented: $showTokenizerPicker) {
            DocumentPicker(allowed: [.json]) { tokenizerURL = $0 }
        }
        .onAppear {
            restoreAssets()
        }
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
        card("Files") {
            actionButton(modelURL == nil ? "Pick .cellm model" : modelURL!.lastPathComponent, icon: "externaldrive") { showModelPicker = true }
            actionButton(tokenizerURL == nil ? "Pick tokenizer.json" : tokenizerURL!.lastPathComponent, icon: "doc.text") { showTokenizerPicker = true }
            actionButton(isDownloading ? "Downloading sample files…" : "Download Qwen sample model + tokenizer", icon: "arrow.down.circle", disabled: isDownloading || isRunning) {
                downloadQwenSampleAssets()
            }
            actionButton(isDownloading ? "Downloading sample files…" : "Download Qwen model only", icon: "shippingbox", disabled: isDownloading || isRunning) {
                downloadQwenModelOnly()
            }
            actionButton(isDownloading ? "Downloading sample files…" : "Download Qwen tokenizer JSON only", icon: "doc.badge.arrow.down", disabled: isDownloading || isRunning) {
                downloadQwenTokenizerOnly()
            }
            actionButton(isDownloading ? "Downloading sample files…" : "Download SmolLM sample model + tokenizer", icon: "arrow.down.circle", disabled: isDownloading || isRunning) {
                downloadSmolLMSampleAssets()
            }
            actionButton(isDownloading ? "Downloading sample files…" : "Download SmolLM model only", icon: "shippingbox", disabled: isDownloading || isRunning) {
                downloadSmolLMModelOnly()
            }
            actionButton(isDownloading ? "Downloading sample files…" : "Download SmolLM tokenizer JSON only", icon: "doc.badge.arrow.down", disabled: isDownloading || isRunning) {
                downloadSmolLMTokenizerOnly()
            }
            actionButton("Run Qwen Smoke Test", icon: "bolt.circle", disabled: isRunning || modelURL == nil || tokenizerURL == nil) {
                runQwenSmokeTest()
            }
            actionButton("Run Scheduler Smoke (Suspend/Resume)", icon: "pause.circle", disabled: isRunning || modelURL == nil || tokenizerURL == nil) {
                runSchedulerSmokeTest()
            }
            if !downloadStatus.isEmpty {
                Text(downloadStatus).font(.footnote).foregroundStyle(.secondary)
            }
            if isDownloading {
                ProgressView(value: downloadProgress)
                    .progressViewStyle(.linear)
                Text("\(Int((downloadProgress * 100.0).rounded()))%")
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
        }
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
            Text("Note: current LLM math path is CPU in this phase; Metal selection verifies backend selection/fallback.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            if isQwenSelected {
                Text("Qwen is currently experimental in mobile runner parity; SmolLM is recommended for stable app tests.")
                    .font(.footnote)
                    .foregroundStyle(.orange)
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

    private func run(exerciseSuspendResume: Bool = false, maxTokensOverride: Int? = nil) {
        // Dismiss keyboard
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)

        errorText = nil
        output = ""
        runDiagnostics = ""
        backendWarning = nil
        guard let modelURL, let tokenizerURL else { return }
        let promptText = prompt
        let backend = selectedBackend
        let thermal = selectedThermalLevel
        let preset = selectedPreset

        isRunning = true
        Task.detached(priority: .userInitiated) {
            do {
                let tok = try CellmTokenizer(tokenizerURL: tokenizerURL)
                let eng = try CellmEngine(
                    modelURL: modelURL,
                    tokenizer: tok,
                    topK: preset.topK,
                    temperature: preset.temperature,
                    repeatPenalty: preset.repeatPenalty,
                    repeatWindow: preset.repeatWindow,
                    seed: 1,
                    backend: backend
                )
                let text = try eng.generate(
                    prompt: promptText,
                    maxNewTokens: maxTokensOverride ?? preset.maxTokens,
                    thermalLevel: thermal,
                    exerciseSuspendResume: exerciseSuspendResume
                )
                await MainActor.run {
                    self.output = prettyOutput(text)
                    if let stats = eng.lastGenerationStats {
                        self.runDiagnostics = formatDiagnostics(stats: stats)
                    }
                    self.activeBackend = eng.activeBackend
                    if backend == .metal && !eng.activeBackend.lowercased().contains("metal") {
                        self.backendWarning = "Metal requested, fell back to \(eng.activeBackend)."
                    }
                    self.isRunning = false
                }
            } catch {
                await MainActor.run {
                    self.errorText = String(describing: error)
                    self.isRunning = false
                }
            }
        }
    }

    private func downloadQwenSampleAssets() {
        errorText = nil
        selectedSampleLabel = "Qwen3.5"
        if let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35FileName),
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
                    from: DemoAssetLinks.qwen35Int4TextOnly,
                    fileName: DemoAssetLinks.qwen35FileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 0.0, progress: p, totalFiles: totalFiles, label: "Qwen", fileName: DemoAssetLinks.qwen35FileName)
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

    private func downloadQwenModelOnly() {
        errorText = nil
        selectedSampleLabel = "Qwen3.5"
        downloadStatus = "Downloading Qwen model only..."
        downloadProgress = 0
        currentDownloadFile = ""
        currentDownloadSizeText = ""
        isDownloading = true
        Task {
            do {
                let modelPath = try await RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.qwen35Int4TextOnly,
                    fileName: DemoAssetLinks.qwen35FileName,
                    progress: { p in
                        setDownloadProgress(completedFiles: 0.0, progress: p, totalFiles: 1.0, label: "Qwen", fileName: DemoAssetLinks.qwen35FileName)
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

        let qwenModel = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35FileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "qwen3.5-0.8b-int4-textonly.cellm")
        let qwenTok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35TokenizerFileName)
            ?? RemoteAssets.existingDocumentsFile(fileName: "tokenizer-qwen3.5-0.8b.json")
        let qwenCfg = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.qwen35TokenizerConfigFileName) != nil
            || RemoteAssets.existingDocumentsFile(fileName: "tokenizer_config.json") != nil
        if let qwenModel, let qwenTok, qwenCfg {
            modelURL = qwenModel
            tokenizerURL = qwenTok
            selectedSampleLabel = "Qwen3.5"
            if downloadStatus.isEmpty { downloadStatus = "Loaded local sample files." }
            return
        }
    }

    private func clearLocalFiles() {
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.qwen35FileName)
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
        } else {
            downloadQwenSampleAssets()
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
            format: "prompt_tokens=%d generated_tokens=%d first_piece=%@ prefill=%.1fms decode=%.1fms total=%.1fms",
            stats.promptTokenCount,
            stats.generatedTokenCount,
            stats.firstPiece,
            stats.prefillMs,
            stats.decodeMs,
            stats.totalMs
        )
    }

    private func setDownloadProgress(completedFiles: Double, progress: RemoteAssets.DownloadProgress, totalFiles: Double, label: String, fileName: String) {
        let clamped = min(1.0, max(0.0, progress.fraction))
        let overall = min(1.0, max(0.0, (completedFiles + clamped) / totalFiles))
        DispatchQueue.main.async {
            self.downloadProgress = overall
            self.downloadStatus = "Downloading \(label) sample files... \(Int((overall * 100).rounded()))%"
            self.currentDownloadFile = URL(fileURLWithPath: fileName).lastPathComponent
            self.currentDownloadSizeText = self.formatSizeProgress(received: progress.bytesReceived, expected: progress.bytesExpected)
        }
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
