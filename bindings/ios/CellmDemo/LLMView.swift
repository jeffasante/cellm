import SwiftUI
import UniformTypeIdentifiers

struct LLMView: View {
    @State private var modelURL: URL?
    @State private var tokenizerURL: URL?
    @State private var prompt: String = "Hello, how are you?"
    @State private var output: String = ""
    @State private var isRunning: Bool = false
    @State private var errorText: String?
    @State private var selectedBackend: CellmBackend = .metal
    @State private var activeBackend: String = ""
    @State private var downloadStatus: String = ""
    @State private var isDownloading: Bool = false
    @State private var backendWarning: String?

    @State private var showModelPicker = false
    @State private var showTokenizerPicker = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                filesCard
                storageCard
                promptCard
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
            actionButton(isDownloading ? "Downloading sample files…" : "Download sample model + tokenizer", icon: "arrow.down.circle", disabled: isDownloading || isRunning) {
                downloadSampleAssets()
            }
            if !downloadStatus.isEmpty {
                Text(downloadStatus).font(.footnote).foregroundStyle(.secondary)
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
            if !activeBackend.isEmpty {
                Text("Active: \(activeBackend)").font(.footnote).foregroundStyle(.secondary)
            }
            if let backendWarning {
                Text(backendWarning).font(.footnote).foregroundStyle(.orange)
            }
            Text("Note: current LLM math path is CPU in this phase; Metal selection verifies backend selection/fallback.")
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

    private func run() {
        // Dismiss keyboard
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)

        errorText = nil
        output = ""
        backendWarning = nil
        guard let modelURL, let tokenizerURL else { return }
        let promptText = prompt
        let backend = selectedBackend

        isRunning = true
        Task.detached(priority: .userInitiated) {
            do {
                let tok = try CellmTokenizer(tokenizerURL: tokenizerURL)
                let eng = try CellmEngine(modelURL: modelURL, tokenizer: tok, backend: backend)
                let text = try eng.generate(prompt: promptText, maxNewTokens: 64)
                await MainActor.run {
                    self.output = prettyOutput(text)
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

    private func downloadSampleAssets() {
        errorText = nil
        if let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2FileName),
           let tok = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerFileName),
           RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerConfigFileName) != nil {
            modelURL = model
            tokenizerURL = tok
            downloadStatus = "Using existing files in Documents."
            return
        }

        downloadStatus = "Downloading sample files..."
        isDownloading = true
        Task {
            do {
                async let model = RemoteAssets.downloadToDocuments(from: DemoAssetLinks.smollm2Int8, fileName: DemoAssetLinks.smollm2FileName)
                async let tok = RemoteAssets.downloadToDocuments(from: DemoAssetLinks.smollm2Tokenizer, fileName: DemoAssetLinks.smollm2TokenizerFileName)
                async let tokCfg = RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.smollm2TokenizerConfig,
                    fileName: DemoAssetLinks.smollm2TokenizerConfigFileName
                )
                let (modelPath, tokPath, _) = try await (model, tok, tokCfg)
                await MainActor.run {
                    self.modelURL = modelPath
                    self.tokenizerURL = tokPath
                    self.downloadStatus = "Saved: \(modelPath.lastPathComponent), \(tokPath.lastPathComponent)"
                    self.isDownloading = false
                }
            } catch {
                await MainActor.run {
                    self.errorText = String(describing: error)
                    self.downloadStatus = ""
                    self.isDownloading = false
                }
            }
        }
    }

    private func restoreAssets() {
        if modelURL == nil, let url = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2FileName) {
            modelURL = url
        }
        if tokenizerURL == nil, let url = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerFileName) {
            tokenizerURL = url
        }
        let hasTokCfg = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerConfigFileName) != nil
        if modelURL != nil && tokenizerURL != nil && hasTokCfg && downloadStatus.isEmpty {
            downloadStatus = "Loaded local sample files."
        }
    }

    private func clearLocalFiles() {
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.smollm2FileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.smollm2TokenizerConfigFileName)
        modelURL = nil
        tokenizerURL = nil
        downloadStatus = "Local sample files deleted."
    }

    private func forceRedownload() {
        clearLocalFiles()
        downloadSampleAssets()
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
