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

    @State private var showModelPicker = false
    @State private var showTokenizerPicker = false

    var body: some View {
        NavigationView {
            Form {
                Section("Files") {
                    Button(modelURL == nil ? "Pick .cellm model" : "Model: \(modelURL!.lastPathComponent)") {
                        showModelPicker = true
                    }
                    Button(tokenizerURL == nil ? "Pick tokenizer.json" : "Tokenizer: \(tokenizerURL!.lastPathComponent)") {
                        showTokenizerPicker = true
                    }
                    Button(isDownloading ? "Downloading…" : "Download sample model + tokenizer") {
                        downloadSampleAssets()
                    }
                    .disabled(isDownloading || isRunning)
                    if !downloadStatus.isEmpty {
                        Text(downloadStatus)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                Section("Prompt") {
                    TextField("Prompt", text: $prompt, axis: .vertical)
                        .lineLimit(3...8)
                }

                Section("Backend") {
                    Picker("Requested", selection: $selectedBackend) {
                        ForEach(CellmBackend.allCases) { backend in
                            Text(backend.label).tag(backend)
                        }
                    }
                    .pickerStyle(.segmented)
                    if !activeBackend.isEmpty {
                        Text("Active: \(activeBackend)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                Section("Run") {
                    Button(isRunning ? "Running…" : "Generate") {
                        run()
                    }
                    .disabled(isRunning || modelURL == nil || tokenizerURL == nil || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                if let errorText {
                    Section("Error") {
                        Text(errorText).foregroundColor(.red)
                    }
                }

                Section("Output") {
                    Text(output)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                }
            }
            .navigationTitle("cellm LLM")
        }
        .sheet(isPresented: $showModelPicker) {
            DocumentPicker(allowed: [.data]) { url in
                modelURL = url
            }
        }
        .sheet(isPresented: $showTokenizerPicker) {
            DocumentPicker(allowed: [.json]) { url in
                tokenizerURL = url
            }
        }
    }

    private func run() {
        errorText = nil
        output = ""
        guard let modelURL, let tokenizerURL else { return }
        let promptText = prompt
        let backend = selectedBackend

        isRunning = true
        Task.detached(priority: .userInitiated) {
            do {
                let tok = try CellmTokenizer(tokenizerURL: tokenizerURL)
                let eng = try CellmEngine(modelURL: modelURL, tokenizer: tok, backend: backend)
                let text = try eng.generate(prompt: promptText, maxNewTokens: 96)
                await MainActor.run {
                    self.output = text
                    self.activeBackend = eng.activeBackend
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
        downloadStatus = "Downloading sample files..."
        isDownloading = true

        Task {
            do {
                async let model = RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.smollm2Int8,
                    fileName: "smollm2-135m-int8.cellm"
                )
                async let tok = RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.smollm2Tokenizer,
                    fileName: "tokenizer-smollm2-135m.json"
                )
                let (modelPath, tokPath) = try await (model, tok)
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

    func makeCoordinator() -> Coordinator {
        Coordinator(onPick: onPick)
    }

    final class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let u = urls.first else { return }
            onPick(u)
        }
    }
}
