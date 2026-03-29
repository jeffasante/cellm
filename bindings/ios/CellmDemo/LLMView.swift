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
        NavigationStack {
            ZStack {
                LinearGradient(
                    colors: [Color(.systemBackground), Color(.systemGray6)],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("cellm LLM")
                                .font(.system(size: 32, weight: .bold))
                            Text("Run local text generation on-device")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }

                        sectionCard("Files") {
                            actionRow(
                                title: modelURL == nil ? "Pick .cellm model" : modelURL!.lastPathComponent,
                                icon: "externaldrive",
                                action: { showModelPicker = true }
                            )
                            actionRow(
                                title: tokenizerURL == nil ? "Pick tokenizer.json" : tokenizerURL!.lastPathComponent,
                                icon: "doc.text",
                                action: { showTokenizerPicker = true }
                            )
                            actionRow(
                                title: isDownloading ? "Downloading sample files…" : "Download sample model + tokenizer",
                                icon: "arrow.down.circle",
                                disabled: isDownloading || isRunning,
                                action: downloadSampleAssets
                            )
                            if !downloadStatus.isEmpty {
                                Text(downloadStatus)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .padding(.top, 6)
                            }
                        }

                        sectionCard("Prompt") {
                            TextEditor(text: $prompt)
                                .frame(minHeight: 110)
                                .padding(10)
                                .background(Color(.secondarySystemBackground))
                                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                        }

                        sectionCard("Backend") {
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
                                    .padding(.top, 4)
                            }
                        }

                        Button {
                            run()
                        } label: {
                            HStack {
                                Spacer()
                                Text(isRunning ? "Generating…" : "Generate")
                                    .fontWeight(.semibold)
                                Spacer()
                            }
                            .padding(.vertical, 14)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isRunning || modelURL == nil || tokenizerURL == nil || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                        if let errorText {
                            sectionCard("Error") {
                                Text(errorText)
                                    .foregroundColor(.red)
                            }
                        }

                        sectionCard("Output") {
                            Text(output.isEmpty ? "No output yet." : output)
                                .font(.system(.body, design: .monospaced))
                                .foregroundColor(output.isEmpty ? .secondary : .primary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(10)
                                .background(Color(.secondarySystemBackground))
                                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                        }
                    }
                    .padding(16)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
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

    private func sectionCard<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)
            content()
        }
        .padding(14)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color(.separator).opacity(0.25), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.03), radius: 8, x: 0, y: 4)
    }

    private func actionRow(title: String, icon: String, disabled: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .foregroundColor(.accentColor)
                Text(title)
                    .multilineTextAlignment(.leading)
                Spacer()
            }
            .padding(.vertical, 12)
            .padding(.horizontal, 10)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .opacity(disabled ? 0.6 : 1)
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
