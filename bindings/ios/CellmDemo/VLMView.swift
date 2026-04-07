import SwiftUI
import PhotosUI

struct VLMView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var image: Image?
    @State private var imageBytes: Data?
    @State private var modelURL: URL?
    @State private var prompt: String = "Describe this image."
    @State private var output: String = ""
    @State private var isRunning: Bool = false
    @State private var errorText: String?
    @State private var infoText: String?
    @State private var selectedBackend: CellmBackend = .metal
    @State private var activeBackend: String = ""
    @State private var downloadStatus: String = ""
    @State private var isDownloading: Bool = false
    @State private var backendWarning: String?
    @State private var timingText: String?
    @State private var showBackendSettings: Bool = false
    @State private var showModelPicker = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("cellm VLM")
                        .font(.system(size: 38, weight: .bold))
                    Text("Run image-to-text on-device")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }

                card("Files") {
                    actionButton(modelURL == nil ? "Pick .cellm model" : modelURL!.lastPathComponent, icon: "externaldrive") { showModelPicker = true }
                    actionButton(isDownloading ? "Downloading sample assets…" : "Download sample VLM model + image", icon: "arrow.down.circle", disabled: isDownloading || isRunning) {
                        downloadSampleAssets()
                    }
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        HStack(spacing: 10) {
                            Image(systemName: "photo").foregroundStyle(.blue)
                            Text(imageBytes == nil ? "Pick image from Photos" : "Image selected")
                            Spacer()
                        }
                        .padding(12)
                        .background(Color(.systemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                    .buttonStyle(.plain)
                    if !downloadStatus.isEmpty {
                        Text(downloadStatus).font(.footnote).foregroundStyle(.secondary)
                    }
                }

                card("Storage") {
                    if modelURL == nil && imageBytes == nil {
                        Text("No local sample files found.")
                            .foregroundStyle(.secondary)
                            .font(.footnote)
                    } else {
                        if let modelSize = RemoteAssets.fileSizeString(url: modelURL) {
                            Text("Model: \(modelSize)").font(.footnote).foregroundStyle(.secondary)
                        }
                        if let imageURL = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.rococoFileName),
                           let imageSize = RemoteAssets.fileSizeString(url: imageURL) {
                            Text("Image: \(imageSize)").font(.footnote).foregroundStyle(.secondary)
                        }
                        if let tokURL = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmTokenizerFileName),
                           let tokSize = RemoteAssets.fileSizeString(url: tokURL) {
                            Text("Tokenizer: \(tokSize)").font(.footnote).foregroundStyle(.secondary)
                        }
                        if let procURL = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmProcessorConfigFileName),
                           let procSize = RemoteAssets.fileSizeString(url: procURL) {
                            Text("Processor config: \(procSize)").font(.footnote).foregroundStyle(.secondary)
                        }
                        if let preprocURL = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmPreprocessorConfigFileName),
                           let preprocSize = RemoteAssets.fileSizeString(url: preprocURL) {
                            Text("Preprocessor config: \(preprocSize)").font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                    HStack(spacing: 10) {
                        Button("Re-download") { forceRedownload() }
                            .buttonStyle(.bordered)
                            .disabled(isDownloading || isRunning)
                        Button("Delete local files") { clearLocalFiles() }
                            .buttonStyle(.bordered)
                            .disabled(isDownloading || isRunning || (modelURL == nil && imageBytes == nil))
                    }
                }

                if let image {
                    card("Preview") {
                        image
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 240)
                            .frame(maxWidth: .infinity)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                }

                card("Prompt") {
                    TextEditor(text: $prompt)
                        .frame(minHeight: 100)
                        .padding(8)
                        .background(Color(.systemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }

                settingsSection

                Button(isRunning ? "Running…" : "Run VLM") { run() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .frame(maxWidth: .infinity)
                    .disabled(isRunning || modelURL == nil || imageBytes == nil || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                if let infoText {
                    card("Status") {
                        Text(infoText).foregroundStyle(.secondary)
                    }
                }

                if let errorText {
                    card("Error") { Text(errorText).foregroundStyle(.red) }
                }

                if let timingText {
                    card("Timings") {
                        Text(timingText)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

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
        .onAppear {
            restoreAssets()
        }
        .onChange(of: selectedItem) { newValue in
            guard let newValue else { return }
            Task {
                if let data = try? await newValue.loadTransferable(type: Data.self),
                   let ui = UIImage(data: data) {
                    await MainActor.run {
                        self.image = Image(uiImage: ui)
                        self.imageBytes = normalizedJPEGData(from: ui) ?? data
                    }
                }
            }
        }
    }

    private var settingsSection: some View {
        VStack(spacing: 12) {
            Button {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                    showBackendSettings.toggle()
                }
            } label: {
                HStack {
                    Label(showBackendSettings ? "Hide Backend Settings" : "Backend Settings", systemImage: "cpu")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Image(systemName: showBackendSettings ? "chevron.up" : "chevron.down")
                        .font(.caption.bold())
                        .foregroundStyle(.secondary)
                }
                .padding(14)
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 16))
            }
            .buttonStyle(.plain)

            if showBackendSettings {
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
                    Text("Note: when Metal is available, VLM uses Metal matmul for linear layers; remaining ops are still CPU in this phase.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
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
                Text(title).frame(maxWidth: .infinity, alignment: .leading)
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
        infoText = nil
        timingText = nil
        output = ""
        backendWarning = nil
        guard let modelURL, let imageBytes else { return }
        let promptText = prompt
        let backend = selectedBackend

        isRunning = true
        Task.detached(priority: .userInitiated) {
            do {
                let eng = try CellmVLMEngine(
                    modelURL: modelURL,
                    topK: 20,
                    temperature: 0.2,
                    repeatPenalty: 1.15,
                    repeatWindow: 64,
                    backend: backend
                )
                let text = try eng.describe(imageBytes: imageBytes, prompt: promptText)
                await MainActor.run {
                    self.output = prettyOutput(text)
                    self.activeBackend = eng.activeBackend
                    if let t = eng.lastTimings {
                        let patchMs = finiteMs(t.patchMs)
                        let encoderMs = finiteMs(t.encoderMs)
                        let decodeMs = finiteMs(t.decodeMs)
                        let totalMs = finiteMs(t.totalMs)
                        var summary = String(
                            format: "patch %.1f ms • encoder %.1f ms • decode %.1f ms • total %.1f ms",
                            patchMs,
                            encoderMs,
                            decodeMs,
                            totalMs
                        )
                        if let maxPair = t.encoderLayerMs.enumerated().max(by: { $0.element < $1.element }) {
                            summary += String(
                                format: "\nencoder layers: %d • hottest L%d %.1f ms",
                                t.encoderLayerMs.count,
                                maxPair.offset,
                                maxPair.element
                            )
                        }
                        self.timingText = summary
                    }
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
        if let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmFileName),
           let imageURL = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.rococoFileName),
           RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmTokenizerFileName) != nil,
           RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmProcessorConfigFileName) != nil,
           RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmPreprocessorConfigFileName) != nil,
           let bytes = try? Data(contentsOf: imageURL),
           let ui = UIImage(data: bytes) {
            modelURL = model
            imageBytes = bytes
            image = Image(uiImage: ui)
            downloadStatus = "Using existing files in Documents."
            return
        }

        downloadStatus = "Downloading sample VLM assets..."
        isDownloading = true
        Task {
            do {
                async let model = RemoteAssets.downloadToDocuments(from: DemoAssetLinks.smolvlmInt8, fileName: DemoAssetLinks.smolvlmFileName)
                async let imageURL = RemoteAssets.downloadToDocuments(from: DemoAssetLinks.rococoImage, fileName: DemoAssetLinks.rococoFileName)
                async let tokenizer = RemoteAssets.downloadToDocuments(from: DemoAssetLinks.smolvlmTokenizer, fileName: DemoAssetLinks.smolvlmTokenizerFileName)
                async let processor = RemoteAssets.downloadToDocuments(from: DemoAssetLinks.smolvlmProcessorConfig, fileName: DemoAssetLinks.smolvlmProcessorConfigFileName)
                async let preprocessor = RemoteAssets.downloadToDocuments(from: DemoAssetLinks.smolvlmPreprocessorConfig, fileName: DemoAssetLinks.smolvlmPreprocessorConfigFileName)
                let (modelPath, imagePath, _, _, _) = try await (model, imageURL, tokenizer, processor, preprocessor)
                let bytes = try Data(contentsOf: imagePath)
                guard let ui = UIImage(data: bytes) else {
                    throw CellmError.message("Downloaded image could not be decoded")
                }
                await MainActor.run {
                    self.modelURL = modelPath
                    self.imageBytes = bytes
                    self.image = Image(uiImage: ui)
                    self.downloadStatus = "Saved model, tokenizer, processor+preprocessor config and sample image"
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
        if modelURL == nil, let url = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmFileName) {
            modelURL = url
        }
        if imageBytes == nil,
           let imageURL = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.rococoFileName),
           let bytes = try? Data(contentsOf: imageURL),
           let ui = UIImage(data: bytes) {
            imageBytes = bytes
            image = Image(uiImage: ui)
        }
        let hasTokenizer = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmTokenizerFileName) != nil
        let hasProcessor = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmProcessorConfigFileName) != nil
        let hasPreprocessor = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmPreprocessorConfigFileName) != nil
        if modelURL != nil && imageBytes != nil && hasTokenizer && hasProcessor && hasPreprocessor && downloadStatus.isEmpty {
            downloadStatus = "Loaded local sample files."
        }
    }

    private func clearLocalFiles() {
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.smolvlmFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.smolvlmTokenizerFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.smolvlmProcessorConfigFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.smolvlmPreprocessorConfigFileName)
        RemoteAssets.removeDocumentsFile(fileName: DemoAssetLinks.rococoFileName)
        modelURL = nil
        imageBytes = nil
        image = nil
        downloadStatus = "Local sample files deleted."
    }

    private func finiteMs(_ x: Double) -> Double {
        guard x.isFinite, x >= 0.0 else { return 0.0 }
        return x
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
