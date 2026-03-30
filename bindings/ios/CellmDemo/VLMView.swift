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
    @State private var selectedBackend: CellmBackend = .metal
    @State private var activeBackend: String = ""
    @State private var downloadStatus: String = ""
    @State private var isDownloading: Bool = false
    @State private var backendWarning: String?
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
                }

                Button(isRunning ? "Running…" : "Run VLM") { run() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .frame(maxWidth: .infinity)
                    .disabled(isRunning || modelURL == nil || imageBytes == nil || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                if let errorText {
                    card("Error") { Text(errorText).foregroundStyle(.red) }
                }

                card("Output") {
                    Text(output.isEmpty ? "No output yet." : output)
                        .font(.system(.body, design: .monospaced))
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
                        self.imageBytes = data
                    }
                }
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
        errorText = nil
        output = ""
        backendWarning = nil
        guard let modelURL, let imageBytes else { return }
        let promptText = prompt
        let backend = selectedBackend

        isRunning = true
        Task.detached(priority: .userInitiated) {
            do {
                let eng = try CellmVLMEngine(modelURL: modelURL, backend: backend)
                let text = try eng.describe(imageBytes: imageBytes, prompt: promptText)
                await MainActor.run {
                    self.output = text
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
        if let model = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.smolvlmFileName),
           let imageURL = RemoteAssets.existingDocumentsFile(fileName: DemoAssetLinks.rococoFileName),
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
                let (modelPath, imagePath) = try await (model, imageURL)
                let bytes = try Data(contentsOf: imagePath)
                guard let ui = UIImage(data: bytes) else {
                    throw CellmError.message("Downloaded image could not be decoded")
                }
                await MainActor.run {
                    self.modelURL = modelPath
                    self.imageBytes = bytes
                    self.image = Image(uiImage: ui)
                    self.downloadStatus = "Saved model and loaded sample image"
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
        if modelURL != nil && imageBytes != nil && downloadStatus.isEmpty {
            downloadStatus = "Loaded local sample files."
        }
    }
}
