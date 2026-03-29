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

    @State private var showModelPicker = false

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
                            Text("cellm VLM")
                                .font(.system(size: 32, weight: .bold))
                            Text("Test vision-language flow on-device")
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
                                title: isDownloading ? "Downloading sample assets…" : "Download sample VLM model + image",
                                icon: "arrow.down.circle",
                                disabled: isDownloading || isRunning,
                                action: downloadSampleAssets
                            )
                            PhotosPicker(selection: $selectedItem, matching: .images) {
                                HStack(spacing: 10) {
                                    Image(systemName: "photo")
                                        .foregroundColor(.accentColor)
                                    Text(imageBytes == nil ? "Pick image from Photos" : "Image selected")
                                    Spacer()
                                }
                                .padding(.vertical, 12)
                                .padding(.horizontal, 10)
                                .background(Color(.secondarySystemBackground))
                                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                            }
                            .buttonStyle(.plain)
                            if !downloadStatus.isEmpty {
                                Text(downloadStatus)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .padding(.top, 6)
                            }
                        }

                        if let image {
                            sectionCard("Preview") {
                                image
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 240)
                                    .frame(maxWidth: .infinity)
                                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                            }
                        }

                        sectionCard("Prompt") {
                            TextEditor(text: $prompt)
                                .frame(minHeight: 100)
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
                                Text(isRunning ? "Running…" : "Run VLM")
                                    .fontWeight(.semibold)
                                Spacer()
                            }
                            .padding(.vertical, 14)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isRunning || modelURL == nil || imageBytes == nil || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

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
        downloadStatus = "Downloading sample VLM assets..."
        isDownloading = true

        Task {
            do {
                async let model = RemoteAssets.downloadToDocuments(
                    from: DemoAssetLinks.smolvlmInt8,
                    fileName: "smolvlm-256m-int8.cellm"
                )
                async let imageData = RemoteAssets.fetchData(from: DemoAssetLinks.rococoImage)
                let (modelPath, bytes) = try await (model, imageData)
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
}
