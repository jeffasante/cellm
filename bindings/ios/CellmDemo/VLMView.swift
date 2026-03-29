import SwiftUI
import PhotosUI

struct VLMView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var image: Image?
    @State private var imageBytes: Data?

    @State private var modelURL: URL?
    @State private var prompt: String = "What is in this image?"
    @State private var output: String = ""
    @State private var isRunning: Bool = false
    @State private var errorText: String?
    @State private var selectedBackend: CellmBackend = .metal
    @State private var activeBackend: String = ""

    @State private var showModelPicker = false

    var body: some View {
        NavigationView {
            Form {
                Section("Files") {
                    Button(modelURL == nil ? "Pick .cellm model" : "Model: \(modelURL!.lastPathComponent)") {
                        showModelPicker = true
                    }
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        Text(imageBytes == nil ? "Pick image" : "Image picked")
                    }
                }

                Section("Prompt") {
                    TextField("Prompt", text: $prompt, axis: .vertical)
                        .lineLimit(2...6)
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
                    Button(isRunning ? "Running…" : "Test VLM (expected to fail for now)") {
                        run()
                    }
                    .disabled(isRunning || modelURL == nil || imageBytes == nil || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                if let image {
                    Section("Preview") {
                        image
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 220)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                }

                if let errorText {
                    Section("Error") {
                        Text(errorText).foregroundColor(.red)
                    }
                }

                Section("Output") {
                    Text(output.isEmpty ? "(no output)" : output)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                }
            }
            .navigationTitle("cellm VLM")
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
}
