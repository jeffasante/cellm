import SwiftUI

struct ModelAssetCard: Identifiable, Hashable {
    var id: String { name }
    let name: String
    let description: String
    let fileName: String
    let url: String
    let tokenizerName: String?
    let tokenizerUrl: String?
}

struct ModelsView: View {
    @State private var downloadProgress: [String: Double] = [:]
    @State private var fileSizes: [String: String] = [:]
    @State private var isDownloading: [String: Bool] = [:]
    @State private var errorMessage: String?
    
    let availableModels: [ModelAssetCard] = [
        ModelAssetCard(
            name: "Gemma 4 (LiteRT)",
            description: "Google's Experimental 2B parameter model formatted for LiteRT. Fast and extremely efficient.",
            fileName: DemoAssetLinks.gemma42p3bFileName,
            url: DemoAssetLinks.gemma42p3bLiteRt,
            tokenizerName: DemoAssetLinks.gemma42p3bTokenizerFileName,
            tokenizerUrl: DemoAssetLinks.gemma42p3bTokenizer
        ),
        ModelAssetCard(
            name: "Gemma 3 (1B)",
            description: "Highly capable 1B model running natively.",
            fileName: DemoAssetLinks.gemma3FileName,
            url: DemoAssetLinks.gemma3Int8,
            tokenizerName: DemoAssetLinks.gemma3TokenizerFileName,
            tokenizerUrl: DemoAssetLinks.gemma3Tokenizer
        ),
        ModelAssetCard(
            name: "Qwen 3.5 (0.8B)",
            description: "Compact and powerful 800M model.",
            fileName: DemoAssetLinks.qwen35StableFileName,
            url: DemoAssetLinks.qwen35Stable,
            tokenizerName: DemoAssetLinks.qwen35TokenizerFileName,
            tokenizerUrl: DemoAssetLinks.qwen35Tokenizer
        ),
        ModelAssetCard(
            name: "SmolLM2 (135M)",
            description: "Very fast 135M parameter LLM ideal for older devices.",
            fileName: DemoAssetLinks.smollm2FileName,
            url: DemoAssetLinks.smollm2Int8,
            tokenizerName: DemoAssetLinks.smollm2TokenizerFileName,
            tokenizerUrl: DemoAssetLinks.smollm2Tokenizer
        ),
        ModelAssetCard(
            name: "SmolVLM (256M)",
            description: "Vision Language Model for image analysis.",
            fileName: DemoAssetLinks.smolvlmFileName,
            url: DemoAssetLinks.smolvlmInt8,
            tokenizerName: DemoAssetLinks.smolvlmTokenizerFileName,
            tokenizerUrl: DemoAssetLinks.smolvlmTokenizer
        )
    ]
    
    var body: some View {
        NavigationView {
            List(availableModels) { model in
                VStack(alignment: .leading, spacing: 8) {
                    Text(model.name)
                        .font(.headline)
                    Text(model.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    HStack {
                        if let size = fileSizes[model.name] {
                            Text("Downloaded (\(size))")
                                .font(.footnote)
                                .foregroundColor(.green)
                            Spacer()
                            Button("Delete", role: .destructive) {
                                deleteModel(model)
                            }
                            .buttonStyle(.borderedProminent)
                        } else {
                            if isDownloading[model.name] == true {
                                ProgressView(value: downloadProgress[model.name] ?? 0.0)
                                    .progressViewStyle(.linear)
                                    .frame(maxWidth: .infinity)
                            } else {
                                Text("Not Downloaded")
                                    .font(.footnote)
                                    .foregroundColor(.red)
                                Spacer()
                                Button("Download") {
                                    Task { await downloadModel(model) }
                                }
                                .buttonStyle(.borderedProminent)
                            }
                        }
                    }
                    .padding(.top, 4)
                }
                .padding(.vertical, 4)
            }
            .navigationTitle("Model Hub")
            .onAppear {
                refreshSizes()
            }
            .alert("Download Error", isPresented: .constant(errorMessage != nil)) {
                Button("OK") { errorMessage = nil }
            } message: {
                Text(errorMessage ?? "")
            }
        }
    }
    
    private func refreshSizes() {
        for model in availableModels {
            let url = RemoteAssets.existingDocumentsFile(fileName: model.fileName)
            let size = RemoteAssets.fileSizeString(url: url)
            fileSizes[model.name] = size
        }
    }
    
    private func deleteModel(_ model: ModelAssetCard) {
        RemoteAssets.removeDocumentsFile(fileName: model.fileName)
        if let t = model.tokenizerName {
            RemoteAssets.removeDocumentsFile(fileName: t)
        }
        refreshSizes()
    }
    
    @MainActor
    private func downloadModel(_ model: ModelAssetCard) async {
        isDownloading[model.name] = true
        downloadProgress[model.name] = 0.0
        
        do {
            _ = try await RemoteAssets.downloadToDocuments(from: model.url, fileName: model.fileName) { p in
                DispatchQueue.main.async {
                    self.downloadProgress[model.name] = p.fraction
                }
            }
            
            if let tUrl = model.tokenizerUrl, let tName = model.tokenizerName {
                _ = try await RemoteAssets.downloadToDocuments(from: tUrl, fileName: tName)
            }
            
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isDownloading[model.name] = false
        refreshSizes()
    }
}
