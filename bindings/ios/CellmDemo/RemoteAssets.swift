import Foundation

enum DemoAssetLinks {
    static let smollm2Int8 = "https://github.com/jeffasante/cellm/blob/main/models/smollm2-135m-int8.cellm"
    static let smolvlmInt8 = "https://github.com/jeffasante/cellm/blob/main/models/smolvlm-256m-int8.cellm"
    static let rococoImage = "https://github.com/jeffasante/cellm/blob/main/models/test_images/rococo_1.jpg"
    static let smollm2Tokenizer = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/tokenizer.json"
}

enum RemoteAssets {
    static func normalizeGitHubBlobURL(_ raw: String) throws -> URL {
        guard let url = URL(string: raw) else {
            throw CellmError.message("Invalid URL: \(raw)")
        }
        guard url.host == "github.com" else {
            return url
        }

        let parts = url.pathComponents.filter { $0 != "/" }
        guard parts.count >= 5, parts[2] == "blob" else {
            return url
        }

        let owner = parts[0]
        let repo = parts[1]
        let ref = parts[3]
        let rest = parts.dropFirst(4).joined(separator: "/")
        guard let rawURL = URL(string: "https://raw.githubusercontent.com/\(owner)/\(repo)/\(ref)/\(rest)") else {
            throw CellmError.message("Failed to convert GitHub URL to raw URL")
        }
        return rawURL
    }

    static func downloadToDocuments(from rawURL: String, fileName: String? = nil) async throws -> URL {
        let url = try normalizeGitHubBlobURL(rawURL)
        let (tmpURL, response) = try await URLSession.shared.download(from: url)

        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw CellmError.message("Download failed: HTTP \(http.statusCode)")
        }

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let targetName = fileName ?? url.lastPathComponent
        let destURL = docs.appendingPathComponent(targetName)

        if FileManager.default.fileExists(atPath: destURL.path) {
            try FileManager.default.removeItem(at: destURL)
        }
        try FileManager.default.moveItem(at: tmpURL, to: destURL)
        return destURL
    }

    static func fetchData(from rawURL: String) async throws -> Data {
        let url = try normalizeGitHubBlobURL(rawURL)
        let (data, response) = try await URLSession.shared.data(from: url)

        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw CellmError.message("Download failed: HTTP \(http.statusCode)")
        }
        return data
    }
}
