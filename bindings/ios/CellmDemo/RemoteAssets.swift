import Foundation

enum DemoAssetLinks {
    static let smollm2Int8 = "https://github.com/jeffasante/cellm/blob/main/models/smollm2-135m-int8.cellm"
    static let smolvlmInt8 = "https://github.com/jeffasante/cellm/blob/main/models/smolvlm-256m-int8.cellm"
    static let rococoImage = "https://github.com/jeffasante/cellm/blob/main/models/test_images/rococo_1.jpg"
    static let smollm2Tokenizer = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/tokenizer.json"
    static let smollm2FileName = "smollm2-135m-int8.cellm"
    static let smollm2TokenizerFileName = "tokenizer-smollm2-135m.json"
    static let smolvlmFileName = "smolvlm-256m-int8.cellm"
    static let rococoFileName = "rococo_1.jpg"
}

enum RemoteAssets {
    static func documentsURL(fileName: String) -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent(fileName)
    }

    static func existingDocumentsFile(fileName: String) -> URL? {
        let url = documentsURL(fileName: fileName)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    static func removeDocumentsFile(fileName: String) {
        let url = documentsURL(fileName: fileName)
        if FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.removeItem(at: url)
        }
    }

    static func fileSizeString(url: URL?) -> String? {
        guard let url else { return nil }
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let bytes = attrs[.size] as? NSNumber else { return nil }
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes.int64Value)
    }

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

    static func candidateURLs(from raw: String) throws -> [URL] {
        guard let url = URL(string: raw) else {
            throw CellmError.message("Invalid URL: \(raw)")
        }

        if url.host == "github.com" {
            let parts = url.pathComponents.filter { $0 != "/" }
            if parts.count >= 5, parts[2] == "blob" {
                let owner = parts[0]
                let repo = parts[1]
                let ref = parts[3]
                let rest = parts.dropFirst(4).joined(separator: "/")
                let rawURL = URL(string: "https://raw.githubusercontent.com/\(owner)/\(repo)/\(ref)/\(rest)")!
                let mediaURL = URL(string: "https://media.githubusercontent.com/media/\(owner)/\(repo)/\(ref)/\(rest)")!
                return [rawURL, mediaURL]
            }
        }

        return [url]
    }

    static func downloadToDocuments(from rawURL: String, fileName: String? = nil) async throws -> URL {
        let urls = try candidateURLs(from: rawURL)
        let targetName = fileName ?? urls[0].lastPathComponent

        var downloadedData: Data?
        var lastError = "Download failed"
        for url in urls {
            do {
                let (data, response) = try await URLSession.shared.data(from: url)
                if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                    lastError = "Download failed: HTTP \(http.statusCode)"
                    continue
                }
                if isLikelyHTML(data) {
                    lastError = "Download returned HTML instead of model data"
                    continue
                }
                if isLfsPointer(data) {
                    lastError = "Download returned Git LFS pointer, trying fallback"
                    continue
                }
                if targetName.lowercased().hasSuffix(".cellm") && !hasCellmMagic(data) {
                    lastError = "Downloaded file is not a valid .cellm binary"
                    continue
                }
                downloadedData = data
                break
            } catch {
                lastError = String(describing: error)
            }
        }

        guard let downloadedData else {
            throw CellmError.message(lastError)
        }

        let destURL = documentsURL(fileName: targetName)

        if FileManager.default.fileExists(atPath: destURL.path) {
            try FileManager.default.removeItem(at: destURL)
        }
        try downloadedData.write(to: destURL, options: .atomic)
        return destURL
    }

    static func fetchData(from rawURL: String) async throws -> Data {
        let urls = try candidateURLs(from: rawURL)

        var lastError = "Download failed"
        for url in urls {
            do {
                let (data, response) = try await URLSession.shared.data(from: url)
                if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                    lastError = "Download failed: HTTP \(http.statusCode)"
                    continue
                }
                if isLikelyHTML(data) {
                    lastError = "Download returned HTML instead of file data"
                    continue
                }
                return data
            } catch {
                lastError = String(describing: error)
            }
        }
        throw CellmError.message(lastError)
    }

    private static func hasCellmMagic(_ data: Data) -> Bool {
        guard data.count >= 5 else { return false }
        return Array(data.prefix(5)) == [0x43, 0x45, 0x4C, 0x4C, 0x4D] // CELLM
    }

    private static func isLfsPointer(_ data: Data) -> Bool {
        guard let text = String(data: data.prefix(256), encoding: .utf8) else { return false }
        return text.contains("git-lfs.github.com/spec/v1")
    }

    private static func isLikelyHTML(_ data: Data) -> Bool {
        guard let text = String(data: data.prefix(256), encoding: .utf8)?.lowercased() else { return false }
        return text.contains("<!doctype html") || text.contains("<html")
    }
}
