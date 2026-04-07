import Foundation

enum DemoAssetLinks {
    static let qwen35Stable = "https://github.com/jeffasante/cellm/blob/main/models/qwen3.5-0.8b.cellm"
    static let qwen35CompactInt4TextOnly = "https://github.com/jeffasante/cellm/blob/main/models/qwen3.5-0.8b-int4-textonly.cellm"
    static let qwen35Tokenizer = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json"
    static let qwen35TokenizerConfig = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer_config.json"
    static let qwen35Dir = "samples/qwen3.5-0.8b"
    static let qwen35StableFileName = "\(qwen35Dir)/qwen3.5-0.8b.cellm"
    static let qwen35CompactFileName = "\(qwen35Dir)/qwen3.5-0.8b-int4-textonly.cellm"
    static let qwen35TokenizerFileName = "\(qwen35Dir)/tokenizer-qwen3.5-0.8b.json"
    static let qwen35TokenizerConfigFileName = "\(qwen35Dir)/tokenizer_config.json"

    static let smollm2Int8 = "https://github.com/jeffasante/cellm/blob/main/models/smollm2-135m-int8.cellm"
    static let smolvlmInt8 = "https://github.com/jeffasante/cellm/blob/main/models/smolvlm-256m-int8.cellm"
    static let rococoImage = "https://github.com/jeffasante/cellm/blob/main/models/test_images/rococo_1.jpg"
    static let smollm2Tokenizer = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/tokenizer.json"
    static let smollm2TokenizerConfig = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/tokenizer_config.json"
    static let smolvlmTokenizer = "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/tokenizer.json"
    static let smolvlmProcessorConfig = "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/processor_config.json"
    static let smolvlmPreprocessorConfig = "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/preprocessor_config.json"
    static let smollm2Dir = "samples/smollm2-135m"
    static let smollm2FileName = "\(smollm2Dir)/smollm2-135m-int8.cellm"
    static let smollm2TokenizerFileName = "\(smollm2Dir)/tokenizer-smollm2-135m.json"
    static let smollm2TokenizerConfigFileName = "\(smollm2Dir)/tokenizer_config.json"

    static let gemma3Int8 = "https://huggingface.co/jeffasante/gemma-3-1b-it-int8-cellm/resolve/main/gemma-3-1b-it-int8.cellmd"
    static let gemma3Tokenizer = "https://huggingface.co/unsloth/gemma-3-1b-it/resolve/main/tokenizer.json"
    static let gemma3TokenizerConfig = "https://huggingface.co/unsloth/gemma-3-1b-it/resolve/main/tokenizer_config.json"
    static let gemma3Dir = "samples/gemma-3-1b-it"
    static let gemma3FileName = "\(gemma3Dir)/gemma-3-1b-it-int8.cellmd"
    static let gemma3TokenizerFileName = "\(gemma3Dir)/tokenizer-gemma-3-1b-it.json"
    static let gemma3TokenizerConfigFileName = "\(gemma3Dir)/tokenizer_config.json"

    static let gemma42p3bLiteRt = "https://huggingface.co/jeffasante/cellm-models/resolve/main/gemma-4-2p3b-it/gemma-4-2p3b-it-litert.cellm"
    static let gemma42p3bTokenizer = "https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/tokenizer.json"
    static let gemma42p3bTokenizerConfig = "https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/tokenizer_config.json"
    static let gemma42p3bDir = "samples/gemma-4-2p3b-it"
    static let gemma42p3bFileName = "\(gemma42p3bDir)/gemma-4-2p3b-it-litert.cellm"
    static let gemma42p3bTokenizerFileName = "\(gemma42p3bDir)/tokenizer-gemma-4-2p3b-it.json"
    static let gemma42p3bTokenizerConfigFileName = "\(gemma42p3bDir)/tokenizer_config.json"

    static let smolvlmFileName = "smolvlm-256m-int8.cellm"
    static let smolvlmTokenizerFileName = "tokenizer-smolvlm-256m.json"
    static let smolvlmProcessorConfigFileName = "processor_config-smolvlm-256m.json"
    static let smolvlmPreprocessorConfigFileName = "preprocessor_config-smolvlm-256m.json"
    static let rococoFileName = "rococo_1.jpg"
}

enum RemoteAssets {
    struct DownloadProgress {
        let fraction: Double
        let bytesReceived: Int64
        let bytesExpected: Int64
    }

    private static func makeDownloadSession() -> URLSession {
        let cfg = URLSessionConfiguration.ephemeral
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.urlCache = nil
        return URLSession(configuration: cfg)
    }

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

        if url.host == "huggingface.co" {
            let parts = url.pathComponents.filter { $0 != "/" }
            if let blobIndex = parts.firstIndex(of: "blob"), blobIndex >= 2, parts.count > blobIndex + 2 {
                let repo = parts[1]
                let ref = parts[blobIndex + 1]
                let rest = parts[(blobIndex + 2)...].joined(separator: "/")
                var resolve = URLComponents()
                resolve.scheme = "https"
                resolve.host = "huggingface.co"
                resolve.path = "/\(parts[0])/\(repo)/resolve/\(ref)/\(rest)"
                var resolveWithDownload = resolve
                resolveWithDownload.queryItems = [URLQueryItem(name: "download", value: "true")]
                var out: [URL] = []
                if let u = resolveWithDownload.url { out.append(u) }
                if let u = resolve.url { out.append(u) }
                out.append(url)
                return out
            }
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

    static func downloadToDocuments(from rawURL: String, fileName: String? = nil, progress: ((DownloadProgress) -> Void)? = nil) async throws -> URL {
        let urls = try candidateURLs(from: rawURL)
        let targetName = fileName ?? urls[0].lastPathComponent

        var lastError = "Download failed"
        for url in urls {
            do {
                let session = makeDownloadSession()
                defer { session.invalidateAndCancel() }
                let (bytes, response) = try await session.bytes(from: url)
                if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                    lastError = "Download failed: HTTP \(http.statusCode)"
                    continue
                }
                let tempURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString)
                    .appendingPathExtension("download")
                FileManager.default.createFile(atPath: tempURL.path, contents: nil)
                let handle = try FileHandle(forWritingTo: tempURL)
                defer {
                    try? handle.close()
                }

                let expected = response.expectedContentLength
                var received: Int64 = 0
                var lastReportedBytes: Int64 = 0
                var chunk: [UInt8] = []
                chunk.reserveCapacity(64 * 1024)
                var prefix = Data()
                prefix.reserveCapacity(512)

                for try await byte in bytes {
                    chunk.append(byte)
                    received += 1
                    if prefix.count < 512 {
                        prefix.append(byte)
                    }
                    if chunk.count >= 64 * 1024 {
                        try handle.write(contentsOf: chunk)
                        chunk.removeAll(keepingCapacity: true)
                    }
                    if expected > 0 {
                        // Throttle progress updates to avoid excessive UI work / task churn.
                        if (received - lastReportedBytes) >= 256 * 1024 || received == expected {
                            lastReportedBytes = received
                            let fraction = min(1.0, Double(received) / Double(expected))
                            progress?(DownloadProgress(
                                fraction: fraction,
                                bytesReceived: received,
                                bytesExpected: expected
                            ))
                        }
                    }
                }
                if !chunk.isEmpty {
                    try handle.write(contentsOf: chunk)
                }
                if expected <= 0 {
                    progress?(DownloadProgress(
                        fraction: 1.0,
                        bytesReceived: received,
                        bytesExpected: expected
                    ))
                }

                if isLikelyHTML(prefix) {
                    lastError = "Download returned HTML instead of model data"
                    try? FileManager.default.removeItem(at: tempURL)
                    continue
                }
                if isLfsPointer(prefix) {
                    lastError = "Download returned Git LFS pointer, trying fallback"
                    try? FileManager.default.removeItem(at: tempURL)
                    continue
                }
                if targetName.lowercased().hasSuffix(".cellm") && !hasCellmMagic(prefix) {
                    lastError = "Downloaded file is not a valid .cellm binary"
                    try? FileManager.default.removeItem(at: tempURL)
                    continue
                }

                let destURL = documentsURL(fileName: targetName)
                let dirURL = destURL.deletingLastPathComponent()
                try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true, attributes: nil)
                if FileManager.default.fileExists(atPath: destURL.path) {
                    try FileManager.default.removeItem(at: destURL)
                }
                try FileManager.default.moveItem(at: tempURL, to: destURL)
                return destURL
            } catch {
                lastError = String(describing: error)
            }
        }

        throw CellmError.message(lastError)
    }

    static func fetchData(from rawURL: String) async throws -> Data {
        let urls = try candidateURLs(from: rawURL)

        var lastError = "Download failed"
        for url in urls {
            do {
                let session = makeDownloadSession()
                defer { session.invalidateAndCancel() }
                let (data, response) = try await session.data(from: url)
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
