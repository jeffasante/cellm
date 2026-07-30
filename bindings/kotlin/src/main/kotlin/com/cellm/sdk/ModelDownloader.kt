package com.cellm.sdk

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

object ModelDownloader {

    val availableModels = listOf(
        ModelSpec(
            id = "qwen2.5-0.5b-int8",
            displayName = "Qwen 2.5 0.5B (Int8)",
            description = "494M parameters, int8 quantized. Fast CPU inference.",
            sizeMb = 620,
            modelUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm?download=true",
            tokenizerUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/qwen2.5-0.5b-int8-v1/tokenizer.json?download=true",
            tokenizerConfigUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/qwen2.5-0.5b-int8-v1/tokenizer_config.json?download=true"
        ),
        ModelSpec(
            id = "smollm2-360m-int8",
            displayName = "SmolLM2 360M (Int8)",
            description = "360M parameters, int8. Good for quick testing.",
            sizeMb = 450,
            modelUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/smollm2-360m-int8-v1/smollm2-360m-int8-v1.cellm?download=true",
            tokenizerUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/smollm2-360m-int8-v1/tokenizer.json?download=true",
            tokenizerConfigUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/smollm2-360m-int8-v1/tokenizer_config.json?download=true"
        ),
        ModelSpec(
            id = "gemma-4-E2B-it-int4",
            displayName = "Gemma 4 2B (Int4)",
            description = "2B parameters, int4 quantized. Best quality on device.",
            sizeMb = 3400,
            modelUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd?download=true",
            tokenizerUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json?download=true",
            tokenizerConfigUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/gemma-4-E2B-it-int4-aggr-v5/tokenizer_config.json?download=true"
        ),
        ModelSpec(
            id = "lfm2.5-350m",
            displayName = "LFM 2.5 350M",
            description = "350M Liquid Foundation Model. Fast context processing.",
            sizeMb = 440,
            modelUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/lfm2.5-350m-v1/lfm2.5-350m-v1.cellm?download=true",
            tokenizerUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/lfm2.5-350m-v1/tokenizer.json?download=true",
            tokenizerConfigUrl = null
        ),
        ModelSpec(
            id = "lfm2.5-350m-int8",
            displayName = "LFM 2.5 350M (Int8)",
            description = "350M Liquid Foundation Model, int8. Fastest CPU decode.",
            sizeMb = 339,
            modelUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/lfm2.5-350m-int8-v1/lfm2.5-350m-int8-v1.cellm?download=true",
            tokenizerUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/lfm2.5-350m-int8-v1/tokenizer.json?download=true",
            tokenizerConfigUrl = null
        ),
        ModelSpec(
            id = "smolvlm-256m-instruct",
            displayName = "SmolVLM 256M (VLM)",
            description = "Vision-language model. Accepts image + text input.",
            sizeMb = 520,
            modelUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/smolvlm-256m-instruct-int8-v1/smolvlm-256m-instruct-int8-v1.cellm?download=true",
            tokenizerUrl = "https://huggingface.co/jeffasante/cellm-models/resolve/main/smolvlm-256m-instruct-int8-v1/tokenizer.json?download=true",
            tokenizerConfigUrl = null
        )
    )

    data class ModelSpec(
        val id: String,
        val displayName: String,
        val description: String,
        val sizeMb: Int,
        val modelUrl: String,
        val tokenizerUrl: String,
        val tokenizerConfigUrl: String?
    )

    data class DownloadProgress(
        val fraction: Float,
        val bytesReceived: Long,
        val bytesExpected: Long
    )

    data class ModelFiles(
        val modelPath: String,
        val tokenizerPath: String,
        val tokenizerConfigPath: String?
    )

    fun modelsDir(context: Context): File {
        val canonicalDir = File("/data/data/${context.packageName}/files/cellm-models")
        canonicalDir.mkdirs()
        return canonicalDir
    }

    fun isDownloaded(context: Context, model: ModelSpec): Boolean {
        val dir = File(modelsDir(context), model.id)
        val modelFile = File(dir, model.modelUrl.substringAfterLast("/").replace("?download=true", ""))
        val tokenizerFile = File(dir, "tokenizer.json")
        return modelFile.exists() && tokenizerFile.exists()
    }

    fun getModelFiles(context: Context, model: ModelSpec): ModelFiles? {
        val dir = File(modelsDir(context), model.id)
        val modelFile = File(dir, model.modelUrl.substringAfterLast("/").replace("?download=true", ""))
        val tokenizerFile = File(dir, "tokenizer.json")
        val configFile = File(dir, "tokenizer_config.json")
        if (!modelFile.exists() || !tokenizerFile.exists()) return null
        return ModelFiles(
            modelPath = modelFile.absolutePath,
            tokenizerPath = tokenizerFile.absolutePath,
            tokenizerConfigPath = if (configFile.exists()) configFile.absolutePath else null
        )
    }

    suspend fun download(
        context: Context,
        model: ModelSpec,
        onProgress: (DownloadProgress) -> Unit = {}
    ): ModelFiles = withContext(Dispatchers.IO) {
        val dir = File(modelsDir(context), model.id)
        dir.mkdirs()

        data class FileToDownload(
            val url: String,
            val localName: String,
            val description: String
        )

        val files = mutableListOf<FileToDownload>()
        files.add(FileToDownload(
            model.modelUrl,
            model.modelUrl.substringAfterLast("/").replace("?download=true", ""),
            "model weights"
        ))
        files.add(FileToDownload(
            model.tokenizerUrl,
            "tokenizer.json",
            "tokenizer"
        ))
        if (model.tokenizerConfigUrl != null) {
            files.add(FileToDownload(
                model.tokenizerConfigUrl,
                "tokenizer_config.json",
                "tokenizer config"
            ))
        }

        var completedSteps = 0
        val totalSteps = files.size

        for (file in files) {
            val dest = File(dir, file.localName)
            if (dest.exists() && dest.length() > 0) {
                completedSteps++
                onProgress(DownloadProgress(
                    completedSteps.toFloat() / totalSteps,
                    dest.length(),
                    dest.length()
                ))
                continue
            }

            downloadFile(file.url, dest) { received, total ->
                val overallFraction = (completedSteps + (if (total > 0) received.toFloat() / total else 0f)) / totalSteps
                onProgress(DownloadProgress(overallFraction, received, total))
            }
            dest.setReadable(true, false)
            completedSteps++
            onProgress(DownloadProgress(
                completedSteps.toFloat() / totalSteps,
                dest.length(),
                dest.length()
            ))
        }

        getModelFiles(context, model)!!
    }

    private fun downloadFile(
        urlString: String,
        dest: File,
        onProgress: (received: Long, total: Long) -> Unit
    ) {
        var lastError = "Download failed"

        val urlsToTry = listOf(
            urlString,
            urlString.replace("?download=true", ""),
            urlString.replace("blob/", "resolve/")
        ).distinct()

        for (url in urlsToTry) {
            try {
                val connection = URL(url).openConnection() as HttpURLConnection
                connection.connectTimeout = 30000
                connection.readTimeout = 300000
                connection.instanceFollowRedirects = true
                connection.setRequestProperty("User-Agent", "cellm-android")

                val responseCode = connection.responseCode
                if (responseCode !in 200..299) {
                    lastError = "HTTP $responseCode for $url"
                    connection.disconnect()
                    continue
                }

                val totalBytes = connection.contentLengthLong
                val input = connection.inputStream
                val output = FileOutputStream(dest)

                val buffer = ByteArray(65536)
                var receivedBytes = 0L
                var bytesRead: Int

                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    receivedBytes += bytesRead
                    if (totalBytes > 0 && receivedBytes % (256 * 1024) == 0L) {
                        onProgress(receivedBytes, totalBytes)
                    }
                }

                output.close()
                input.close()
                connection.disconnect()

                if (totalBytes > 0) {
                    onProgress(receivedBytes, totalBytes)
                }

                if (isLikelyHtml(dest)) {
                    dest.delete()
                    lastError = "Download returned HTML instead of model data"
                    continue
                }

                return
            } catch (e: Exception) {
                lastError = e.message ?: "Unknown error"
                dest.delete()
            }
        }

        throw RuntimeException(lastError)
    }

    private fun isLikelyHtml(file: File): Boolean {
        if (file.length() < 10) return false
        val header = ByteArray(minOf(file.length(), 512).toInt())
        file.inputStream().use { it.read(header) }
        val text = String(header, Charsets.UTF_8).lowercase()
        return text.contains("<!doctype html") || text.contains("<html")
    }
}
