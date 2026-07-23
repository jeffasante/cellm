#!/usr/bin/env python3
"""Regression suite for the frozen Llama 3.2 3B affine-Q4 baseline.

Run from the repository root:
    python3 -m unittest discover -s tests/baselines -p 'test_*.py' -v

The inference test requires the release `infer` binary and the local model artifacts.
"""

import hashlib
import json
import mmap
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "models/Llama-3.2-3B-Instruct-Q4-affine-v1.manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cellm_header(handle):
    handle.seek(0)
    if handle.read(5) != b"CELLM":
        raise AssertionError("artifact does not begin with CELLM magic")
    version = handle.read(1)[0]
    header_len = struct.unpack("<I", handle.read(4))[0]
    return version, json.loads(handle.read(header_len))


def load_base_header(handle):
    handle.seek(0)
    prefix = handle.read(16)
    if prefix[:4] != b"BASE":
        raise AssertionError("source does not begin with BASE magic")
    version = struct.unpack("<I", prefix[4:8])[0]
    metadata_len = struct.unpack("<Q", prefix[8:16])[0]
    metadata = json.loads(handle.read(metadata_len))
    blob_start = ((16 + metadata_len + 65535) // 65536) * 65536
    return version, metadata, blob_start


def map_tensor_name(source_name: str):
    if source_name == "embed_tokens.weight":
        return "model.embed_tokens.weight"
    if source_name == "final_norm.weight":
        return "model.norm.weight"
    if source_name.startswith("layers."):
        layer, suffix = source_name[len("layers."):].split(".", 1)
        suffix = {
            "input_norm.weight": "input_layernorm.weight",
            "post_attn_norm.weight": "post_attention_layernorm.weight",
        }.get(suffix, suffix)
        return f"model.layers.{layer}.{suffix}"
    return None


class LlamaAffineV1Baseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MANIFEST_PATH.is_file():
            raise AssertionError(f"baseline manifest is missing: {MANIFEST_PATH}")
        cls.manifest = json.loads(MANIFEST_PATH.read_text())
        cls.artifact = ROOT / cls.manifest["artifact"]["path"]
        cls.source = ROOT / cls.manifest["source_model"]["path"]
        cls.tokenizer = ROOT / cls.manifest["tokenizer"]["path"]
        cls.tokenizer_config = ROOT / cls.manifest["tokenizer"]["config_path"]
        cls.oracle_path = ROOT / cls.manifest["oracle_fixture"]

    def assert_file_contract(self, path: Path, expected_size: int, expected_hash: str):
        self.assertTrue(path.is_file(), f"required baseline file is missing: {path}")
        self.assertEqual(
            path.stat().st_size,
            expected_size,
            f"size changed for {path}; this is not the frozen baseline artifact",
        )
        self.assertEqual(
            sha256(path),
            expected_hash,
            f"SHA-256 changed for {path}; regenerate/version the baseline intentionally",
        )

    def test_01_frozen_artifact_and_inputs_match_manifest(self):
        self.assert_file_contract(
            self.artifact,
            self.manifest["artifact"]["expected_file_size"],
            self.manifest["artifact"]["sha256"],
        )
        self.assert_file_contract(
            self.source,
            self.manifest["source_model"]["expected_file_size"],
            self.manifest["source_model"]["sha256"],
        )
        self.assertEqual(sha256(self.tokenizer), self.manifest["tokenizer"]["sha256"])
        self.assertEqual(
            sha256(self.tokenizer_config), self.manifest["tokenizer"]["config_sha256"]
        )

    def test_02_converter_and_runtime_sources_match_manifest(self):
        conversion = self.manifest["conversion"]
        self.assertEqual(
            sha256(ROOT / conversion["converter_path"]),
            conversion["converter_sha256"],
            "converter source changed; version or regenerate the baseline manifest",
        )
        for relative, expected in self.manifest["runtime"]["source_sha256"].items():
            self.assertEqual(
                sha256(ROOT / relative),
                expected,
                f"runtime baseline source changed: {relative}",
            )

    def test_03_affine_q4_payload_scales_and_biases_are_preserved(self):
        self.assertEqual(self.manifest["conversion"]["group_size"], 64)
        self.assertEqual(
            self.manifest["conversion"]["quantization_scheme"], "unsigned-affine-i4"
        )

        with self.source.open("rb") as source_file, self.artifact.open("rb") as artifact_file:
            source_map = mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ)
            artifact_map = mmap.mmap(artifact_file.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                _, source_header, source_blob = load_base_header(source_file)
                version, cellm_header = load_cellm_header(artifact_file)
                self.assertEqual(version, self.manifest["artifact"]["format_version"])
                tensors = {tensor["name"]: tensor for tensor in cellm_header["tensors"]}
                affine_count = 0

                for source_tensor in source_header["tensors"]:
                    if source_tensor["dtype"] != "base_q4":
                        continue
                    target_name = map_tensor_name(source_tensor["name"])
                    if target_name is None:
                        continue
                    affine_count += 1
                    rows, cols = source_tensor["shape"]
                    group_size = source_tensor["group_size"]
                    self.assertEqual(group_size, 64, source_tensor["name"])
                    groups = (cols + group_size - 1) // group_size
                    packed_bytes = rows * groups * (group_size // 2)
                    source_offset = source_blob + source_tensor["offset"]

                    target = tensors.get(target_name)
                    if target is None:
                        self.fail(f"missing affine weight: {target_name}")
                    self.assertEqual(target["dtype"], "u32", target_name)
                    self.assertEqual(target["shape"], [rows, cols], target_name)
                    self.assertEqual(target["nbytes"], packed_bytes, target_name)
                    target_offset = target["offset_bytes"]
                    self.assertEqual(
                        artifact_map[target_offset:target_offset + packed_bytes],
                        source_map[source_offset:source_offset + packed_bytes],
                        f"packed Q4 payload changed: {target_name}",
                    )

                    base = target_name.removesuffix(".weight")
                    for kind, source_relative in (
                        ("scales", source_tensor["scale_offset"]),
                        ("biases", source_tensor["bias_offset"]),
                    ):
                        parameter = tensors.get(f"{base}.{kind}")
                        if parameter is None:
                            self.fail(f"missing {kind}: {base}")
                        self.assertEqual(parameter["dtype"], "f32", f"{base}.{kind}")
                        self.assertEqual(parameter["shape"], [rows, groups], f"{base}.{kind}")
                        count = rows * groups
                        source_values = np.frombuffer(
                            source_map,
                            dtype="<u2",
                            count=count,
                            offset=source_offset + source_relative,
                        )
                        target_bits = np.frombuffer(
                            artifact_map,
                            dtype="<u4",
                            count=count,
                            offset=parameter["offset_bytes"],
                        )
                        expected_bits = source_values.astype(np.uint32) << np.uint32(16)
                        self.assertTrue(
                            np.array_equal(target_bits, expected_bits),
                            f"BASE bf16 {kind} were not preserved exactly: {base}",
                        )
                        # Release numpy's exported mmap views before closing the maps.
                        del source_values, target_bits, expected_bits

                self.assertEqual(
                    affine_count,
                    self.manifest["artifact"]["tensor_dtype_counts"]["u32"],
                    "unexpected affine-Q4 tensor count",
                )
                self.assertEqual(len(cellm_header["tensors"]), 646)
            finally:
                artifact_map.close()
                source_map.close()

    def test_04_interleaved_scaled_rope_and_chat_metadata(self):
        with self.artifact.open("rb") as handle:
            _, header = load_cellm_header(handle)
        config = self.manifest["model_config"]
        scaling = config["rope_scaling"]
        self.assertIs(header.get("rope_interleaved"), True, "interleaved RoPE is not selected")
        self.assertEqual(header.get("rope_scaling_type"), "llama3")
        self.assertEqual(header.get("rope_scaling_factor"), scaling["factor"])
        self.assertEqual(
            header.get("rope_scaling_original_max_position_embeddings"),
            scaling["original_max_position_embeddings"],
        )
        self.assertEqual(header.get("rope_scaling_low_freq_factor"), scaling["low_freq_factor"])
        self.assertEqual(header.get("rope_scaling_high_freq_factor"), scaling["high_freq_factor"])

        tokenizer_config = json.loads(self.tokenizer_config.read_text())
        template = tokenizer_config.get("chat_template", "")
        for marker in self.manifest["tokenizer"]["chat_template_markers"]:
            self.assertIn(marker, template, f"chat template marker missing: {marker}")

    def test_05_cpu_chat_tokens_logits_and_eot_stop_match_oracle(self):
        infer = ROOT / "target/release/infer"
        self.assertTrue(
            infer.is_file(),
            "release infer binary is missing; run `cargo build --release --bin infer`",
        )
        oracle = json.loads(self.oracle_path.read_text())
        with tempfile.TemporaryDirectory() as temp_dir:
            actual_path = Path(temp_dir) / "actual.json"
            command = [
                str(infer),
                "--model", str(self.artifact),
                "--tokenizer", str(self.tokenizer),
                "--prompt", oracle["prompt"],
                "--chat",
                "--gen", str(oracle["inference"]["max_new_tokens"]),
                "--temperature", str(oracle["inference"]["temperature"]),
                "--top-k", str(oracle["inference"]["top_k"]),
                "--repeat-penalty", str(oracle["inference"]["repeat_penalty"]),
                "--backend", oracle["inference"]["backend"],
                "--kv-encoding", oracle["inference"]["kv_encoding"],
                "--oracle-trace", str(actual_path),
            ]
            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"baseline inference failed:\n{result.stdout}",
            )
            self.assertTrue(actual_path.is_file(), "infer did not write the oracle trace")
            actual = json.loads(actual_path.read_text())

        for field in (
            "prompt_token_ids",
            "generated_token_ids",
            "output_text",
            "stop_reason",
            "stop_token_id",
        ):
            self.assertEqual(actual[field], oracle[field], f"oracle field changed: {field}")
        self.assertEqual(actual["stop_reason"], "eot_id", "generation hit a limit instead of EOT")
        self.assertLess(
            len(actual["generated_token_ids"]),
            oracle["inference"]["max_new_tokens"],
            "generation exhausted max tokens; EOT termination regressed",
        )
        self.assertEqual(len(actual["initial_top_logits"]), len(oracle["initial_top_logits"]))
        for index, (actual_logit, expected_logit) in enumerate(
            zip(actual["initial_top_logits"], oracle["initial_top_logits"])
        ):
            self.assertEqual(
                actual_logit["token_id"],
                expected_logit["token_id"],
                f"initial top-logit token changed at rank {index}",
            )
            self.assertAlmostEqual(
                actual_logit["logit"],
                expected_logit["logit"],
                places=4,
                msg=f"initial logit changed at rank {index}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
