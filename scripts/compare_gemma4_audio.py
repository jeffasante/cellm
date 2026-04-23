#!/usr/bin/env python3
"""Compare Gemma4 audio encoder intermediate tensors with Rust `audio-direct` dumps.

Usage:
  ./scripts/compare_gemma4_audio.py --model models/gemma-4-E2B-it/ --audio /tmp/test_audio.wav

This script runs under a Python venv with torch and transformers installed (e.g. .venv-hf).
It prints min/max/mean/std for intermediate tensors using labels that mirror the Rust `AUDIO_STATS` log.
"""

import argparse
import wave
import numpy as np
import torch
from transformers import Gemma4AudioFeatureExtractor, Gemma4ForConditionalGeneration


def load_wav(path):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def stats(tensor, name):
    a = tensor.detach().cpu().numpy()
    print(f"HF_STATS {name}: shape={a.shape} min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f} std={a.std():.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--audio', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    samples, sr = load_wav(args.audio)
    print(f"Loaded audio: n={len(samples)} sr={sr}")

    fe = Gemma4AudioFeatureExtractor.from_pretrained(args.model)
    # Use HF extractor to create input features
    result = fe(samples, sampling_rate=sr, return_tensors='pt')
    mel = result['input_audio_embeds']  # shape [1, T, F]
    stats(mel, 'mel_features')

    device = torch.device(args.device)
    model = Gemma4ForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    # Hook internal modules by running a forward and instrumenting
    # This relies on the HF model naming; adjust if mismatch.

    # 1) Pass mel through the model's audio processor if available
    # Some HF variants embed audio processing inside the processor; we already computed 'mel'.
    mel = mel.to(device)

    # 2) Forward through audio tower layers manually if available
    # Try to access model.audio_tower or model.gemma4_audio_tower
    audio_tower = None
    for name in ['audio_tower', 'gemma4_audio_tower', 'audio_encoder']:
        audio_tower = getattr(model, name, None)
        if audio_tower is not None:
            print('Found audio tower:', name)
            break

    if audio_tower is None:
        # Try to run model's forward with audio inputs and capture outputs
        print('audio_tower not found; running model with audio inputs via processor')
        # Build a minimal input dict via processor (text empty)
        inputs = fe.apply_audio_template(mel.squeeze(0)).unsqueeze(0) if hasattr(fe, 'apply_audio_template') else None
        # Fall back to high-level call
        try:
            inputs = fe(samples, sampling_rate=sr, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, return_dict=True)
            # Try to find audio-related tensors in the model outputs
            if hasattr(out, 'audio_outputs'):
                stats(out.audio_outputs, 'D_out_proj')
            else:
                print('No audio_outputs in model forward output; please inspect model structure.')
            return
        except Exception as e:
            print('Failed high-level forward:', e)
            return

    # If audio_tower is present, attempt to forward mel and capture intermediate tensors
    hooks = {}

    def make_hook(name):
        def hook(module, inp, out):
            # Choose first tensor-like object to summarize
            t = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (list, tuple)) and isinstance(out[0], torch.Tensor) else None)
            if t is not None:
                stats(t, name)
        return hook

    # Register hooks on likely submodules
    for nm, mod in audio_tower.named_modules():
        if any(k in nm for k in ['subsample', 'conv', 'proj', 'rmsnorm', 'conformer', 'ffn', 'self_attn']):
            hooks[nm] = mod.register_forward_hook(make_hook(f'AT_{nm}'))

    # Run forward
    try:
        with torch.no_grad():
            # audio_tower expects features shaped [B, T, F]
            out = audio_tower(mel.to(device))
            # final audio embedding projection
            if hasattr(model, 'embed_audio'):
                emb = model.embed_audio(out)
                stats(emb, 'E_embed_out')
            else:
                stats(out, 'D_out_proj')
    finally:
        for h in hooks.values():
            h.remove()


if __name__ == '__main__':
    main()
