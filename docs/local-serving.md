# Local / offline serving

Requirement, not a stretch goal: the finished model must run fully offline (no network
calls) on consumer hardware, matching what Laya already demonstrated. This document is
about *serving* the trained model locally — training still needs real GPU compute
(see `docs/architecture-decisions.md`); those are separate concerns.

## Reference hardware

Checked directly rather than assumed, since "does it run locally" is a hardware
question:

| | Reference machine (this project) | Laya's own published benchmark |
|---|---|---|
| CPU | 12th Gen Intel i5-1235U, 10 cores / 12 threads | Apple M-series (MacBook Air) |
| RAM | 15.7 GB | 16 GB |
| GPU | Intel Iris Xe (integrated, no CUDA) | Apple integrated GPU (no CUDA) |

Both are CPU-bound, no discrete NVIDIA card. Laya reports ~32.8ms single-question
inference with zero network calls on hardware in this exact tier — that's the bar.

## Why the architecture doesn't fight this

- **NeoBERT is 250M params (~1GB fp32 weights)** — smaller than Laya's own English
  checkpoint (ModernBERT-large, 421M). Our base footprint starts lighter, not heavier.
- **MoE experts don't multiply the footprint.** Each expert is a small head (comparable
  to Laya's 2-layer transformer head), not a separate encoder. Six experts loaded
  together add a few percent over the encoder's size, not 6x — the shared-encoder
  design is exactly what makes this affordable.
- **15.7GB of RAM is generous headroom** even for all six experts plus the gate loaded
  simultaneously; Laya's own Node.js/ONNX port targets ~2GB for a comparable single-
  checkpoint setup.

## Serving path (Phase 6)

1. **Export to ONNX.** Same approach Laya's `receptron/laya` port uses: trace the
   PyTorch model (encoder + gate + expert heads) into one ONNX graph, verify numerical
   parity against the PyTorch forward pass (Laya's own export reports ~1e-5 max logit
   difference — that's the bar for our export too).
2. **CPU-first execution providers.** Target ONNX Runtime's CPU provider as the
   baseline (works everywhere, no driver dependency), DirectML as an optional
   Windows-specific speedup path (relevant given the reference machine's Intel Iris Xe
   supports DirectML but not CUDA).
3. **Quantization.** fp16 or int8 post-training quantization to cut memory and improve
   CPU throughput, following the same measurement discipline as anywhere else in this
   project: benchmark accuracy/calibration loss from quantization explicitly, don't
   assume it's free.
4. **Package for redistribution.** A minimal runtime (weights + tokenizer + config, no
   PyTorch/Python dependency needed for inference) the way Laya's ONNX bundle
   (`laya.onnx`, `laya.onnx.data`, `laya_config.json`, tokenizer files) ships — so
   self-hosting doesn't require the full training stack.

## Open questions

- Whether all 6 (eventually more) expert heads should always load together, or lazy-
  load per first-use — matters more once expert count grows past v1's six domains.
- Whether local serving needs its own stripped-down feature-flag behavior (e.g. is
  `explain`'s attention-based rationale still "free" after quantization, or does
  quantization degrade the attention weights enough to make the rationale noisier?)
  — needs empirical testing once Phase 4 and this phase overlap.
