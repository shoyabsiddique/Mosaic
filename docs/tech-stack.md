# Library / tech stack

Grouped by what part of the plan each library serves. Where Laya's own published code
already proved a library works for this exact model family (encoder + safetensors +
ONNX serving), that's noted — it de-risks the choice rather than being a reason to
copy their design.

## Model & training

- **PyTorch** (2.x) — model implementation and training. Same as Laya/Jev's
  presumed stack; no real alternative for this kind of custom head/gate work.
- **transformers** (Hugging Face) — loads NeoBERT's pretrained weights/config and
  tokenizer. Laya uses this same library for ModernBERT, so the loading/fine-tuning
  path is proven for this class of encoder.
- **tokenizers** (Hugging Face, transitive via transformers) — fast tokenization.
- **safetensors** — checkpoint format. Same choice Laya made; safe (no arbitrary code
  execution on load, unlike pickle) and fast.
- **accelerate** (Hugging Face) — simplifies mixed-precision / multi-GPU training loop
  management, relevant since Phase 2/3 training runs on rented cloud GPUs, not the
  local machine (see `local-serving.md` for the local-vs-cloud split).
- **datasets** (Hugging Face) — consistent loading/streaming interface across the
  verified Tier-1 datasets (Kaggle CSVs, HF-hosted sets, CFPB bulk data), instead of
  hand-rolling a loader per source format.
- **einops** — clearer tensor reshape/attention code for the cross-attention expert
  heads (`expert-head-design.md`) than raw `.view()`/`.permute()` chains.

## Calibration & metrics

- **numpy** / **scipy** — numerical utilities, calibration metrics.
- **scikit-learn** — standard AUROC/precision-recall utilities (Laya hand-rolled these
  in `rl_common.py`; using sklearn is less code to maintain and equally correct).
- **Custom implementation, no library**: Brier score and RPS (`reward-design.md`) are
  a few lines each, same as Laya's own `proper_reward` — no dependency needed.
- **Custom implementation, no maintained library**: MMCE (`reward-design.md`) —
  implemented from the paper (Kumar et al., ICML 2018) directly, since there's no
  actively maintained package for it.

## Serving

- **onnx** / **onnxruntime** — export format and inference runtime
  (`local-serving.md`). Same approach Laya's own `receptron/laya` port uses, with a
  confirmed ~1e-5 numerical parity bar to match.
- **onnxruntime-directml** — Windows-specific execution provider, relevant given the
  reference machine's Intel Iris Xe has no CUDA path but does support DirectML.
- **FastAPI** — HTTP API layer. Laya's own example server (`examples/server.py`) also
  uses FastAPI — a validated choice for this exact kind of typed-request/response API.
- **uvicorn** — ASGI server to run FastAPI.
- **pydantic** — request/response schema validation (ships with FastAPI); models the
  `choice`/`score`/`noul` question types and the `explain`/`sla_threshold` feature
  flags as typed schemas rather than untyped dicts.

## Config & experiment tracking

- **OmegaConf + plain YAML** — configuration for encoder/head/gate hyperparameters and
  the per-domain expert registry. Simpler than Hydra for this project's scale; revisit
  if the config surface grows enough to need Hydra's composition features.
- **Weights & Biases (wandb)** — experiment tracking across the ablations Phase 2/3
  will actually need to run (encoder validation, soft-vs-hard routing, reward
  hyperparameters `lambda_expert`/`lambda_calib`/`lambda_balance`). Not load-bearing —
  MLflow would work equally well; wandb chosen for lower setup friction.

## Data collection

- **requests** / **httpx** — pulling data from API-based sources (NVD's REST API).
- **pandas** — tabular cleanup for the CSV-based datasets (CFPB, Kaggle sets).

## Dev tooling

- **pytest** — test suite, matching the style already seen in Laya's own
  `tests/test_training.py` (plain assertions, not a heavy framework).
- **ruff** — linting and formatting in one tool.
- **mypy** — type checking, given `types.py` and pydantic schemas are meant to be
  the source of truth for data shapes throughout the pipeline.

## Explicitly not used

- **No LoRA / PEFT libraries** — we're training from a pretrained encoder plus
  from-scratch heads, not fine-tuning a frozen LLM with adapters. Not the right tool
  for this architecture.
- **No vLLM / TGI-style serving frameworks** — those are built for autoregressive
  generation; this model does a single forward pass per request, and ONNX Runtime
  is the correct-weight tool, not an LLM-serving stack.
