# Phase 2 implementation notes

Records concrete decisions made while actually building the single-expert baseline
that `docs/expert-head-design.md` described architecturally but didn't spell out at
the code level. Read alongside that doc, not instead of it.

## Confirmed encoder facts (were assumptions before this phase)

Loading `chandar-lab/NeoBERT` live confirmed: 768 hidden size, 28 layers, 12 attention
heads, 221.7M encoder parameters (the advertised 250M includes an MLM decoder head we
don't load), `trust_remote_code=True` required, and the modeling file hard-requires
`xformers` to be *importable* even though its compiled CUDA extensions aren't needed --
on this CPU-only machine it falls back to a plain-Python attention path with a warning,
not an error. Worth knowing before anyone tries to strip `xformers` as an "unused"
dependency.

## How the option query gets built (not fully specified in expert-head-design.md)

The design doc says each option's query vector should be "question-aware, not just
label-aware" but didn't specify the exact construction. Implemented as: the option's
rendered text (`rendering.py`) and the question's `instructions` are tokenized together
as a sentence pair (`tokenizer(instructions, option_text)`), then mean-pooled and
type-conditioned in `option_encoder.py`. This is a standard, generic sentence-pair
encoding technique (not anyone's proprietary mechanism) and was the natural way to
make the query aware of *what's being asked*, not just which option label it represents.

## Training batches independently, not per-state

`expert-head-design.md`'s "encode the state once" efficiency claim is about *serving*
multiple questions in one API call cheaply -- it is not a training-time requirement.
`training/dataset.py` batches across (state, question) pairs independently, re-encoding
the state per training example. This is simpler and is standard practice for training;
the serving-time optimization is Phase 6 work, tracked in `docs/local-serving.md`, not
something Phase 2 needed to implement to be correct.

## Reward wiring for a single head

`docs/reward-design.md`'s per-expert auxiliary reward and the gate's load-balancing
loss are both MoE-specific and don't apply yet -- there's one head, no gate. Phase 2's
`compute_loss` implements exactly the two things that *do* apply to a single head:
Brier/RPS dispatch by question type, and the optional MMCE calibration term (off by
default, `lambda_calib=0.0`, pending the empirical validation `reward-design.md`
already flagged as unresolved).

## What "Phase 2 complete" means here, precisely

`scripts/train.py` proves the full pipeline -- real NeoBERT weights, real data pulled
by the Phase 1 loaders, real Brier/RPS loss -- runs end-to-end without shape errors,
produces finite loss, and backpropagates finite gradients. It is deliberately not a
convergence run (tiny step count, truncated sequences, CPU-only). Reading its loss
curve as "how accurate is the model" would be a mistake; the correct reading is "does
the pipeline work end to end," which is what this phase was scoped to establish before
Phase 3 adds MoE complexity on top. Real training needs rented GPU compute per
`docs/tech-stack.md`'s `accelerate` entry -- not attempted on this machine.
