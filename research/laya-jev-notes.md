# Notes: how Jev and Laya work (for understanding, not for copying)

Both launched September 2026 as "System 1 decision models": given a state (text, ticket,
email, JSON) and typed questions (`choice` / `score` / `noul`), return calibrated
probabilities in one forward pass. No text generation, nothing to hallucinate.

## Jev (TypeSafe AI) — closed source

Only marketing-level detail is public: "a new model architecture, a parallel sampler,
and a training method called Reinforcement Learning for Calibrated Decisions (RLCD)."
No parameter count, tokenization, or architecture diagram published.
Source: https://typesafe.ai/blog/introducing-system-one-models-and-jev

## Laya (Convai Innovations) — Apache 2.0, fully open

Source read directly from Hugging Face (`convaiinnovations/laya/rl_common.py`,
`rl_agent_api.py`, `rl_agent_config.json`) and GitHub (`NandhaKishorM/laya`).

**Sequence construction**: `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`.
Every answer option gets its own `[MASK]` marker token before its text.

**Model**: pretrained bidirectional encoder (ModernBERT-large, 421M params) → hidden
states at each `[MASK]` position gathered → small 2-layer Transformer head → linear
scorer → softmax over the options in one pass. Three checkpoints (English 421M,
multilingual 322M, typed-decisions fine-tune), a lightweight `Router` picks between
them per request based on detected language.

**Answer types**: `choice` (softmax over K), `score` (softmax over ordinal levels, then
expected value), `noul` (forced binary [false,true], report P(true)).

**Calibration**: post-hoc temperature scaling, fit separately per question type AND per
option-count bucket (2 vs 3-5 vs 6-10 vs 11+ options need different sharpening).

**Training objective (RLCD)**: reward = log-score + spherical score, minus a ranked
probability score penalty for ordinal (`score`) questions. All three terms are strictly
proper scoring rules — reward is maximized only when reported probability equals true
probability (verified by their own property test in `tests/test_training.py`). This is
the core trick that avoids RLHF's "optimize for what raters like" and RLVR's "optimize
for what's verifiable" — instead it optimizes for honesty.

**Multi-turn**: TD(lambda) bootstrapped targets — trained on every prefix of a
conversation, early-turn targets built by blending real outcome with the model's own
(detached) later-turn prediction.

**Secondary head**: `act_head`, a tiny MLP on pooled `[CLS]` + 4 summary stats of the
answer distribution (top prob, margin, entropy, option count) → predicts a cost-
sensitive action (e.g. auto-act vs escalate), trained against a configured cost matrix.
Negligible extra compute — this is the template for our SLA/abstention feature flag.

**Why this matters for us**: the attention weights between each option's `[MASK]` token
and the state tokens are already computed in the single forward pass and then discarded.
Surfacing them cheaply gives lightweight explainability without extra passes — the
template for our `explain` feature flag.

## What we're deliberately doing differently

- Own sequence encoding scheme (not the mask-marker trick specifically)
- Own head/gating architecture (MoE across domain experts, not one shared head)
- Own reward formulation (may keep "strictly proper" as a property, not the exact
  log+spherical+RPS weighting)
- Not defaulting to ModernBERT — encoder choice is an open Phase 0 decision
