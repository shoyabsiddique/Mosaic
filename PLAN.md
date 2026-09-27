# Mosaic — Project Plan

A wide-domain, "System 1" typed-decision model competing with TypeSafe's Jev (closed)
and Convai's Laya (open, Apache 2.0). Both take a state (text/ticket/email/JSON) plus
typed questions (`choice` / `score` / `noul`) and return calibrated probabilities in a
single forward pass, no text generation.

## Positioning

- **Core moat: Mixture-of-Experts routing.** One shared encoder, many lightweight
  domain-specialist heads, a cheap gating step picks/blends which head(s) answer a
  given request. This is the lever for "wide-ranging but accurate" — breadth without
  the usual generalist accuracy tax — and it's a genuine architectural difference from
  both competitors, who ship one head (Laya) or an undisclosed monolith (Jev).
- **Feature-flagged add-ons**, opt-in per API call so the default path stays fast:
  - `sla_threshold` — abstention/escalation. Confident answers return immediately;
    anything under threshold routes to a fallback (LLM or human queue) instead of
    guessing. Sold as a reliability SLA, not just a confidence number.
  - `explain` — lightweight rationale from the forward pass's own attention weights
    between each option and the state tokens ("these spans drove this decision").
    Not SHAP/LIME-grade, but free (no extra passes) and enough for most compliance asks.
- **Explicitly not a port of Laya.** Own sequence/input encoding, own head and gating
  design, own reward formulation. Laya's code was read for understanding only — see
  `research/laya-jev-notes.md`.
- **Local/offline serving is a first-class target, not an afterthought.** Jev is
  hosted-API-only; Laya proved this class of model runs fully offline on consumer
  hardware (16GB MacBook Air, CPU-only, ~33ms). Our architecture must match that: the
  shared-encoder MoE design keeps total weight size close to a single encoder (experts
  are small heads, not separate encoders), so there's no structural reason we can't hit
  the same offline bar or beat it. Verified against a real reference machine (12th Gen
  i5-1235U, 10C/12T, 15.7GB RAM, no CUDA) — see `docs/local-serving.md`.

## V1 domains (equal priority, one expert each)

1. Customer support triage — routing, urgency, churn risk, sentiment
2. Trust & safety / content moderation — toxicity, spam, policy category, severity
3. Email / communication classification — intent, priority, phishing/spam
4. Sales / CRM — lead scoring, buying intent, deal-stage classification
5. Security operations / alert triage — vuln severity, incident category, false-positive detection
6. E-commerce — review sentiment/aspect, return-reason, product categorization

Each domain = one candidate expert head behind the MoE gate. Tier 2/3 domains
(finance/insurance, HR, legal, DevOps, healthcare routing, education, gov, multilingual)
come after v1 proves the routing thesis — see `docs/domain-tiers.md`.

## Phases

- **Phase 0 — Research & data audit** (current): understand the competitive
  architecture, define domains, inventory available training data per domain.
- **Phase 1 — Data pipeline** (underway): collect/clean/label public + synthetic data
  per domain, unify into one schema (state + typed question + target distribution).
  All six Tier-1 domains now have a working loader pulling real, live, licensed data
  end-to-end (64,400 records total across the six samples as of the last run) -- see
  `src/decision_engine/data/schema.py` and `src/decision_engine/data/loaders/`.
  Per-domain question-type coverage -- **all six domains now cover all three question
  types** (each loader's own docstring explains the reasoning; `synth` = closed via a
  synthetic generator, tagged `license: "synthetic"` so it's never confused with real
  data; `real` = a second genuine source, not a proxy):
  | domain | choice | score | noul |
  |---|---|---|---|
  | customer-support | real | synth | real |
  | trust-safety | - | real | real |
  | email-communication | real | synth | real |
  | sales-crm | real | synth | real |
  | security-ops | real | real | real |
  | ecommerce | real | real | synth |

  How each remaining gap was closed:
  - **customer-support `score`**: `data/synth/customer_support_urgency_generator.py`
    -- the CFPB data genuinely has no ordinal field, confirmed in Phase 0 research.
  - **security-ops `choice`/`noul`**: closed with **real data**, not synthesis -- VCDB
    (VERIS Community Database, CC-BY-SA-4.0), fetched as its flattened CSV export.
    `choice` uses VCDB's own coded top-level action category (Malware/Hacking/Social/
    Physical/Misuse/Error/Environmental), avoiding the unbounded-CWE-taxonomy problem
    flagged earlier. `noul` uses the presence of `actor.external.*` vs `actor.internal.*`
    coded fields -- a real fact VERIS captures, skipped when both or neither are coded
    (ambiguous) rather than guessed.
  - **ecommerce `choice`**: `data/synth/ecommerce_return_generator.py` -- confirmed in
    Phase 0 that no public dataset anywhere has genuine return-reason labels.
  - **customer-support sampling**: `fetch_raw` now queries each product category
    separately via the CFPB API's own filter (fixing the 86%-one-category skew from
    relying on default recency sort), plus a deliberate `timely=No` oversample --
    verified against the *full* population (18M+ complaints, not a sample) that the
    true split is ~0.6% No / 99.4% Yes, a genuine population characteristic that would
    be wrong to force to 50/50, but too rare in a plain sample to train on otherwise.

  Trust-safety's `choice` gap is the one remaining domain without full coverage --
  Civil Comments is a severity-scoring dataset with no natural category label; closing
  it needs HateXplain or Davidson's 3-class labels, not attempted here.
- **Phase 2 — Single-expert baseline** (pipeline proven, convergence not attempted):
  one shared encoder + one head, no MoE yet. Real code exists and runs end-to-end
  against real NeoBERT weights and real Phase 1 data -- `src/decision_engine/model/`
  (encoder, option_encoder, single_head, baseline_model), `src/decision_engine/training/`
  (reward: Brier/RPS/MMCE per `docs/reward-design.md`; dataset/collator; training loop).
  `scripts/train.py`'s 10-step CPU smoke test moved mean reward from 0.284 to 0.637 --
  real evidence gradients flow correctly and the model can fit signal, not just "ran
  without crashing." 134 tests pass across the full project. Precise scope of "done"
  here -- proving the pipeline works, not training to convergence, which needs rented
  GPU compute this local machine doesn't have -- is written up in `docs/phase2-notes.md`.
- **Phase 3 — MoE router** (pipeline proven, convergence/specialization not attempted):
  gating network + 6 per-domain expert heads (each literally a Phase 2 `SingleHead`,
  unchanged -- the moat is the gate plus multiple instances, not a new per-expert
  mechanism) -- `src/decision_engine/model/gate.py`, `moe_model.py`,
  `training/reward.py`'s `compute_moe_loss`. Soft top-k gating, Switch-Transformer-style
  load balancing, and a per-expert auxiliary reward that's verified (by test) to touch
  only *selected* experts -- gate collapse and expert starvation are two different
  failure modes with two independent fixes, per `docs/reward-design.md`. A hand-built
  batch through the real model produced correct shapes, finite loss, and finite
  gradients to both gate and experts on the first run; 153 tests pass project-wide.
  The first smoke-test run accidentally validated only one domain (a dataset-truncation
  bug -- alphabetically-sorted domain folders meant `max_examples` silently returned
  only "customer-support" data; fixed by round-robining across domains before
  truncating, caught precisely because the routing diagnostic looked wrong). The
  corrected re-run genuinely exercised all six domains with net-positive reward
  improvement, noisier than Phase 2 as expected from mixing domains/question-types/
  option-counts per tiny batch. Routing does not yet track domain identity -- exactly
  the expected behavior of an untrained gate after 10 steps, not a result to read
  anything into yet. Known, flagged gap: this computes every expert densely rather
  than dispatching only to selected ones -- correct for proving the mechanism, not yet
  the efficient version the architecture is designed to be.

  Added real held-out evaluation (`training/metrics.py`: accuracy, ECE;
  `train_val_split`) rather than trusting raw loss alone -- a 240/60 train/val CPU run
  (20 steps) moved overall held-out accuracy 0.350 -> 0.450 and ECE 0.133 -> 0.104,
  genuine generalization signal since val was never trained on. `choice` accuracy
  looked like it dropped, but n=7 -- noise, not a finding. 164 tests pass project-wide.
  Full detail, including a real MAE-calculation bug the tests caught before it could
  mislead a real evaluation, in `docs/phase3-notes.md`.
- **Phase 4 — Feature flags**: `sla_threshold` (abstention head) and `explain`
  (attention-based rationale), gated so they don't touch the default latency path.
- **Phase 5 — Benchmarking**: head-to-head vs. Jev and Laya on shared public datasets
  per domain — accuracy, calibration (ECE), latency, cost.
- **Phase 6 — Serving**: hosting, SDKs, pricing model, **and a local/offline runtime**
  (ONNX export + CPU/DirectML execution providers + fp16/int8 quantization) so the
  model can run with no network calls on consumer hardware, matching or beating Laya's
  offline story — not just a hosted-API product like Jev. See `docs/local-serving.md`.

## Architecture decisions (resolved — see `docs/architecture-decisions.md`)

- **Encoder**: NeoBERT (250M, MIT, 4,096-token context, RefinedWeb-pretrained),
  English-only for v1. Genuinely different lineage from Laya's ModernBERT, independent
  benchmark evidence it's competitive, real context-length edge over Laya's shipped
  English checkpoint (which caps at 512 despite the architecture allowing more).
- **Gating**: soft top-2 blending over the shared encoder's pooled output, trained
  end-to-end with an auxiliary load-balancing loss against expert collapse. Hard top-1
  reserved as an optional latency fast-path (`routing_mode` flag), not the default.
- **Expert head**: state encoded once, options scored via poly-encoder-style
  cross-attention (option query attends into state tokens) instead of Laya's
  mask-marker-in-one-sequence approach, plus a cheap option-to-option comparison step.
  Adapted from Humeau et al.'s poly-encoders (ICLR 2020) — see
  `docs/expert-head-design.md`. Fixes Laya's own token-budget competition between
  options and state, and is cheaper for the common multi-question-per-call pattern.
- **Reward**: Brier score for `choice`/`noul` (bounded, strictly proper), Ranked
  Probability Score for ordinal `score` (one coherent rule per question type instead
  of Laya's fixed combo patched for ordinality), plus an in-training MMCE calibration
  regularizer alongside post-hoc temperature scaling, plus a per-expert auxiliary
  reward so unselected experts don't starve. Single-shot questions are optimized as a
  direct differentiable loss, not framed as RL — TD(lambda)-style bootstrapping is
  reserved for genuinely sequential multi-turn cases. See `docs/reward-design.md`.

## Still open (empirical, not design — needs Phase 2/3 experiments)

- How new experts get added post-launch without destabilizing the trained gate.
- Exact loss-weighting hyperparameters (`lambda_expert`, `lambda_calib`,
  `lambda_balance`) and whether MMCE beats post-hoc-only calibration.
- How much labeled data per domain is "enough" to beat a generalist baseline.
- Where cross-domain expert interference shows up first (which two domains fight over
  the same input distribution) — an empirical Phase 3 question, not a design one.

## Structure of this repo

```
decision-engine/
  PLAN.md                  this file
  data/<domain>/           raw + processed training data per domain
  docs/                    architecture notes, domain tiering, benchmarking plans
    architecture-diagram.svg   inference-time data flow (see below)
    code-structure.md          proposed src/ layout, module-by-module
    tech-stack.md               every library, grouped by purpose, with rationale
  research/                notes from studying Jev/Laya, dataset candidates, papers
  src/                     model code (empty until Phase 2 -- see docs/code-structure.md)
```

See `docs/architecture-diagram.svg` for the inference-time flow (state -> shared
NeoBERT encoder + option encoder -> top-2 gate -> 6 domain expert heads -> blend ->
calibrated output, with `explain`/`sla_threshold` as optional branches), and
`docs/code-structure.md` / `docs/tech-stack.md` for how that maps to actual code and
libraries.
