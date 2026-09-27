# Architecture decisions: encoder + gating

Resolves two of the open questions from PLAN.md. Decisions below, with reasoning and
what would make us revisit them.

## Encoder: NeoBERT (250M), English-only for v1

**Choice**: [NeoBERT](https://huggingface.co/chandar-lab/NeoBERT) — MIT licensed,
250M params, native 4,096-token context, pretrained from scratch on RefinedWeb.
Confirmed via its Hugging Face model card and GitHub repo (chandar-lab/NeoBERT), not
recalled from memory. Its own paper (arXiv:2502.19587) reports it beating ModernBERT
under identical fine-tuning conditions on MTEB.

**Why not ModernBERT (what Laya uses)**: different pretraining data (RefinedWeb),
different lab, independently trained — not a relabeled fork, and there's published
evidence it's competitive rather than us picking something worse just to be different.

**Why not a multilingual encoder now**: every Tier 1 dataset the research pass verified
is English-first. Committing to a multilingual encoder (or reusing mmBERT, which is
literally Laya's multilingual checkpoint) is a Tier 3 decision, not a v1 one — revisit
with EuroBERT or an equivalent modern multilingual encoder when non-English domains
actually enter scope.

**Real capability edge worth naming**: Laya's own English checkpoint config caps
`max_len` at 512 tokens (confirmed in `rl_agent_config.json`) even though the
underlying ModernBERT architecture could go longer — that's their product choice, not
an architecture limit. NeoBERT's native 4,096-token context means our English
checkpoint won't truncate long tickets/emails/reviews the way Laya's shipped checkpoint
does today. Worth stating as a real advantage, not overclaiming it as an architectural
breakthrough — it's a config/choice difference we get almost for free.

**Revisit if**: NeoBERT's real-world fine-tuning behavior (once we actually train
against it in Phase 2) underperforms a same-size ModernBERT baseline on our own data —
the paper's benchmarks are on MTEB, not our domains, so this needs our own validation
before we treat it as settled.

## Gating: soft top-k (k=2) blending over shared-encoder pooled output

**Design**: one NeoBERT forward pass per request (shared across all experts — this is
the whole point of the MoE moat, not repeated per domain). A gating network — a small
linear/MLP layer on the encoder's pooled output — scores all domain experts and blends
the top-2 by softmax weight. Each expert is a lightweight head (comparable in cost to
Laya's 2-layer transformer head), so running 2 instead of 1 is a small, bounded cost
increase, not a multiplier on the expensive part of the forward pass.

**Why soft top-k over hard top-1**:
- Domain boundaries are exactly where a generalist-vs-specialist tradeoff breaks —
  an ambiguous input (a support ticket that's really a security complaint, a sales
  email that's also a churn signal) shouldn't be forced into one bucket by a
  single misroute.
- Blending gives a reusable signal — the gate's own confidence spread doubles as a
  "domain ambiguity" indicator, which is useful input to the SLA/abstention feature
  flag we already scoped (low top-1 gate confidence is itself a reason to escalate).
- The cost difference between k=1 and k=2 is small because the heads are cheap and the
  encoder is shared — this isn't classic large-scale MoE where routing to more experts
  meaningfully multiplies cost.

**Why gate on the shared encoder's output, not a separate pre-classifier**: the
encoder runs regardless (there's no compute to save by routing before it, since we're
not using per-domain encoders). Gating post-encoder lets the whole thing train
end-to-end — the gate learns to route based on what actually improves expert accuracy,
not a proxy task like Laya's language-ID router. Simpler pipeline, one training loop
instead of two.

**Known risk to design against**: expert collapse (the gate learns to favor 1-2 experts
and starves the rest of gradient signal), a well-documented MoE training pathology.
Mitigate with an auxiliary load-balancing loss that penalizes non-uniform routing
across a batch, and track per-expert utilization + per-domain calibration (ECE)
throughout training as a first-class metric, not just aggregate accuracy.

**Fast-path option**: expose `routing_mode: "soft" | "hard"` as a later API-level
choice — hard top-1 for latency-sensitive callers willing to trade robustness at
domain boundaries for speed, soft top-2 (the default) for everyone else. This is the
same feature-flag philosophy as `sla_threshold` and `explain`.

**Revisit if**: Phase 3 experiments show soft blending doesn't actually beat hard
routing on accuracy once trained (it's a hypothesis, not a guarantee) — the whole
point of Phase 3 is measuring this before committing further.

## What's still open

- Exact expert head architecture (how many layers, how options/questions get scored —
  we're designing our own mechanism here, not reusing Laya's mask-marker trick, and
  haven't specified the replacement yet)
- How new experts get added post-launch without destabilizing the trained gate
  (freeze-and-extend vs full retrain)
- Reward/calibration formulation for training (own version of "strictly proper," not
  Laya's exact log+spherical+RPS weights)
