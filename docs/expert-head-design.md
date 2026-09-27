# Expert head architecture

Resolves the last major open design question from PLAN.md: how an expert head scores
options, replacing Laya's mask-marker-in-one-sequence mechanism with our own.

## Lineage: adapted from poly-encoders, not invented from nothing

Grounded in [Poly-encoders](https://arxiv.org/abs/1905.01969) (Humeau, Shuster,
Lachaux, Weston — ICLR 2020): the established insight that a bi-encoder (encode
everything separately, compare with a dot product — fast but low-fidelity) and a
cross-encoder (jam context and candidate into one sequence, full self-attention —
high-fidelity but must re-encode per candidate) aren't the only two options. A
poly-encoder encodes context once, then uses a lightweight attention step to produce
candidate-sensitive scores cheaply. Laya's mask-marker approach is architecturally a
cross-encoder (one packed sequence, full self-attention over everything, re-run per
question). Ours takes the poly-encoder side of that tradeoff, adapted for typed
decisions rather than dialogue response selection.

**Deviation from vanilla poly-encoders**: the original paper compresses context into a
fixed number of learned "code" vectors before the final attention step. We skip that
compression and let each option query attend directly into the full state token
sequence. Reason: we need token-level attention weights for the `explain` feature flag
(which input spans drove a decision) — compressing into abstract codes first would
destroy exactly the signal that feature needs.

## Mechanism

1. **Encode the state once.** NeoBERT runs on the state text alone -> `H_state`
   (full token hidden states). One forward pass per request, reused across every
   question asked about that state in the same call — unlike Laya, which re-runs its
   encoder over [instructions + options + state] per question.

2. **Encode each option as a query vector.** Short encoder pass over "label:
   description" text (reusing the same NeoBERT weights for embedding consistency,
   revisit for a lighter shared tower only if Phase 2 benchmarks show it matters — it's
   cheap since options are a handful of tokens). The question's type (`choice` /
   `score` / `noul`) is folded into this query vector, not broadcast across every state
   token the way Laya's type embedding is.

3. **Per-expert cross-attention (the actual "expert").** 1-2 layers of multi-head
   cross-attention per domain expert: option query attends over `H_state` as keys/
   values. Depth matched to Laya's 2-layer head for a fair latency comparison. Output:
   a state-conditioned representation per option, plus attention weights kept as the
   `explain` feature's evidence.

4. **Cheap option-to-option comparison.** One self-attention layer across the small set
   of option representations for a single question (a handful of options, not the
   whole state — negligible cost). Recovers the comparative signal Laya gets for free
   from packing options into one sequence, without inheriting Laya's actual cost of
   doing that: their `head_max_len` budget forces every option's text to shrink evenly
   when there are too many/long options (visible directly in their `build_sequence`
   code). Our options never compete with the state for context-window space.

5. **Score, softmax, blend.** Small MLP -> scalar logit per option -> softmax within
   the question -> a probability distribution, per expert. The gate (see
   `architecture-decisions.md`) blends the top-2 experts' distributions in probability
   space. Mixing two proper-scoring-rule-compatible distributions remains compatible
   with the reward's strict properness regardless of how the mixture was formed — this
   needs restating explicitly in the training doc when we design the reward, not just
   assumed here.

## Why this beats the mask-marker mechanism, concretely

- No token-budget fight between options and state — the actual problem visible in
  Laya's own code (`opt_budget < 16` triggers shrinking every option's text evenly).
- No sequence-packing ceiling on option count.
- Cheaper for the realistic usage pattern: one state, multiple typed questions per
  call (every published Jev/Laya example does this) — state is encoded once, not once
  per question.
- Attention weights fall out as a built-in explainability signal for the `explain`
  flag, rather than needing to be reconstructed post-hoc from a packed sequence.

## Open implementation questions (Phase 2)

- Full NeoBERT pass for option encoding vs. a lighter shared sub-network — benchmark
  before optimizing prematurely.
- Exact cross-attention head count / dimension — start matched to NeoBERT's own
  attention config, tune from there.
- Whether the option-comparison self-attention layer should be shared across experts
  (one comparison step after blending) or per-expert (before blending) — affects
  whether comparison happens pre- or post-MoE blend; needs an ablation, not a guess.
