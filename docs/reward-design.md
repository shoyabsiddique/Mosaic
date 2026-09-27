# Reward / training objective design

Resolves the last open design question from PLAN.md: our own version of "strictly
proper," not Laya's exact log+spherical+RPS weighting.

## Principle: match the scoring rule to the answer space's structure

Laya applies the same fixed combination (log score + 0.5x spherical score, minus 1.0x
Ranked Probability Score for ordinal `score` questions only) across every question
type, patching in ordinal structure as a subtracted penalty rather than treating it as
a first-class distinction. We split by structure instead:

- **`choice` / `noul` (unordered or binary)** — **Brier score** (quadratic scoring
  rule): `R = 1 - sum_i (q_i - y_i)^2`. Strictly proper (Brier, 1950; formalized in
  Gneiting & Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation,"
  JASA 2007). Bounded, unlike the log score, whose penalty for a confident-wrong
  prediction is unbounded as `p -> 0`. That boundedness matters specifically because
  we're optimizing a distribution the model hasn't yet learned to calibrate — unbounded
  penalties early in training produce spiky, destabilizing gradients.
- **`score` (ordinal)** — **Ranked Probability Score (RPS)** as the sole term, not a
  bolted-on penalty. RPS is the natural generalization of Brier to cumulative
  distributions over ordered categories: `sum_k (CDF_q(k) - CDF_target(k))^2`. It
  already encodes "one level off should hurt less than three levels off" by
  construction — the entire reason Laya needs an extra term for score questions in the
  first place. One coherent, strictly proper rule per question type, chosen for its
  mathematical structure, rather than one generic combination patched for the ordinal
  case.

Both are individually strictly proper. A positive linear combination of proper scoring
rules stays proper (standard result in the scoring-rules literature), but we don't even
need a combination here — each question type gets exactly one rule matched to it.

## Calibration trained in, not just fitted after the fact

Laya's own `rl_common.py` comment says its temperature is "fitted post-hoc in
evaluate.py" — calibration is a correction applied after training finishes, not part
of the training objective. Add a differentiable calibration regularizer during
training instead: **MMCE** (Maximum Mean Calibration Error — Kumar, Sarawagi & Jain,
ICML 2018, arXiv:1802.00389), a published, differentiable calibration penalty based on
kernel mean embeddings, applied alongside the scoring-rule reward as an auxiliary loss.
Keep post-hoc temperature scaling too, as a final polish — belt and suspenders, not
temperature scaling as the sole calibration mechanism.

**Needs empirical validation, not assumed as a win**: MMCE is real and published, but
whether it actually improves our calibration beyond what post-hoc temperature scaling
alone achieves is a Phase 2/3 experiment, not a guarantee.

## MoE-specific reward wiring

Two requirements that don't exist in Laya's single-head design, because Laya has no
gate:

1. **Primary reward on the blended (top-2) distribution** — the actual API output;
   this is what must be calibrated.
2. **Smaller-weight auxiliary reward on each individual expert's own pre-blend
   distribution.** Without this, an expert the gate rarely selects gets almost no
   gradient signal and stagnates — this is the same expert-starvation risk flagged in
   `architecture-decisions.md`'s gating section, and this is where it actually gets
   fixed in the objective, not just via the gate's load-balancing loss. The two
   mechanisms (load-balancing loss on routing weights, auxiliary reward on individual
   expert outputs) address different failure modes and should both be present:
   load-balancing keeps the *gate* from collapsing onto one expert; the auxiliary
   reward keeps *unselected* experts from starving even if the gate is reasonably
   balanced.

## Reframing "RL": single-shot questions don't need it

Worth being precise rather than inheriting Laya/Jev's "Reinforcement Learning for
Calibrated Decisions" framing wholesale. A strictly proper scoring rule is a
differentiable function of the reported distribution (a softmax output) and a fixed
target — for a single-shot typed question, this is just a differentiable loss you
backprop through directly. No environment, no sampling, no policy gradient required.

The one place genuine sequential RL-style reasoning is actually warranted is
multi-turn conversation handling (Laya's TD(lambda) bootstrapping across conversation
prefixes), where the target itself depends on a future turn's outcome — a real
sequential credit-assignment problem. Keep that distinction sharp in our own
implementation: direct differentiable scoring-rule optimization for single-shot
questions; TD-style bootstrapping reserved specifically for multi-turn episodes, where
it's mathematically earned rather than a branding choice.

## Summary formulation

For a single-shot question with reported distribution `q` (or per-expert `q_1, q_2`
blended into `q_blend` by the gate) and target `y`:

```
score_choice_or_noul(q, y) = Brier(q, y)                    # bounded, strictly proper
score_ordinal(q, y)        = -RPS(q, y)                     # strictly proper, ordinal-aware
reward(q, y, qtype)        = score_choice_or_noul or score_ordinal, by type

total_loss = -reward(q_blend, y, qtype)                     # primary, on API output
           + lambda_expert * sum_i [-reward(q_i, y, qtype)] # per-expert, i in top-2
           + lambda_calib  * MMCE(q_blend, y)                # in-training calibration
           + lambda_balance * load_balancing_loss(gate)      # from architecture-decisions.md
```

`lambda_expert`, `lambda_calib`, `lambda_balance` are hyperparameters to tune in
Phase 2/3 — not fixed here, since Laya's own weights (0.5, 1.0) were clearly tuned
empirically rather than derived, and ours should be too.

## Still open (moved from architecture to empirical)

- Exact values for `lambda_expert`, `lambda_calib`, `lambda_balance` — Phase 2/3 tuning.
- Whether MMCE actually beats post-hoc-only calibration on our data — needs the A/B,
  not assumed.
- TD(lambda) details for multi-turn episodes, if/when a domain needs conversation
  handling (none of the six v1 domains obviously require it yet — flag for Phase 3).
