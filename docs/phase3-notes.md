# Phase 3 implementation notes

Records concrete decisions made building the MoE mechanism, alongside
`docs/architecture-decisions.md` (the design) and `docs/phase2-notes.md` (the same
kind of notes for the single-head baseline this extends).

## Each expert is unchanged from Phase 2

`model/moe_model.py`'s experts are literally `SingleHead` instances (one per domain),
not a new per-expert mechanism. The MoE moat is entirely in having several of them plus
a learned gate blending their outputs -- exactly the differentiation argument from the
original moat discussion: the architecture underneath needed to be genuinely different
from Laya/Jev, and the router is where that difference lives, not a reinvented head.

## Dense compute, not sparse dispatch -- a known, flagged gap

`MoEModel.forward` runs every expert on every example, then blends by gate weight
(zero for non-selected experts). This is correct for proving the mechanism works
(Phase 3's actual goal) but is not how this should run in production: the entire
"MoE routing is cheap" claim in `architecture-decisions.md` depends on only running
the *selected* top-k experts per example, not all of them. Sparse dispatch (grouping
examples by which experts they route to, forwarding each group through only its
experts) is real, non-trivial batching work for Phase 6, deliberately not attempted
here -- same scoping discipline as Phase 2's "encode the state once" serving
optimization.

## The per-expert auxiliary reward only touches *selected* experts

`compute_moe_loss` masks the auxiliary reward to experts actually in an example's
top-k (via `topk_idx`), verified directly by test: mutating a *non-selected* expert's
prediction to be confidently wrong leaves the loss unchanged, while mutating a
*selected* expert's prediction increases it. This matters because gradient signal
leaking to unselected experts would blur domain specialization -- the entire reason
the MoE architecture exists -- by pulling every expert toward fitting every example
regardless of routing.

## Load balancing and expert starvation are two different fixes for two different failures

`gate.py`'s `load_balancing_loss` (Switch-Transformer-style) keeps the *gate* from
collapsing onto a few experts. The per-expert auxiliary reward in `compute_moe_loss`
keeps *unselected* experts from starving even if the gate's routing distribution looks
reasonably balanced. Both are present and independently controllable
(`lambda_balance`, `lambda_expert`) because they address different failure modes, per
`reward-design.md`'s original design.

## Routing diagnostics exist, but a smoke test can't prove specialization

`moe_loop.py` prints each batch's first example's real domain next to which experts
the gate routed it to. With a randomly-initialized gate and only a handful of steps,
this should NOT show meaningful alignment yet -- that's the expected, correct behavior
of an untrained gate, not a bug. Whether domain-aligned specialization actually emerges
with real training is a Phase 3 *research* question this smoke test is not positioned
to answer; the diagnostic exists so a real training run can be watched for it later.

## A real bug this diagnostic caught: dataset truncation wasn't sampling across domains

The first smoke-test run showed `ex0_actual_domain='customer-support'` on *every* step,
never anything else. Not a coincidence: `TypedDecisionDataset` globbed
`data/<domain>/*.jsonl` in sorted path order and truncated at `max_examples` by simple
prefix -- and domain folder names sort alphabetically, so `max_examples=100` silently
returned only "customer-support" examples. The routing diagnostic never actually saw a
second domain. Fixed by round-robining across files before truncating (verified by
`tests/unit/test_typed_decision_dataset.py`). This is exactly the kind of thing a
routing diagnostic is *for*: it wasn't the gate or the model that was broken, but a
smoke test's own data sampling silently not testing what it claimed to.

The corrected re-run genuinely exercised all six domains (security-ops,
email-communication, ecommerce, customer-support, trust-safety, sales-crm all
appeared). Loss was noisier than Phase 2's single-domain run (0.1488 -> -0.7859 across
10 steps, non-monotonic) but net reward still improved (-0.106 -> 0.749 mean reward) --
expected, since each tiny batch now mixes different domains, question types, and
option counts, a harder optimization landscape than a single homogeneous domain.
Routing did not track domain identity consistently across steps (e.g. security-ops
examples routed to a different expert pair nearly every time it appeared) -- exactly
the "no meaningful alignment yet" behavior predicted above for an untrained gate after
10 steps, not a discouraging result.

## Real held-out evaluation, not just training-batch loss

Loss/reward numbers above prove the pipeline runs; they don't say whether the model is
learning anything useful, since they're measured on the same batch being trained on.
Added `training/metrics.py` (accuracy, per-question-type breakdown, expected
calibration error -- standard formula, implemented independently, not Laya's `ece_score`
we read in Phase 0 even though the two necessarily agree on any given input) and
`training/dataset.py`'s `train_val_split`, then ran `scripts/train_moe_eval.py`: 240
train / 60 held-out val examples, evaluate before training, 20 steps, evaluate after.

Results, measured on data the model never trained on:

| Metric | Before | After |
|---|---|---|
| Overall accuracy | 0.350 | 0.450 |
| ECE | 0.133 | 0.104 |
| `noul` accuracy (n=33) | 0.364 | 0.485 |
| `score` accuracy (n=20) | 0.200 | 0.500 |
| `choice` accuracy (n=7) | 0.714 | 0.143 |

Both headline numbers moved the right direction on held-out data -- real evidence of
generalization, not memorization of the training batch. Two things not to oversell:
the `choice` figure is 7 examples swinging from 5/7 to 1/7 correct over 20 noisy steps
-- noise, not a finding, given the sample size. And training loss itself was
non-monotonic (roughly -0.72 to +0.29 across the 20 steps), expected from tiny
batches (size 2) mixing different domains/question-types/option-counts with no
learning-rate schedule or warmup -- fine for a CPU correctness check, would need
addressing in an actual training run.

A real evaluation bug was caught building this: `score_mean_absolute_error` initially
divided the expected-value calculation by the option count a second time, after the
probabilities were already a proper distribution summing to 1 -- double-normalizing
and silently shrinking every MAE by roughly the option count. Caught by a test
asserting MAE scales with actual level-distance (1 vs 3 levels off should differ,
not both collapse toward the same tiny number), before it was ever used to judge a
real training run.

## Verified live before the smoke test ran

A hand-built 3-example batch (one from each of three different domains) through the
real MoEModel produced correct shapes (`blended_probs` [3,4], `expert_probs` [3,6,4],
`gate_logits` [3,6], `topk_idx` [3,2]), a finite loss, and finite gradients flowing to
both the gate's projection weights and an expert's scorer weights -- on the first run,
no debugging needed. Same validate-before-committing-to-a-slow-run discipline as
Phase 2.
