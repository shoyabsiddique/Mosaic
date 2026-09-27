# Code structure

Proposed `src/` layout, each module tied to the design doc that specifies it. Nothing
here exists yet (Phase 2 onward) — this is the target structure to build against.

```
decision-engine/
  src/
    decision_engine/
      config/
        model.yaml              # encoder, head, gate hyperparameters
        training.yaml            # reward weights, optimizer, schedule
        domains.yaml              # per-domain expert registry (name, dataset paths)

      data/
        schema.py                 # unified {state, question, target} record type
        loaders/
          customer_support.py     # per-domain dataset loaders, one per Tier-1 domain
          trust_safety.py
          email_communication.py
          sales_crm.py
          security_ops.py
          ecommerce.py
        synth/
          sales_crm_generator.py         # synthetic conversational states (confirmed gap)
          ecommerce_return_generator.py  # synthetic return-reason labels (confirmed gap)
          security_ops_fp_generator.py   # synthetic SOC false-positive narratives
        preprocess.py              # tokenization, state/option sequence prep

      model/
        encoder.py                 # NeoBERT wrapper -- see architecture-decisions.md
        option_encoder.py          # short-text option + type-embedding query builder
        expert_head.py             # cross-attention + comparison self-attn + scorer
                                    # -- see expert-head-design.md
        gate.py                    # soft top-2 routing, load-balancing loss
                                    # -- see architecture-decisions.md
        moe_model.py                # ties encoder + gate + experts into one forward pass
        types.py                    # QTYPES (choice/score/noul), shared dataclasses

      training/
        reward.py                   # Brier (choice/noul), RPS (score), MMCE calibration
                                    # -- see reward-design.md
        loop.py                     # training loop, optimizer, scheduler
        calibration.py               # post-hoc temperature fitting (final polish)
        metrics.py                   # ECE, AUROC, per-domain accuracy, expert utilization

      serving/
        export_onnx.py               # ONNX export + numerical parity check
        runtime_onnx.py               # ONNX Runtime wrapper (CPU / DirectML)
                                      # -- see local-serving.md
        api/
          app.py                      # FastAPI app
          schemas.py                   # pydantic request/response models, incl. feature flags
          routes.py                    # /predict, handles explain / sla_threshold flags

      explain/
        attention_rationale.py         # formats cross-attention weights into evidence spans

  tests/
    unit/                              # per-module tests, pytest
    integration/                       # end-to-end request -> response tests

  scripts/
    train.py
    evaluate.py
    export.py
    benchmark_vs_competitors.py        # Phase 5: head-to-head vs Jev/Laya on shared datasets

  data/<domain>/                       # already exists -- raw + processed training data
  docs/                                # already exists -- this file lives here
  research/                            # already exists -- Jev/Laya notes, dataset audits
```

## Design principle behind the layout

`model/` mirrors the request's actual data flow (encoder -> option_encoder -> gate ->
expert_head -> moe_model ties them together), so reading the directory top to bottom
reads the same way `docs/expert-head-design.md` and `architecture-decisions.md`
describe the forward pass. `training/` and `serving/` are separated because they have
genuinely different lifecycles and dependencies (training needs the full data/reward
stack; serving should be runnable with only `runtime_onnx.py` and no PyTorch/training
code at all, matching the local-serving requirement).
