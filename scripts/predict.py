"""Runs a single typed-decision question through a trained checkpoint (from
train_moe_eval.py's --checkpoint-path) and prints the predicted option probabilities
plus which expert(s) the gate routed to. CPU-friendly -- no GPU needed for inference,
the checkpoint already holds the fine-tuned weights.

Example (score question, customer-support style):
    python scripts/predict.py --checkpoint-path checkpoints/moe_model.pt \\
        --instructions "How urgent is this support ticket?" \\
        --question-type score --criteria '["low", "medium", "high", "critical"]' \\
        --state "The customer said this was already escalated once and breached an SLA."

Example (choice question):
    python scripts/predict.py --checkpoint-path checkpoints/moe_model.pt \\
        --instructions "What category best fits this ticket?" \\
        --question-type choice \\
        --criteria '{"billing": "payment or invoice issue", "bug": "product not working as expected"}' \\
        --state "I was charged twice for my subscription this month."

Example (noul question -- criteria is always ["no", "yes"], no --criteria needed):
    python scripts/predict.py --checkpoint-path checkpoints/moe_model.pt \\
        --instructions "Does this message contain a threat of violence?" \\
        --question-type noul --state "I'm going to make them regret this."
"""
from __future__ import annotations

import argparse
import json

import torch

from decision_engine.model.encoder import load_tokenizer
from decision_engine.model.moe_model import MoEModel
from decision_engine.model.rendering import render_option_texts
from decision_engine.training.dataset import Collator
from decision_engine.training.device_utils import move_batch_to_device, resolve_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--state", required=True, help="the situation text, or a JSON object for structured state")
    p.add_argument("--instructions", required=True, help="the question being asked")
    p.add_argument("--question-type", required=True, choices=["choice", "score", "noul"])
    p.add_argument("--criteria", default=None,
                   help="JSON dict of {label: description} for choice, JSON list of ordered levels for score; "
                        "omit for noul (always [\"no\", \"yes\"])")
    p.add_argument("--device", default=None, help="defaults to CUDA if available, else CPU")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    domain_names = checkpoint["domain_names"]
    encoder_name = checkpoint["encoder_name"]

    print(f"loading {encoder_name} MoE model ({len(domain_names)} experts) on {device}...")
    tokenizer = load_tokenizer(encoder_name)
    model = MoEModel(domain_names, encoder_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    state = json.loads(args.state) if args.state.lstrip().startswith("{") else args.state
    criteria = json.loads(args.criteria) if args.criteria is not None else None
    option_texts = render_option_texts(args.question_type, criteria)

    example = {
        "state": state,
        "instructions": args.instructions,
        "question_type": args.question_type,
        "criteria": criteria,
        "target_distribution": [0.0] * len(option_texts),  # unused at inference, Collator just needs the shape
        "domain": "n/a",  # unused at inference, Collator only carries it through as metadata
    }
    collate = Collator(tokenizer)
    batch = move_batch_to_device(collate([example]), device)

    with torch.no_grad():
        out = model(batch)

    probs = out["blended_probs"][0, :len(option_texts)].tolist()
    routed_experts = [domain_names[i] for i in out["topk_idx"][0].tolist()]
    routing_weights = out["blend_weights"][0, out["topk_idx"][0]].tolist()

    print(f"\nrouted to: " + ", ".join(f"{d} ({w:.2f})" for d, w in zip(routed_experts, routing_weights)))
    print("\npredicted distribution:")
    for text, p in sorted(zip(option_texts, probs), key=lambda x: -x[1]):
        print(f"  {p:.4f}  {text}")


if __name__ == "__main__":
    main()
