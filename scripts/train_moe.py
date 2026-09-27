"""Phase 3 smoke test: proves the MoE mechanism (gate + 6 domain experts + MoE-aware
reward) runs end-to-end on real data. Deliberately tiny -- see moe_loop.py's docstring.
"""
from pathlib import Path

from decision_engine.training.moe_loop import run_moe_smoke_test

if __name__ == "__main__":
    data_dir = Path(__file__).parents[1] / "data"
    losses = run_moe_smoke_test(data_dir, n_steps=10, batch_size=2, max_examples=100, max_state_len=192)
    print()
    print(f"first loss: {losses[0]:.4f}")
    print(f"last loss:  {losses[-1]:.4f}")
    print("all finite:", all(l == l and abs(l) != float("inf") for l in losses))
