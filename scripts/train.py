"""Phase 2 smoke test: proves the full pipeline (real NeoBERT encoder, real data,
real reward) runs end-to-end on this machine. Deliberately tiny (few steps, small
batch, truncated sequences) -- this is a correctness check, not a training run. Real
training needs rented GPU compute; see docs/local-serving.md.
"""
from pathlib import Path

from decision_engine.training.loop import run_smoke_test

if __name__ == "__main__":
    data_dir = Path(__file__).parents[1] / "data"
    losses = run_smoke_test(data_dir, n_steps=10, batch_size=2, max_examples=100, max_state_len=192)
    print()
    print(f"first loss: {losses[0]:.4f}")
    print(f"last loss:  {losses[-1]:.4f}")
    print("all finite:", all(l == l and abs(l) != float("inf") for l in losses))
