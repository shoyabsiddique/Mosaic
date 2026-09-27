"""Shared, tensor-friendly constants for the model package."""
QTYPE_TO_IDX = {"choice": 0, "score": 1, "noul": 2}
IDX_TO_QTYPE = {v: k for k, v in QTYPE_TO_IDX.items()}
