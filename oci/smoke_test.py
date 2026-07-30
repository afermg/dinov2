#!/usr/bin/env python3
"""End-to-end smoke inference against the DINOv2 OCI container."""

import json
import os

os.environ.setdefault("NAHUAL_IPC_TIMEOUT_MS", "900000")

import numpy as np
from nahual.process import dispatch_setup_process


def main() -> None:
    address = os.environ.get("NAHUAL_ADDRESS", "tcp://127.0.0.1:5555")
    setup, process = dispatch_setup_process("dinov2")

    info = setup(
        {
            "repo_or_dir": "facebookresearch/dinov2",
            "model_name": "dinov2_vits14",
            "pretrained": True,
        },
        address=address,
    )
    pixels = np.random.default_rng(42).random((1, 3, 1, 28, 28), dtype=np.float32)
    result = process(pixels, address=address)

    assert result.shape == (1, 384), result.shape
    assert np.isfinite(result).all()
    print(
        json.dumps(
            {"setup": info, "shape": list(result.shape), "dtype": str(result.dtype)}
        )
    )


if __name__ == "__main__":
    main()
