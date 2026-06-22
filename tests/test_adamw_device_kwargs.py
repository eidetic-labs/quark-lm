"""Device-aware AdamW kernel selection (fused on CUDA only).

Feature-completeness across GPU/NPU/CPU: the fused AdamW kernel is CUDA-only and
raises on CPU/MPS, so it must be selected only on CUDA. Every other device (CPU,
MPS, an NPU backend) falls back to torch's default path -- which the parity tests
validate -- so the default behavior is unchanged. Testable without a GPU.
"""

from __future__ import annotations

import unittest

import support  # noqa: F401  (puts src/ on sys.path)

from transformer_no_decay_mask import adamw_device_kwargs


class AdamwDeviceKwargsTest(unittest.TestCase):
    def test_cuda_selects_fused(self) -> None:
        self.assertEqual(adamw_device_kwargs("cuda"), {"fused": True})

    def test_non_cuda_devices_use_default_path(self) -> None:
        for device in ("cpu", "mps", "npu"):
            self.assertEqual(adamw_device_kwargs(device), {}, device)


if __name__ == "__main__":
    unittest.main()
