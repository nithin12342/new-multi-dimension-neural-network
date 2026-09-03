"""
Unit Test: Pillar 4 - Pinned Memory Pool
Validates circular buffer acquisition, stability, and pinned memory allocation.
"""

import unittest
import torch

from src.infrastructure.memory.pinned_pool import PinnedTensorPool

class TestPinnedMemoryPool(unittest.TestCase):

    def test_circular_acquisition_and_stability(self):
        capacity = 4
        specs = {
            "image": ((8, 3, 224, 224), torch.float32),
            "text": ((8, 128), torch.int64),
        }

        pool = PinnedTensorPool(capacity=capacity, tensor_specs=specs)

        # Acquire 4 distinct image buffers
        acquired = [pool.acquire("image") for _ in range(capacity)]
        self.assertEqual(len(acquired), 4)

        # Ensure shapes match specs
        for tensor in acquired:
            self.assertEqual(tensor.shape, (8, 3, 224, 224))
            self.assertEqual(tensor.dtype, torch.float32)

        # 5th acquisition should cycle back to the 1st buffer (same data_ptr)
        cycle_1 = pool.acquire("image")
        self.assertEqual(cycle_1.data_ptr(), acquired[0].data_ptr())

        # Check pinned status if CUDA is available
        if torch.cuda.is_available():
            self.assertTrue(acquired[0].is_pinned())

    def test_invalid_key_handling(self):
        pool = PinnedTensorPool(capacity=2, tensor_specs={"image": ((4, 4), torch.float32)})
        with self.assertRaises(KeyError):
            pool.acquire("audio")

if __name__ == "__main__":
    unittest.main()
