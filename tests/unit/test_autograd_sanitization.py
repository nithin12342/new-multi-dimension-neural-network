"""
Unit Test: Pillar 1 - Autograd Sanitization
Verifies output is native Python float and gradient history is cleanly stripped.
"""

import unittest
import torch
from src.engine.autograd import to_clean_scalar

class TestAutogradSanitization(unittest.TestCase):

    def test_autograd_detachment_and_float_type(self):
        # Create a tensor requiring gradients
        x = torch.tensor([42.5], requires_grad=True)
        y = x * 2.0 + 3.0
        
        self.assertTrue(y.requires_grad)
        
        # Apply to_clean_scalar
        scalar = to_clean_scalar(y)
        
        # Validation assertions
        self.assertIsInstance(scalar, float)
        self.assertAlmostEqual(scalar, 88.0, places=4)
        
        # Tensor without gradients
        z = torch.tensor([12.34], requires_grad=False)
        z_scalar = to_clean_scalar(z)
        self.assertIsInstance(z_scalar, float)
        self.assertAlmostEqual(z_scalar, 12.34, places=4)

if __name__ == "__main__":
    unittest.main()
