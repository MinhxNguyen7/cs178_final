import unittest

from models import LittleModel
from testing.utils import single_dataloaders
from training import train

class TestLittleModel(unittest.TestCase):
    def test_single_sample(self):
        """
        Tests the model with a single training datapoint and a single (different) testing datapoint.
        
        This test is meant to ensure that the model is able to overfit a single datapoint.
        """
        model, loss = LittleModel.create_default()
        
        losses = train(model, *single_dataloaders(), loss, epochs=100, verbose=False)
        
        # Ensure that the model overfits the training datapoint
        self.assertLess(losses[0], 0.001)
        self.assertGreater(losses[-1], losses[0] * 100) # training loss should be 100x smaller than testing loss

if __name__ == '__main__':
    unittest.main()