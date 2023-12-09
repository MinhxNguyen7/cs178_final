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
        
        losses = train(model, *single_dataloaders(), loss, epochs=3, verbose=False)
        
        training_loss = losses['train'][-1]
        testing_loss = losses['val'][-1]
        
        # Ensure that the model overfits the training datapoint
        self.assertLess(training_loss, 0.001)
        self.assertGreater(testing_loss, training_loss * 100) # training loss should be 100x smaller than testing loss

if __name__ == '__main__':
    unittest.main()