""" Test: load_data.py utility """

# Imports
import unittest
import pandas as pd

from src.utils import load_data

# Test utility
class TestLoadHeartDiseaseData(unittest.TestCase):
    """ Tests for load_data.py utility/function """

    def test_download_and_metadata(self):
        """ Actual test function. Checks for dataset size. """
        features, target = load_data.get_data_from_uci(False)

        # Check for successful download
        self.assertIsNotNone(features)
        self.assertIsNotNone(target)

        # Check data types
        # Both are returned as dataframes
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIsInstance(target, pd.DataFrame)

        # Check basic dimensions
        self.assertGreater(features.shape[0], 0)  # Ensure at least one row
        self.assertGreater(features.shape[1], 0)  # Ensure at least one feature
        self.assertEqual(target.shape[0], features.shape[0])  # Ensure target matches features

# entrypoint for test script
if __name__ == '__main__':
    unittest.main()

# End.
