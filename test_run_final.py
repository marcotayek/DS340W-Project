"""Unittest smoke tests for the clean-machine reproducibility command."""
import tempfile
import unittest
from pathlib import Path
from run_final import DATA, EXPECTED, run, solve

class ReproductionTests(unittest.TestCase):
    def test_linear_solver(self):
        x = solve([[3, 1], [1, 2]], [9, 8])
        self.assertEqual([round(v, 8) for v in x], [2.0, 3.0])

    def test_dataset_and_paper_metrics(self):
        self.assertTrue(DATA.is_file(), f"Missing checked-in dataset: {DATA}")
        with tempfile.TemporaryDirectory() as directory:
            result = run(data=DATA, output=Path(directory) / "result.json")
            self.assertEqual(result["rows"], EXPECTED["rows"])
            self.assertEqual(result["players"], EXPECTED["players"])
            self.assertEqual(result["train"], EXPECTED["train"])
            self.assertEqual(result["test"], EXPECTED["test"])

if __name__ == "__main__":
    unittest.main()
