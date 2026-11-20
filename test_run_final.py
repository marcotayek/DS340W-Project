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

    def test_context_ablation_and_expected_scores(self):
        from ablation_experiment import run_ablation
        with tempfile.TemporaryDirectory() as directory:
            result = run_ablation(output=Path(directory) / "ablation.json", verbose=False)
            self.assertEqual(result["train_rows"], 51)
            self.assertEqual(result["test_rows"], 23)
            targets = {
                "player_only": (0.1545, 0.1985, 0.0631),
                "player_team": (0.1618, 0.2040, 0.0109),
                "full_context": (0.1574, 0.2004, 0.0453),
            }
            for name, (mae, rmse, r2) in targets.items():
                values = result["ridge_variants"][name]
                self.assertAlmostEqual(values["mae"], mae, delta=0.00005)
                self.assertAlmostEqual(values["rmse"], rmse, delta=0.00005)
                self.assertAlmostEqual(values["r2"], r2, delta=0.00005)
            self.assertLess(result["ridge_variants"]["player_only"]["rmse"],
                            result["ridge_variants"]["full_context"]["rmse"])

if __name__ == "__main__":
    unittest.main()
