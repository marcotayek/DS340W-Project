"""Fast feature-engineering checks; no Wyscout download or PyTorch required."""

import unittest

import pandas as pd

from novel_context_experiment import (
    build_event_history,
    normalized_features,
)


def sample_events():
    return pd.DataFrame(
        {"match_id": mid, "h": i % 2, "e": i % 8,
         "x": i * 5, "y": i * 3, "t": i % 3, "a": 0}
        for mid in (1001, 2002) for i in range(10)
    )


class NovelContextFeatureTests(unittest.TestCase):
    def setUp(self):
        self.history = build_event_history(sample_events(), max_history=5)

    def test_no_context_crosses_matches(self):
        self.assertEqual(len(self.history), 10)
        self.assertEqual(self.history.groupby("match_id").size().to_dict(),
                         {1001: 5, 2002: 5})
        first_second_match = self.history[self.history.match_id == 2002].iloc[0]
        self.assertEqual(first_second_match["lag1_x"], 20)
        self.assertEqual(first_second_match["lag5_x"], 0)

    def test_all_variants_use_exactly_same_rows(self):
        for n, transitions, expected_features in (
            (1, False, 6), (3, False, 18),
            (5, False, 30), (5, True, 42)
        ):
            features, column_names = normalized_features(self.history, n, transitions)
            self.assertEqual(features.shape, (10, expected_features))
            self.assertEqual(len(column_names), expected_features)

    def test_transition_uses_only_history(self):
        features, columns = normalized_features(self.history, 5, True)
        row = features[0]
        self.assertAlmostEqual(row[columns.index("change1_x")], 0.05)
        self.assertAlmostEqual(row[columns.index("change1_y")], 0.03)
        self.assertEqual(row[columns.index("change1_team")], 1.0)
        # Changing the CURRENT target-event attributes must not change inputs.
        changed = self.history.copy()
        changed.loc[changed.index[0], ["x", "y", "h", "e"]] = [99, 99, 0, 7]
        altered_features, _ = normalized_features(changed, 5, True)
        self.assertTrue((features[0] == altered_features[0]).all())


if __name__ == "__main__":
    unittest.main()
