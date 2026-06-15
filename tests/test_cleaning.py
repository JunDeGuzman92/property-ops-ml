import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_ops_ml.cleaning import normalize_column_names, parse_money_series, parse_percent_series


class CleaningTests(unittest.TestCase):
    def test_normalize_column_names(self):
        df = pd.DataFrame({"Current Rent ($)": [1], "Lease End Date": [2]})
        self.assertEqual(list(normalize_column_names(df).columns), ["current_rent", "lease_end_date"])

    def test_parse_money_series(self):
        values = parse_money_series(pd.Series(["$1,250.50", "(300)"]))
        self.assertAlmostEqual(values.iloc[0], 1250.50)
        self.assertAlmostEqual(values.iloc[1], -300.0)

    def test_parse_percent_series(self):
        values = parse_percent_series(pd.Series(["95%", "0.82"]))
        self.assertAlmostEqual(values.iloc[0], 95.0)
        self.assertAlmostEqual(values.iloc[1], 82.0)


if __name__ == "__main__":
    unittest.main()

