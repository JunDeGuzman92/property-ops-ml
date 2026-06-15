import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_ops_ml.features import build_maintenance_features, build_renewal_features
from property_ops_ml.pipelines import maintenance_priority_pipeline, market_rent_review_pipeline


class FeaturePipelineTests(unittest.TestCase):
    def test_maintenance_features_include_categories(self):
        df = pd.DataFrame(
            {
                "created_date": ["2026-01-01"],
                "category": ["Heat is not working"],
                "status": ["Open"],
                "occupied_unit": [1],
            }
        )
        features = build_maintenance_features(df, today="2026-01-05")
        self.assertIn("category_hvac", features.columns)
        self.assertEqual(int(features["category_hvac"].iloc[0]), 1)

    def test_renewal_features_compute_rent_gap(self):
        df = pd.DataFrame(
            {
                "lease_start_date": ["2025-01-01"],
                "lease_end_date": ["2026-01-01"],
                "current_rent": ["$1,000"],
                "proposed_rent": ["$1,050"],
                "market_rent": ["$1,100"],
            }
        )
        features = build_renewal_features(df)
        self.assertAlmostEqual(features["rent_increase_pct"].iloc[0], 5.0)
        self.assertAlmostEqual(features["market_gap_pct"].iloc[0], 10.0)

    def test_maintenance_pipeline_scores_rows(self):
        df = pd.DataFrame(
            {
                "opened": ["2026-01-01"],
                "issue": ["Smoke alarm issue"],
                "status": ["Open"],
                "occupied": [1],
            }
        )
        result = maintenance_priority_pipeline(
            df,
            schema_mapping={"opened": "created_date", "issue": "category", "occupied": "occupied_unit"},
        )
        self.assertIn("maintenance_priority_score", result.columns)
        self.assertGreater(result["maintenance_priority_score"].iloc[0], 0)

    def test_market_pipeline_scores_rows(self):
        df = pd.DataFrame(
            {
                "current_rent": ["$1,000"],
                "market_rent": ["$1,150"],
                "prior_market_rent": ["$1,100"],
                "comp_count": [2],
            }
        )
        result = market_rent_review_pipeline(df)
        self.assertIn("market_review_score", result.columns)
        self.assertGreater(result["market_review_score"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()

