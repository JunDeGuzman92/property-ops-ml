import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_ops_ml.pipelines import (
    maintenance_priority_pipeline,
    market_rent_review_pipeline,
    renewal_review_pipeline,
)


def main():
    sample_dir = ROOT / "examples" / "sample_data"

    work_orders = pd.read_csv(sample_dir / "work_orders.csv")
    renewals = pd.read_csv(sample_dir / "renewals.csv")
    market = pd.read_csv(sample_dir / "market_rent.csv")

    maintenance = maintenance_priority_pipeline(
        work_orders,
        schema_mapping={
            "ticket_id": "work_order_id",
            "opened": "created_date",
            "closed": "closed_date",
            "issue": "category",
            "occupied": "occupied_unit",
            "repeat_count": "recurrence_count",
        },
    )
    renewal = renewal_review_pipeline(renewals)
    market_review = market_rent_review_pipeline(market)

    print("\nMaintenance priority sample")
    print(maintenance[["work_order_id", "maintenance_priority_score", "maintenance_review_band"]].to_string(index=False))

    print("\nRenewal review sample")
    print(renewal[["unit_id", "renewal_review_score", "renewal_review_band"]].to_string(index=False))

    print("\nMarket rent review sample")
    print(market_review[["unit_id", "market_review_score", "market_review_band"]].to_string(index=False))


if __name__ == "__main__":
    main()

