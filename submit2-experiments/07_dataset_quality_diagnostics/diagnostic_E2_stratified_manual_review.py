"""
Diagnostic Tool E2: Complete manual review of 200 samples. Diagnostic Tool E only samples
the 10 highest-scoring mismatched samples (high difficulty, worst case, biased),
and only verifies the reasonableness of the labels of the most difficult-to-confuse samples,
without covering the remaining mismatched samples as well as aligned and weakly aligned samples
"""

import pandas as pd

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]

# Read the 10 most disorganized items in item_ids
ALREADY_REVIEWED = {
    "Q_062", "Q_037", "Q_050", "Q_057", "Q_027",
    "Q_061", "Q_046", "Q_003", "Q_007", "Q_049",
}


def main():
    print("*" * 70)
    print("  Diagnostic E2: Full-coverage manual review (all items)")
    print("*" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items. Reviewing all {len(df)} -- no sampling")

    df["_already_reviewed_in_E"] = df["item_id"].isin(ALREADY_REVIEWED)
    df["_label_order"] = df["gold_label"].apply(lambda l: LABELS.index(l) if l in LABELS else len(LABELS))
    df = df.sort_values(["_label_order", "item_id"]).reset_index(drop=True)

    cols = [c for c in ["item_id", "gold_label", "target_role", "item_text", "_already_reviewed_in_E"] if c in df.columns]
    df[cols].to_csv("diagnostic_E2_full_review.csv", index=False)

    for label in LABELS:
        n = (df["gold_label"] == label).sum()
        print(f"  {label}: {n} items")

    print("\n" + "*" * 70)
    print(f"  All {len(df)} items (grouped by class)")
    print("*" * 70)
    for label in LABELS:
        print(f"\n    {label}    ")
        for _, row in df[df["gold_label"] == label].iterrows():
            flag = "  [already read in Diagnostic E]" if row["_already_reviewed_in_E"] else ""
            print(f"\n  item_id={row['item_id']}  target_role={row.get('target_role', 'n/a')}{flag}")
            print(f"  text: {row['item_text']}")

    print("\n OK, Saved diagnostic_E2_full_review.csv (all items).")

if __name__ == "__main__":
    main()
