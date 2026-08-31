"""
    The literal “Explain:” label is formally removed from the item_text of the frozen dataset,
    and a new frozen version (v2) is generated
"""

import re
import pandas as pd

SRC = "dataset_items_final.csv"
OUT_CSV = "dataset_items_final_v2.csv"
DIFF_CSV = "dataset_items_final_v2_diff.csv"


def strip_explain_label(text):
    t = text.replace("Explain:", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    df = pd.read_csv(SRC)
    df.columns = [c.strip() for c in df.columns]
    assert len(df) == 200, f"expected 200 rows, got {len(df)}"

    old_texts = df["item_text"].tolist()
    new_texts = [strip_explain_label(t) for t in old_texts]

    df_new = df.copy()
    df_new["item_text"] = new_texts

    n_changed = sum(1 for a, b in zip(old_texts, new_texts) if a != b)
    n_had_explain = sum(1 for t in old_texts if "Explain:" in t)

    print(f"OK, Loaded {len(df)} rows from {SRC}")
    print(f"Rows where item_text changed: {n_changed} / {len(df)}")
    print(f"Rows that originally contained 'Explain:': {n_had_explain}")
    print(f"Rows still containing 'Explain:' after transform: "
          f"{sum(1 for t in new_texts if 'Explain:' in t)} (should be 0)")


    period_before = sum(t.count(".") for t in old_texts)
    period_after = sum(t.count(".") for t in new_texts)
    print(f"Total periods across dataset BEFORE: {period_before}")
    print(f"Total periods across dataset AFTER:  {period_after} "
          f"(should be unchanged , punctuation not touched)")

    df_new.to_csv(OUT_CSV, index=False)
    print(f"\n OK, Saved {OUT_CSV}")

    diff_df = pd.DataFrame({
        "item_id": df["item_id"],
        "gold_label": df["gold_label"],
        "changed": [a != b for a, b in zip(old_texts, new_texts)],
        "old_text": old_texts,
        "new_text": new_texts,
    })

    diff_df.to_csv(DIFF_CSV, index=False)
    print(f"OK, Saved {DIFF_CSV} (A complete before-and-after comparison is provided for manual spot checks)")

    print("\n    Sample changed rows    ")
    shown = 0
    for _, row in diff_df[diff_df["changed"]].iterrows():
        print(f"\n[{row['item_id']} | {row['gold_label']}]")
        print(f"  BEFORE: {row['old_text']}")
        print(f"  AFTER:  {row['new_text']}")
        shown += 1
        if shown >= 5:
            break


if __name__ == "__main__":
    main()
