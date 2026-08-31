
import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T

def classify_score(score: float, policy: dict) -> str:

    if score >= policy["aligned_threshold"]:
        return "Aligned"
    elif score >= policy["weakly_aligned_threshold"]:
        return "Weakly Aligned"
    else:
        return policy["mismatched_label"]


def main():
    print("*" * 50)
    print("  Edge Node Simulation v2 - Local Semantic Matching")
    print("  (using distribution_v2.pkl: specialised + general foundations)")
    print("*" * 50)

    artifact_path = "/kaggle/working/distribution_v2.pkl"
    dataset_path = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"

    #Load the cloud-generated artifact
    print(f"\n Loading cloud-generated artifact: {artifact_path}")
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)

    role_id = artifact["role_id"]
    capability_vectors = np.asarray(artifact["capability_vectors"])
    decision_policy = artifact["decision_policy"]
    model_id = artifact["metadata"]["model_identifier"]
    print(f"OK, Artifact loaded. Role: {role_id}, capability blocks: {capability_vectors.shape[0]}")

    # Load Model
    print(f"\n Loading edge-side semantic model: {model_id}")
    model_path = '/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_path, local_files_only=True)
    print("OK, Edge inference model loaded.")

    # Load dataset
    print(f"\n Loading evaluation dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} evaluation items")

    # Encoding
    print("\n Encoding evaluation items locally on the edge")
    item_vectors = model.encode(df["item_text"].tolist(), show_progress_bar=True)

    print("\n Computing cosine similarity against role capability vectors (max-similarity strategy)")
    sim_matrix = cosine_similarity_matrix(np.asarray(item_vectors), capability_vectors)
    max_scores = sim_matrix.max(axis=1)
    best_block_idx = sim_matrix.argmax(axis=1)

    df["similarity_score"] = max_scores
    df["best_capability_block"] = best_block_idx
    df["predicted_label"] = [classify_score(s, decision_policy) for s in max_scores]

    # Compare the prediction results with the gold_label of the engineering annotation.
    print("\n Comparing predictions against gold_label")
    y_true = df["gold_label"]
    y_pred = df["predicted_label"]

    labels_order = ["Aligned", "Weakly Aligned", "Mismatched"]
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    print(f"\n Overall Accuracy: {acc:.4f}\n")
    print("Per-class metrics:")
    for lbl, p, r, f, s in zip(labels_order, precision, recall, f1, support):
        print(f"  {lbl:15s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  n={s}")

    print(f"\n Confusion Matrix (rows=true, cols=pred), label order = {labels_order}")
    print(cm)

    out_path = "edge_inference_results_v2.csv"
    df.to_csv(out_path, index=False)
    print(f"\nOK, Full per-item results saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
