"""

Based on the known cross-items reported in diagnostic_C3_weakly_aligned_cluster_inspection.py, recalculate
the accurate WA-D/WA-F cross-items with KMeans clustering:

- Cluster 0 (n=25) = All WA-D items except Q_115, Q_148, and Q_144 and Q_108

- Cluster 1 (n=25) = All WA-F items except Q_144, Q_108, and Q_115 and Q_148

This will reconstruct the complete cluster membership for each item, and then use sklearn to correctly calculate
purity, adjust the Rand index, and normalize the mutual information

"""
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

WA_D_IDS = {
    "Q_104", "Q_113", "Q_114", "Q_115", "Q_116", "Q_119", "Q_120", "Q_121", "Q_122",
    "Q_123", "Q_124", "Q_125", "Q_126", "Q_127", "Q_128", "Q_130", "Q_136", "Q_138",
    "Q_139", "Q_143", "Q_146", "Q_147", "Q_148", "Q_149", "Q_150",
}
WA_F_IDS = {
    "Q_101", "Q_102", "Q_103", "Q_105", "Q_106", "Q_107", "Q_108", "Q_109", "Q_110",
    "Q_111", "Q_112", "Q_117", "Q_118", "Q_129", "Q_131", "Q_132", "Q_133", "Q_134",
    "Q_135", "Q_137", "Q_140", "Q_141", "Q_142", "Q_144", "Q_145",
}

# Diagnostic_C3 report cross-items
MOVED_TO_CLUSTER1 = {"Q_115", "Q_148"}
MOVED_TO_CLUSTER0 = {"Q_144", "Q_108"}

assert MOVED_TO_CLUSTER1 <= WA_D_IDS
assert MOVED_TO_CLUSTER0 <= WA_F_IDS

cluster0 = (WA_D_IDS - MOVED_TO_CLUSTER1) | MOVED_TO_CLUSTER0
cluster1 = (WA_F_IDS - MOVED_TO_CLUSTER0) | MOVED_TO_CLUSTER1

print(f"Cluster 0 size = {len(cluster0)}  (expected 25)")
print(f"Cluster 1 size = {len(cluster1)}  (expected 25)")
assert len(cluster0) == 25 and len(cluster1) == 25
assert cluster0.isdisjoint(cluster1)
assert cluster0 | cluster1 == WA_D_IDS | WA_F_IDS

all_ids = sorted(WA_D_IDS | WA_F_IDS)
subtype_labels = np.array(["WA-D" if i in WA_D_IDS else "WA-F" for i in all_ids])
cluster_labels = np.array([0 if i in cluster0 else 1 for i in all_ids])

# Full 2x2 crosstab
n_d0 = int(np.sum((subtype_labels == "WA-D") & (cluster_labels == 0)))
n_d1 = int(np.sum((subtype_labels == "WA-D") & (cluster_labels == 1)))
n_f0 = int(np.sum((subtype_labels == "WA-F") & (cluster_labels == 0)))
n_f1 = int(np.sum((subtype_labels == "WA-F") & (cluster_labels == 1)))

print("\n     Full 2x2 crosstab (subtype x KMeans cluster)    ")
print(f"{'':<8}{'Cluster0':>10}{'Cluster1':>10}{'Total':>10}")
print(f"{'WA-D':<8}{n_d0:>10}{n_d1:>10}{n_d0 + n_d1:>10}")
print(f"{'WA-F':<8}{n_f0:>10}{n_f1:>10}{n_f0 + n_f1:>10}")
print(f"{'Total':<8}{n_d0 + n_f0:>10}{n_d1 + n_f1:>10}{n_d0 + n_d1 + n_f0 + n_f1:>10}")

correct = n_d0 + n_f1
misaligned = n_d1 + n_f0
purity = correct / len(all_ids)
print(f"\n Correctly subtype-aligned: {correct}/50 = {purity:.4f}")
print(f"Cross-over (misaligned):   {misaligned}/50  "
      f"({n_d1} WA-D->Cluster1, {n_f0} WA-F->Cluster0)")

ari = adjusted_rand_score(subtype_labels, cluster_labels)
nmi = normalized_mutual_info_score(subtype_labels, cluster_labels)
print(f"\n Adjusted Rand Index (subtype vs cluster):        {ari:.4f}")
print(f"Normalized Mutual Information (subtype vs cluster): {nmi:.4f}")

print("\n NOTE, The crossover items:")
print(f"  WA-D items assigned to the WA-F-majority cluster: {sorted(MOVED_TO_CLUSTER1)}")
print(f"  WA-F items assigned to the WA-D-majority cluster: {sorted(MOVED_TO_CLUSTER0)}")
print(f"  Total crossover count = {len(MOVED_TO_CLUSTER1) + len(MOVED_TO_CLUSTER0)} items, "
      f"not 2 -- the original report's '23/25, only Q_144/Q_108 crossed over' phrasing")
print("  only described ONE side of the symmetric 2x2 table and was misleading")
