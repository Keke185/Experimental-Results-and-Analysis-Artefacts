import time
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

N_BLOCKS = 6
N_WARMUP = 1000
N_FORMAL = 10000
POLICY = {"aligned_threshold": 0.75, "weakly_aligned_threshold": 0.55, "mismatched_label": "Mismatched"}
OUT_PATH = "decision_logic_microbench.csv"


def old_logic(sim_vec, policy):
    score = float(sim_vec.max())
    if score >= policy["aligned_threshold"]:
        return "Aligned"
    elif score >= policy["weakly_aligned_threshold"]:
        return "Weakly Aligned"
    else:
        return policy["mismatched_label"]


def build_features(sim_vec):
    mean_sim = sim_vec.mean()
    std_sim = sim_vec.std()
    sorted_sim = np.sort(sim_vec)[::-1]
    margin = sorted_sim[0] - sorted_sim[1]
    return np.concatenate([sim_vec, [mean_sim, std_sim, margin]])

def fit_new_logic_pipeline(n_train=2000, seed=42):

    rng = np.random.default_rng(seed)
    X_raw = rng.uniform(0.1, 0.95, size=(n_train, N_BLOCKS))
    X = np.array([build_features(row) for row in X_raw])
    y = rng.integers(0, 3, size=n_train)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=200).fit(scaler.transform(X), y)
    return scaler, clf


def new_logic(sim_vec, scaler, clf):
    feat = build_features(sim_vec).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    return int(clf.predict(feat_scaled)[0])


def stats(arr):
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "sd": float(np.std(arr, ddof=1)),
        "p95": float(np.percentile(arr, 95)),
    }


def main():
    rng = np.random.default_rng(123)
    scaler, clf = fit_new_logic_pipeline()

    all_vecs = rng.uniform(0.1, 0.95, size=(N_WARMUP + N_FORMAL, N_BLOCKS))

    for i in range(N_WARMUP):
        old_logic(all_vecs[i], POLICY)
        new_logic(all_vecs[i], scaler, clf)

    old_times = np.empty(N_FORMAL)
    new_times = np.empty(N_FORMAL)

    for i in range(N_FORMAL):
        vec = all_vecs[N_WARMUP + i]

        t0 = time.perf_counter()
        old_logic(vec, POLICY)
        t1 = time.perf_counter()
        old_times[i] = (t1 - t0) * 1000

        t0 = time.perf_counter()
        new_logic(vec, scaler, clf)
        t1 = time.perf_counter()
        new_times[i] = (t1 - t0) * 1000

    s_old = stats(old_times)
    s_new = stats(new_times)
    delta_mean = s_new["mean"] - s_old["mean"]

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep_index", "old_ms", "new_ms"])
        for i in range(N_FORMAL):
            w.writerow([i, old_times[i], new_times[i]])

    print(f"N_WARMUP={N_WARMUP} N_FORMAL={N_FORMAL}")
    print(f"T_old  mean={s_old['mean']:.4f}ms median={s_old['median']:.4f}ms sd={s_old['sd']:.4f}ms p95={s_old['p95']:.4f}ms")
    print(f"T_new  mean={s_new['mean']:.4f}ms median={s_new['median']:.4f}ms sd={s_new['sd']:.4f}ms p95={s_new['p95']:.4f}ms")
    print(f"DeltaT(mean) = {delta_mean:.4f}ms")


if __name__ == "__main__":
    main()
