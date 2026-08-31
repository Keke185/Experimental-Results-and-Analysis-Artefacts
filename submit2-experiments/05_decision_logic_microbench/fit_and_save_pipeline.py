import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

N_BLOCKS = 6
OUT_PATH = "new_logic_pipeline.pkl"


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

if __name__ == "__main__":
    scaler, clf = fit_new_logic_pipeline()
    with open(OUT_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf}, f)
    print(f"saved {OUT_PATH}")
