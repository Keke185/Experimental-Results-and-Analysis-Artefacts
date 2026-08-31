import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items-final/dataset_items_final.csv"
SEMANTIC_MODELS = {
    "MiniLM": "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2",
    "BGE-large-v1.5": "/kaggle/input/models/levantaokkz/bge-large-en-v1.5/transformers/default/1/bge-large-en-v1.5",
}

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]
LABEL_TO_CODE = {lbl: i for i, lbl in enumerate(LABELS)}
CODE_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_CODE.items()}
PLOT_LABELS = ["Aligned", "Weakly\nAligned", "Mismatched"]

CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]

RESULTS = {}

def fast_macro_f1(pred_code, true_code):
    idx = true_code * 3 + pred_code
    cm = np.bincount(idx, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_totals = cm.sum(axis=0)
    precision = np.where(pred_totals > 0, tp / np.maximum(pred_totals, 1), 0.0)
    recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    return f1.mean()


def code_classify(scores, weak_th, aligned_th):
    return np.where(scores >= aligned_th, 0, np.where(scores >= weak_th, 1, 2))


def best_thresholds_on(scores, true_code, th_candidates):
    best = None
    for weak_th in th_candidates:
        for aligned_th in th_candidates:
            if aligned_th <= weak_th:
                continue
            pred_code = code_classify(scores, weak_th, aligned_th)
            f1 = fast_macro_f1(pred_code, true_code)
            if best is None or f1 > best[0]:
                best = (f1, weak_th, aligned_th)
    return best


def build_features(sims):
    mean_sim = sims.mean(axis=1, keepdims=True)
    std_sim = sims.std(axis=1, keepdims=True)
    max_sim = sims.max(axis=1, keepdims=True)
    margin = max_sim - mean_sim
    return np.hstack([sims, mean_sim, std_sim, margin])


def record(name, gold_labels, pred_labels):
    cm = confusion_matrix(gold_labels, pred_labels, labels=LABELS)
    macro_f1 = f1_score(gold_labels, pred_labels, labels=LABELS, average="macro", zero_division=0)
    RESULTS[name] = (cm, macro_f1)
    print(f"[{name}] macro-F1={macro_f1:.4f}")
    print(cm)


def compute_semantic_sims(df, model_path):
    model = SentenceTransformer(model_path, local_files_only=True)
    cap_vectors = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True, show_progress_bar=True)
    item_vectors = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    return item_vectors @ cap_vectors.T


def run_tfidf_leakfixed(df, true_code, gold_labels, skf, th_candidates):
    texts = df["item_text"].tolist()
    n = len(df)
    th_oof = np.full(n, -1, dtype=int)
    clf_oof = np.full(n, -1, dtype=int)

    for train_idx, test_idx in skf.split(np.zeros(n), true_code):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        vec = TfidfVectorizer(stop_words="english")
        vec.fit(CAPABILITY_CHUNKS + train_texts)
        cap_tfidf = vec.transform(CAPABILITY_CHUNKS)
        train_sims = (vec.transform(train_texts) @ cap_tfidf.T).toarray()
        test_sims = (vec.transform(test_texts) @ cap_tfidf.T).toarray()

        train_max = train_sims.max(axis=1)
        test_max = test_sims.max(axis=1)
        _, weak_th, aligned_th = best_thresholds_on(train_max, true_code[train_idx], th_candidates)
        th_oof[test_idx] = code_classify(test_max, weak_th, aligned_th)

        X_train = build_features(train_sims)
        X_test = build_features(test_sims)
        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(X_train), [gold_labels[i] for i in train_idx])
        pred_labels_test = clf.predict(scaler.transform(X_test))
        clf_oof[test_idx] = np.array([LABEL_TO_CODE[l] for l in pred_labels_test])

    record("TF-IDF -- threshold (leak-fixed)", gold_labels, [CODE_TO_LABEL[c] for c in th_oof])
    record("TF-IDF -- classifier (leak-fixed)", gold_labels, [CODE_TO_LABEL[c] for c in clf_oof])


def run_semantic(model_name, sem_sims, true_code, gold_labels, skf, th_candidates):
    n = len(true_code)
    sem_max = sem_sims.max(axis=1)
    X = build_features(sem_sims)
    gold_arr = np.array(gold_labels)

    th_oof = np.full(n, -1, dtype=int)
    clf_oof = np.full(n, -1, dtype=int)

    for train_idx, test_idx in skf.split(np.zeros(n), true_code):
        _, weak_th, aligned_th = best_thresholds_on(sem_max[train_idx], true_code[train_idx], th_candidates)
        th_oof[test_idx] = code_classify(sem_max[test_idx], weak_th, aligned_th)

        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(X[train_idx]), gold_arr[train_idx])
        pred_labels_test = clf.predict(scaler.transform(X[test_idx]))
        clf_oof[test_idx] = np.array([LABEL_TO_CODE[l] for l in pred_labels_test])

    record(f"{model_name} -- threshold", gold_labels, [CODE_TO_LABEL[c] for c in th_oof])
    record(f"{model_name} -- classifier", gold_labels, [CODE_TO_LABEL[c] for c in clf_oof])


def run_hybrid_leakfixed(df, minilm_sims, true_code, gold_labels, skf, th_candidates, alphas):
    texts = df["item_text"].tolist()
    n = len(df)
    sem_max = minilm_sims.max(axis=1)
    sem_X_full = build_features(minilm_sims)
    gold_arr = np.array(gold_labels)

    th_oof = np.full(n, -1, dtype=int)
    clf_oof = np.full(n, -1, dtype=int)

    for train_idx, test_idx in skf.split(np.zeros(n), true_code):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        vec = TfidfVectorizer(stop_words="english")
        vec.fit(CAPABILITY_CHUNKS + train_texts)
        cap_tfidf = vec.transform(CAPABILITY_CHUNKS)
        kw_train_sims = (vec.transform(train_texts) @ cap_tfidf.T).toarray()
        kw_test_sims = (vec.transform(test_texts) @ cap_tfidf.T).toarray()
        kw_train_max = kw_train_sims.max(axis=1)
        kw_test_max = kw_test_sims.max(axis=1)

        sem_train_max = sem_max[train_idx]
        sem_test_max = sem_max[test_idx]
        sem_lo, sem_hi = sem_train_max.min(), sem_train_max.max()
        kw_lo, kw_hi = kw_train_max.min(), kw_train_max.max()
        sem_norm_tr = (sem_train_max - sem_lo) / max(sem_hi - sem_lo, 1e-12)
        kw_norm_tr = (kw_train_max - kw_lo) / max(kw_hi - kw_lo, 1e-12)

        best_overall = None
        for alpha in alphas:
            hybrid_tr = alpha * sem_norm_tr + (1 - alpha) * kw_norm_tr
            f1, weak_th, aligned_th = best_thresholds_on(hybrid_tr, true_code[train_idx], th_candidates)
            if best_overall is None or f1 > best_overall[0]:
                best_overall = (f1, alpha, weak_th, aligned_th)
        _, alpha, weak_th, aligned_th = best_overall

        sem_te = (sem_test_max - sem_lo) / max(sem_hi - sem_lo, 1e-12)
        kw_te = (kw_test_max - kw_lo) / max(kw_hi - kw_lo, 1e-12)
        hybrid_te = alpha * sem_te + (1 - alpha) * kw_te
        th_oof[test_idx] = code_classify(hybrid_te, weak_th, aligned_th)

        kw_X_train = build_features(kw_train_sims)
        kw_X_test = build_features(kw_test_sims)
        X_train = np.hstack([sem_X_full[train_idx], kw_X_train])
        X_test = np.hstack([sem_X_full[test_idx], kw_X_test])
        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(X_train), gold_arr[train_idx])
        pred_labels_test = clf.predict(scaler.transform(X_test))
        clf_oof[test_idx] = np.array([LABEL_TO_CODE[l] for l in pred_labels_test])

    record("Hybrid -- threshold (leak-fixed)", gold_labels, [CODE_TO_LABEL[c] for c in th_oof])
    record("Hybrid -- classifier, 18-dim (leak-fixed)", gold_labels, [CODE_TO_LABEL[c] for c in clf_oof])


def plot_one(ax, cm, title, f1):
    cm = np.array(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / row_sums
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(PLOT_LABELS, fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(PLOT_LABELS, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    for i in range(3):
        for j in range(3):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i,j]*100:.0f}%)", ha="center", va="center",
                     fontsize=8, color=color)
    ax.set_title(f"{title}\nmacro-F1={f1:.4f}", fontsize=9)


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items")
    gold_labels = df["gold_label"].tolist()
    true_code = np.array([LABEL_TO_CODE[l] for l in gold_labels])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    th_candidates = np.arange(0.05, 0.96, 0.03)
    alphas = np.round(np.arange(0.0, 1.01, 0.1), 2)

    t0 = time.time()

    print("\n" + "*" * 70 + "\n  A. KEYWORD (TF-IDF), leak-fixed\n" + "*" * 70)
    run_tfidf_leakfixed(df, true_code, gold_labels, skf, th_candidates)

    semantic_sims = {}
    for model_name, model_path in SEMANTIC_MODELS.items():
        print("\n" + "*" * 70 + f"\n  SEMANTIC: {model_name}\n" + "*" * 70)
        print(f" Loading {model_path}")
        sem_sims = compute_semantic_sims(df, model_path)
        semantic_sims[model_name] = sem_sims
        run_semantic(model_name, sem_sims, true_code, gold_labels, skf, th_candidates)

    print("\n" + "*" * 70 + "\n  D. HYBRID (Keyword + Semantic/MiniLM), leak-fixed\n" + "*" * 70)
    run_hybrid_leakfixed(df, semantic_sims["MiniLM"], true_code, gold_labels, skf, th_candidates, alphas)

    print(f"\n Total runtime: {time.time() - t0:.1f}s")

    method_order = [
        ("TF-IDF", "TF-IDF -- threshold (leak-fixed)", "TF-IDF -- classifier (leak-fixed)"),
        ("MiniLM", "MiniLM -- threshold", "MiniLM -- classifier"),
        ("BGE-large-v1.5", "BGE-large-v1.5 -- threshold", "BGE-large-v1.5 -- classifier"),
        ("Hybrid", "Hybrid -- threshold (leak-fixed)", "Hybrid -- classifier, 18-dim (leak-fixed)"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(9, 16))
    for i, (method, th_key, clf_key) in enumerate(method_order):
        for j, key in enumerate([th_key, clf_key]):
            cm, f1 = RESULTS[key]
            plot_one(axes[i, j], cm, key, f1)
    fig.suptitle("Final dataset -- live Kaggle run, leak-fixed 10-fold CV\n(rows=gold, cols=predicted, %=row-normalized)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("cm_kaggle_all_methods_grid.png", dpi=150)
    print("\nOK, Saved cm_kaggle_all_methods_grid.png")

    for method, th_key, clf_key in method_order:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        for j, key in enumerate([th_key, clf_key]):
            cm, f1 = RESULTS[key]
            plot_one(axes[j], cm, key, f1)
        fig.tight_layout()
        safe_name = method.replace(" ", "_").replace("-", "_").lower()
        fname = f"cm_kaggle_{safe_name}.png"
        fig.savefig(fname, dpi=150)
        print(f"OK, Saved {fname}")

    plt.close("all")

    print("\n" + "*" * 78)
    print("  FINAL SUMMARY")
    print("*" * 78)
    for name, (cm, f1) in RESULTS.items():
        print(f"{name:<45} macro-F1={f1:.4f}")


if __name__ == "__main__":
    main()
