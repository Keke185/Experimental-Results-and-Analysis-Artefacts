
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items-final/dataset_items_final.csv"
SEMANTIC_MODELS = {
    "MiniLM": "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2",
    "BGE-large-v1.5": "/kaggle/input/models/levantaokkz/bge-large-en-v1.5/transformers/default/1/bge-large-en-v1.5",
}

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]
LABEL_TO_CODE = {lbl: i for i, lbl in enumerate(LABELS)}
CODE_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_CODE.items()}

CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]

SUMMARY_ROWS = []


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


def report_summary(name, gold_labels, pred_labels, fold_f1s):
    acc = accuracy_score(gold_labels, pred_labels)
    f1_pooled = f1_score(gold_labels, pred_labels, labels=LABELS, average="macro", zero_division=0)
    cm = confusion_matrix(gold_labels, pred_labels, labels=LABELS)
    fold_arr = np.array(fold_f1s)
    mean_, std_ = fold_arr.mean(), fold_arr.std(ddof=1)
    ci95 = 1.96 * std_ / np.sqrt(len(fold_arr))
    print(f"\n[{name}]")
    print(f"  pooled out-of-fold macro-F1 = {f1_pooled:.4f}, accuracy = {acc:.4f}")
    print(f"  per-fold macro-F1: mean={mean_:.4f}  std={std_:.4f}  "
          f"approx 95% CI = [{mean_-ci95:.4f}, {mean_+ci95:.4f}]")
    print(f"  per-fold values: {np.round(fold_arr, 4).tolist()}")
    print(f"  confusion matrix (rows=gold, cols=pred, labels={LABELS}):")
    print(cm)
    SUMMARY_ROWS.append((name, f1_pooled, mean_, std_, ci95, acc))
    return f1_pooled, cm


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
    th_fold_f1, clf_fold_f1 = [], []

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
        pred_code_test = code_classify(test_max, weak_th, aligned_th)
        th_oof[test_idx] = pred_code_test
        th_fold_f1.append(fast_macro_f1(pred_code_test, true_code[test_idx]))

        X_train = build_features(train_sims)
        X_test = build_features(test_sims)
        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(X_train), [gold_labels[i] for i in train_idx])
        pred_labels_test = clf.predict(scaler.transform(X_test))
        pred_code_test_clf = np.array([LABEL_TO_CODE[l] for l in pred_labels_test])
        clf_oof[test_idx] = pred_code_test_clf
        clf_fold_f1.append(fast_macro_f1(pred_code_test_clf, true_code[test_idx]))

    th_labels = [CODE_TO_LABEL[c] for c in th_oof]
    clf_labels = [CODE_TO_LABEL[c] for c in clf_oof]
    report_summary("Keyword (TF-IDF) -- threshold, CV (fair, leak-fixed)", gold_labels, th_labels, th_fold_f1)
    report_summary("Keyword (TF-IDF) -- classifier, CV (leak-fixed)", gold_labels, clf_labels, clf_fold_f1)


def run_semantic_with_std(model_name, sem_sims, true_code, gold_labels, skf, th_candidates):
    n = len(true_code)
    sem_max = sem_sims.max(axis=1)
    X = build_features(sem_sims)

    th_oof = np.full(n, -1, dtype=int)
    clf_oof = np.full(n, -1, dtype=int)
    th_fold_f1, clf_fold_f1 = [], []

    for train_idx, test_idx in skf.split(np.zeros(n), true_code):
        _, weak_th, aligned_th = best_thresholds_on(sem_max[train_idx], true_code[train_idx], th_candidates)
        pred_code_test = code_classify(sem_max[test_idx], weak_th, aligned_th)
        th_oof[test_idx] = pred_code_test
        th_fold_f1.append(fast_macro_f1(pred_code_test, true_code[test_idx]))

        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        gold_arr = np.array(gold_labels)
        clf.fit(scaler.transform(X[train_idx]), gold_arr[train_idx])
        pred_labels_test = clf.predict(scaler.transform(X[test_idx]))
        pred_code_test_clf = np.array([LABEL_TO_CODE[l] for l in pred_labels_test])
        clf_oof[test_idx] = pred_code_test_clf
        clf_fold_f1.append(fast_macro_f1(pred_code_test_clf, true_code[test_idx]))

    th_labels = [CODE_TO_LABEL[c] for c in th_oof]
    clf_labels = [CODE_TO_LABEL[c] for c in clf_oof]
    report_summary(f"Semantic ({model_name}) -- threshold, CV (fair)", gold_labels, th_labels, th_fold_f1)
    report_summary(f"Semantic ({model_name}) -- classifier, CV", gold_labels, clf_labels, clf_fold_f1)


def run_hybrid_leakfixed(df, minilm_sims, true_code, gold_labels, skf, th_candidates, alphas):
    texts = df["item_text"].tolist()
    n = len(df)
    sem_max = minilm_sims.max(axis=1)
    sem_X_full = build_features(minilm_sims)

    th_oof = np.full(n, -1, dtype=int)
    clf_oof = np.full(n, -1, dtype=int)
    th_fold_f1, clf_fold_f1 = [], []

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
        pred_code_test = code_classify(hybrid_te, weak_th, aligned_th)
        th_oof[test_idx] = pred_code_test
        th_fold_f1.append(fast_macro_f1(pred_code_test, true_code[test_idx]))

        kw_X_train = build_features(kw_train_sims)
        kw_X_test = build_features(kw_test_sims)
        X_train = np.hstack([sem_X_full[train_idx], kw_X_train])
        X_test = np.hstack([sem_X_full[test_idx], kw_X_test])
        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        gold_arr = np.array(gold_labels)
        clf.fit(scaler.transform(X_train), gold_arr[train_idx])
        pred_labels_test = clf.predict(scaler.transform(X_test))
        pred_code_test_clf = np.array([LABEL_TO_CODE[l] for l in pred_labels_test])
        clf_oof[test_idx] = pred_code_test_clf
        clf_fold_f1.append(fast_macro_f1(pred_code_test_clf, true_code[test_idx]))

    th_labels = [CODE_TO_LABEL[c] for c in th_oof]
    clf_labels = [CODE_TO_LABEL[c] for c in clf_oof]
    report_summary("Hybrid (alpha-fusion) -- threshold, CV (fair, leak-fixed)", gold_labels, th_labels, th_fold_f1)
    report_summary("Hybrid (concat features, 18-dim) -- classifier, CV (leak-fixed)", gold_labels, clf_labels, clf_fold_f1)


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f" Loaded {len(df)} items")
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
        run_semantic_with_std(model_name, sem_sims, true_code, gold_labels, skf, th_candidates)

    print("\n" + "*" * 70 + "\n  D. HYBRID (Keyword + Semantic/MiniLM), leak-fixed\n" + "*" * 70)
    run_hybrid_leakfixed(df, semantic_sims["MiniLM"], true_code, gold_labels, skf, th_candidates, alphas)

    print(f"\n Total runtime: {time.time() - t0:.1f}s")

    print("\n" + "*" * 90)
    print("  FINAL SUMMARY: leak-fixed, fully symmetric 10-fold CV, all methods")
    print("*" * 90)
    print(f"{'Method':<50}{'pooled F1':>10}{'mean F1':>10}{'std':>8}{'95% CI +/-':>12}{'acc':>8}")
    print("-" * 90)
    for name, f1_pooled, mean_, std_, ci95, acc in SUMMARY_ROWS:
        print(f"{name:<50}{f1_pooled:>10.4f}{mean_:>10.4f}{std_:>8.4f}{ci95:>12.4f}{acc:>8.4f}")

    out = pd.DataFrame(SUMMARY_ROWS, columns=["method", "pooled_macro_f1", "mean_fold_f1", "std_fold_f1", "ci95_halfwidth", "accuracy"])
    out.to_csv("experiment_rigorous_cv_v3_leakfixed_summary.csv", index=False)
    print("\n OK, Saved experiment_rigorous_cv_v3_leakfixed_summary.csv")


if __name__ == "__main__":
    main()
