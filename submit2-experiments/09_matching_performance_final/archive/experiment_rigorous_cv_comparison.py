
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
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


def fast_macro_f1_acc(pred_code, true_code):
    idx = true_code * 3 + pred_code
    cm = np.bincount(idx, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_totals = cm.sum(axis=0)
    precision = np.where(pred_totals > 0, tp / np.maximum(pred_totals, 1), 0.0)
    recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    return f1.mean(), tp.sum() / true_code.shape[0]


def code_classify(scores, weak_th, aligned_th):
    return np.where(scores >= aligned_th, 0, np.where(scores >= weak_th, 1, 2))


def best_thresholds_on(scores, true_code, th_candidates):
    best = None
    for weak_th in th_candidates:
        for aligned_th in th_candidates:
            if aligned_th <= weak_th:
                continue
            pred_code = code_classify(scores, weak_th, aligned_th)
            f1, acc = fast_macro_f1_acc(pred_code, true_code)
            if best is None or f1 > best[0]:
                best = (f1, weak_th, aligned_th)
    return best


def min_max_normalise(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def build_features(sims):
    mean_sim = sims.mean(axis=1, keepdims=True)
    std_sim = sims.std(axis=1, keepdims=True)
    max_sim = sims.max(axis=1, keepdims=True)
    margin = max_sim - mean_sim
    return np.hstack([sims, mean_sim, std_sim, margin])


def report_confusion(name, gold_labels, pred_labels):
    acc = accuracy_score(gold_labels, pred_labels)
    f1 = f1_score(gold_labels, pred_labels, labels=LABELS, average="macro", zero_division=0)
    per_class = f1_score(gold_labels, pred_labels, labels=LABELS, average=None, zero_division=0)
    cm = confusion_matrix(gold_labels, pred_labels, labels=LABELS)
    print(f"\n[{name}] accuracy={acc:.4f}  macro-F1={f1:.4f}")
    print(f"  per-class F1 ({LABELS}): {per_class}")
    print(f"  confusion matrix (rows=gold, cols=pred, labels={LABELS}):")
    print(cm)
    return f1, acc


def cv_threshold_method(scores, true_code, skf, th_candidates):

    n = len(true_code)
    oof_pred = np.full(n, -1, dtype=int)
    for train_idx, test_idx in skf.split(scores.reshape(-1, 1), true_code):
        best = best_thresholds_on(scores[train_idx], true_code[train_idx], th_candidates)
        _, weak_th, aligned_th = best
        oof_pred[test_idx] = code_classify(scores[test_idx], weak_th, aligned_th)
    assert (oof_pred >= 0).all()
    return oof_pred


def cv_hybrid_threshold_method(sem_scores, kw_scores, true_code, skf, alphas, th_candidates):

    n = len(true_code)
    oof_pred = np.full(n, -1, dtype=int)
    for train_idx, test_idx in skf.split(sem_scores.reshape(-1, 1), true_code):
        sem_tr, kw_tr = sem_scores[train_idx], kw_scores[train_idx]
        sem_lo, sem_hi = sem_tr.min(), sem_tr.max()
        kw_lo, kw_hi = kw_tr.min(), kw_tr.max()
        sem_norm_tr = (sem_tr - sem_lo) / max(sem_hi - sem_lo, 1e-12)
        kw_norm_tr = (kw_tr - kw_lo) / max(kw_hi - kw_lo, 1e-12)

        best_overall = None
        for alpha in alphas:
            hybrid_tr = alpha * sem_norm_tr + (1 - alpha) * kw_norm_tr
            f1, weak_th, aligned_th = best_thresholds_on(hybrid_tr, true_code[train_idx], th_candidates)
            if best_overall is None or f1 > best_overall[0]:
                best_overall = (f1, alpha, weak_th, aligned_th)
        _, alpha, weak_th, aligned_th = best_overall

        sem_te = (sem_scores[test_idx] - sem_lo) / max(sem_hi - sem_lo, 1e-12)
        kw_te = (kw_scores[test_idx] - kw_lo) / max(kw_hi - kw_lo, 1e-12)
        hybrid_te = alpha * sem_te + (1 - alpha) * kw_te
        oof_pred[test_idx] = code_classify(hybrid_te, weak_th, aligned_th)
    assert (oof_pred >= 0).all()
    return oof_pred


def cv_classifier_method(X, gold_labels, skf):
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    return cross_val_predict(clf, X, gold_labels, cv=skf)


def compute_tfidf_sims(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    corpus = CAPABILITY_CHUNKS + df["item_text"].tolist()
    X = vectorizer.fit_transform(corpus)
    chunk_vectors = X[: len(CAPABILITY_CHUNKS)]
    item_vectors = X[len(CAPABILITY_CHUNKS):]
    return sk_cosine_similarity(item_vectors, chunk_vectors)


def compute_semantic_sims(df, model_path):
    model = SentenceTransformer(model_path, local_files_only=True)
    cap_vectors = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True, show_progress_bar=True)
    item_vectors = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    return item_vectors @ cap_vectors.T


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items")
    gold_labels = df["gold_label"].tolist()
    true_code = np.array([LABEL_TO_CODE[l] for l in gold_labels])

    # SAME fold assignment reused for every method
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    th_candidates = np.arange(0.05, 0.96, 0.03)
    alphas = np.round(np.arange(0.0, 1.01, 0.1), 2)

    t0 = time.time()

    print("\n" + "*" * 70 + "\n  A. KEYWORD (TF-IDF)\n" + "*" * 70)
    kw_sims = compute_tfidf_sims(df)
    kw_max = kw_sims.max(axis=1)

    kw_th_oof = cv_threshold_method(kw_max, true_code, skf, th_candidates)
    kw_th_labels = [CODE_TO_LABEL[c] for c in kw_th_oof]
    f1, acc = report_confusion("Keyword -- threshold, 10-fold CV (fair)", gold_labels, kw_th_labels)
    SUMMARY_ROWS.append(("Keyword (TF-IDF)", "threshold, CV (fair)", f1, acc))

    kw_X = build_features(kw_sims)
    kw_clf_oof = cv_classifier_method(kw_X, gold_labels, skf)
    f1, acc = report_confusion("Keyword -- classifier, 10-fold CV", gold_labels, kw_clf_oof)
    SUMMARY_ROWS.append(("Keyword (TF-IDF)", "classifier, CV", f1, acc))

    semantic_sims = {}
    for model_name, model_path in SEMANTIC_MODELS.items():
        print("\n" + "*" * 70 + f"\n  SEMANTIC: {model_name}\n" + "*" * 70)
        print(f"Loading {model_path} ...")
        sem_sims = compute_semantic_sims(df, model_path)
        semantic_sims[model_name] = sem_sims
        sem_max = sem_sims.max(axis=1)

        sem_th_oof = cv_threshold_method(sem_max, true_code, skf, th_candidates)
        sem_th_labels = [CODE_TO_LABEL[c] for c in sem_th_oof]
        f1, acc = report_confusion(f"Semantic ({model_name}) -- threshold, 10-fold CV (fair)", gold_labels, sem_th_labels)
        SUMMARY_ROWS.append((f"Semantic ({model_name})", "threshold, CV (fair)", f1, acc))

        sem_X = build_features(sem_sims)
        sem_clf_oof = cv_classifier_method(sem_X, gold_labels, skf)
        f1, acc = report_confusion(f"Semantic ({model_name}) -- classifier, 10-fold CV", gold_labels, sem_clf_oof)
        SUMMARY_ROWS.append((f"Semantic ({model_name})", "classifier, CV", f1, acc))

    print("\n" + "*" * 70 + "\n  D. HYBRID (Keyword + Semantic/MiniLM)\n" + "*" * 70)
    sem_sims_minilm = semantic_sims["MiniLM"]
    sem_max_minilm = sem_sims_minilm.max(axis=1)

    hyb_th_oof = cv_hybrid_threshold_method(sem_max_minilm, kw_max, true_code, skf, alphas, th_candidates)
    hyb_th_labels = [CODE_TO_LABEL[c] for c in hyb_th_oof]
    f1, acc = report_confusion("Hybrid -- threshold (alpha-fusion), 10-fold CV (fair)", gold_labels, hyb_th_labels)
    SUMMARY_ROWS.append(("Hybrid (alpha-fusion)", "threshold, CV (fair)", f1, acc))

    sem_X_minilm = build_features(sem_sims_minilm)
    kw_X_local = build_features(kw_sims)
    hybrid_X = np.hstack([sem_X_minilm, kw_X_local])
    hyb_clf_oof = cv_classifier_method(hybrid_X, gold_labels, skf)
    f1, acc = report_confusion("Hybrid (concat features) -- classifier, 10-fold CV", gold_labels, hyb_clf_oof)
    SUMMARY_ROWS.append(("Hybrid (concat features)", "classifier, CV", f1, acc))

    print(f"\n Total runtime: {time.time() - t0:.1f}s")

    print("\n" + "*" * 70)
    print("  FINAL SUMMARY: fully symmetric 10-fold CV comparison, all methods")
    print("*" * 70)
    print(f"{'Method':<28}{'Decision logic':<24}{'macro-F1':>10}{'accuracy':>10}")
    print("-" * 70)
    for method, logic, f1, acc in SUMMARY_ROWS:
        print(f"{method:<28}{logic:<24}{f1:>10.4f}{acc:>10.4f}")

    out = pd.DataFrame(SUMMARY_ROWS, columns=["method", "decision_logic", "macro_f1", "accuracy"])
    out.to_csv("experiment_rigorous_cv_comparison_summary.csv", index=False)
    print("\nOK. Saved experiment_rigorous_cv_comparison_summary.csv")

if __name__ == "__main__":
    main()
