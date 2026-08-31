"""
Full re-comparison of ALL original matching methods (Keyword TF-IDF baseline,
Semantic SBERT, Hybrid fusion) under the NEW classifier-based decision logic,
against their original threshold-based numbers -- all recomputed fresh in this single run for guaranteed consistency.
"""

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
    "MiniLM (thesis baseline)": "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2",
    "BGE-large-v1.5": "/kaggle/input/models/levantaokkz/bge-large-en-v1.5/transformers/default/1/bge-large-en-v1.5",
}

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]

CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]

SUMMARY_ROWS = []

def classify_threshold(score, weak_th, aligned_th):
    if score >= aligned_th:
        return "Aligned"
    elif score >= weak_th:
        return "Weakly Aligned"
    else:
        return "Mismatched"


def grid_search_best_threshold(scores, gold, th_candidates=None):
    if th_candidates is None:
        th_candidates = np.arange(0.05, 0.96, 0.02)
    best = None
    for weak_th in th_candidates:
        for aligned_th in th_candidates:
            if aligned_th <= weak_th:
                continue
            preds = [classify_threshold(s, weak_th, aligned_th) for s in scores]
            f1 = f1_score(gold, preds, labels=LABELS, average="macro", zero_division=0)
            if best is None or f1 > best[0]:
                best = (f1, weak_th, aligned_th, preds)
    return best


def build_features(sims):
    mean_sim = sims.mean(axis=1, keepdims=True)
    std_sim = sims.std(axis=1, keepdims=True)
    max_sim = sims.max(axis=1, keepdims=True)
    margin = max_sim - mean_sim
    return np.hstack([sims, mean_sim, std_sim, margin])


def min_max_normalise(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def report_confusion(name, gold, preds):
    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, labels=LABELS, average="macro", zero_division=0)
    per_class = f1_score(gold, preds, labels=LABELS, average=None, zero_division=0)
    cm = confusion_matrix(gold, preds, labels=LABELS)
    print(f"\n[{name}] accuracy={acc:.4f}  macro-F1={f1:.4f}")
    print(f"  per-class F1 ({LABELS}): {per_class}")
    print(f"  confusion matrix (rows=gold, cols=pred, labels={LABELS}):")
    print(cm)
    return f1, acc


def cv_classifier(X, gold, name):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    preds = cross_val_predict(clf, X, gold, cv=skf)
    f1, acc = report_confusion(f"{name} -- classifier, 10-fold CV", gold, preds)
    return f1, acc


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
    gold = df["gold_label"].tolist()

    print("\n" + "*" * 70)
    print("  A. KEYWORD (TF-IDF)")
    print("*" * 70)
    kw_sims = compute_tfidf_sims(df)
    kw_max = kw_sims.max(axis=1)
    kw_f1_thresh, _, _, kw_preds_thresh = grid_search_best_threshold(kw_max, gold)
    report_confusion("Keyword -- threshold baseline (in-sample)", gold, kw_preds_thresh)
    SUMMARY_ROWS.append(("Keyword (TF-IDF)", "threshold (in-sample)", kw_f1_thresh, accuracy_score(gold, kw_preds_thresh)))

    kw_X = build_features(kw_sims)
    kw_f1_cv, kw_acc_cv = cv_classifier(kw_X, gold, "Keyword")
    SUMMARY_ROWS.append(("Keyword (TF-IDF)", "classifier (10-fold CV)", kw_f1_cv, kw_acc_cv))


    semantic_sims_by_model = {}
    for model_name, model_path in SEMANTIC_MODELS.items():
        print("\n" + "*" * 70)
        print(f"  SEMANTIC: {model_name}")
        print("*" * 70)
        print(f" Loading {model_path}")
        sem_sims = compute_semantic_sims(df, model_path)
        semantic_sims_by_model[model_name] = sem_sims

        sem_max = sem_sims.max(axis=1)
        sem_f1_thresh, _, _, sem_preds_thresh = grid_search_best_threshold(sem_max, gold)
        report_confusion(f"Semantic ({model_name}) -- threshold baseline (in-sample)", gold, sem_preds_thresh)
        SUMMARY_ROWS.append((f"Semantic ({model_name})", "threshold (in-sample)", sem_f1_thresh, accuracy_score(gold, sem_preds_thresh)))

        sem_X = build_features(sem_sims)
        sem_f1_cv, sem_acc_cv = cv_classifier(sem_X, gold, f"Semantic ({model_name})")
        SUMMARY_ROWS.append((f"Semantic ({model_name})", "classifier (10-fold CV)", sem_f1_cv, sem_acc_cv))


    print("\n" + "*" * 70)
    print("  D. HYBRID (Keyword + Semantic/MiniLM)")
    print("*" * 70)
    minilm_name = "MiniLM (thesis baseline)"
    sem_sims = semantic_sims_by_model[minilm_name]
    sem_max = sem_sims.max(axis=1)

    sem_norm = min_max_normalise(sem_max)
    kw_norm = min_max_normalise(kw_max)
    alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)
    th_candidates = np.round(np.arange(0.0, 1.01, 0.02), 3)
    best_hybrid = None
    for alpha in alphas:
        hybrid_score = alpha * sem_norm + (1 - alpha) * kw_norm
        f1, weak_th, aligned_th, preds = grid_search_best_threshold(hybrid_score, gold, th_candidates)
        if best_hybrid is None or f1 > best_hybrid[0]:
            best_hybrid = (f1, alpha, weak_th, aligned_th, preds)
    hyb_f1_thresh, best_alpha, _, _, hyb_preds_thresh = best_hybrid
    print(f"\n Best hybrid alpha (semantic weight) = {best_alpha}")
    report_confusion("Hybrid -- threshold baseline (in-sample, alpha-fusion)", gold, hyb_preds_thresh)
    SUMMARY_ROWS.append(("Hybrid (alpha-fusion)", "threshold (in-sample)", hyb_f1_thresh, accuracy_score(gold, hyb_preds_thresh)))

    sem_X = build_features(sem_sims)
    kw_X_local = build_features(kw_sims)
    hybrid_X = np.hstack([sem_X, kw_X_local])  # 18-dim: 9 semantic + 9 keyword
    hyb_f1_cv, hyb_acc_cv = cv_classifier(hybrid_X, gold, "Hybrid (concat features)")
    SUMMARY_ROWS.append(("Hybrid (concat features)", "classifier (10-fold CV)", hyb_f1_cv, hyb_acc_cv))


    print("\n" + "*" * 70)
    print("  SUMMARY: all methods x both decision logics")
    print("*" * 70)
    print(f"{'Method':<32}{'Decision logic':<26}{'macro-F1':>10}{'accuracy':>10}")
    print("-" * 70)
    for method, logic, f1, acc in SUMMARY_ROWS:
        print(f"{method:<32}{logic:<26}{f1:>10.4f}{acc:>10.4f}")

    summary_df = pd.DataFrame(SUMMARY_ROWS, columns=["method", "decision_logic", "macro_f1", "accuracy"])
    summary_df.to_csv("experiment_full_method_comparison_summary.csv", index=False)
    print("\nOK, Saved experiment_full_method_comparison_summary.csv")


if __name__ == "__main__":
    main()
