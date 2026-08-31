"""
Experiment: replace max-similarity + global threshold with a lightweight
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
MODEL_CONFIGS = {
    "MiniLM (thesis baseline)": "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2",
    "BGE-large-v1.5 (strongest tested)": "/kaggle/input/models/levantaokkz/bge-large-en-v1.5/transformers/default/1/bge-large-en-v1.5",
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


def classify_threshold(score, weak_th, aligned_th):
    if score >= aligned_th:
        return "Aligned"
    elif score >= weak_th:
        return "Weakly Aligned"
    else:
        return "Mismatched"


def grid_search_best_threshold(scores, gold):
    best = None
    for weak_th in np.arange(0.05, 0.96, 0.02):
        for aligned_th in np.arange(weak_th, 0.96, 0.02):
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


def report_confusion(name, gold, preds, labels=LABELS):
    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, labels=labels, average="macro", zero_division=0)
    per_class = f1_score(gold, preds, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(gold, preds, labels=labels)
    print(f"\n[{name}] accuracy={acc:.4f}  macro-F1={f1:.4f}")
    print(f"  per-class F1 ({labels}): {per_class}")
    print(f"  confusion matrix (rows=gold, cols=pred, labels={labels}):")
    print(cm)
    return f1, acc, cm


def run_for_model(model_name, model_path, df):
    print("\n" + "*" * 70)
    print(f"  MODEL: {model_name}")
    print("*" * 70)

    print(f" Loading {model_path}")
    model = SentenceTransformer(model_path, local_files_only=True)

    print("Encoding capability chunks and dataset items")
    cap_vectors = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True, show_progress_bar=True)
    item_vectors = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)

    sims = item_vectors @ cap_vectors.T
    gold = df["gold_label"].tolist()

    max_scores = sims.max(axis=1)
    best_f1, weak_th, aligned_th, thresh_preds = grid_search_best_threshold(max_scores, gold)
    print(f"\n 1.Threshold baseline (in-sample grid search, weak={weak_th:.2f}, aligned={aligned_th:.2f})")
    report_confusion("Threshold baseline (in-sample)", gold, thresh_preds)

    # Obtain classifier features
    X = build_features(sims)

    clf_logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, multi_class="multinomial"),
    )

    clf_logreg.fit(X, gold)
    insample_preds = clf_logreg.predict(X)
    print(f"\n 2.Logistic regression, IN-SAMPLE (fit+predict on same 200 items  optimistic, for fair comparison to #1)")
    report_confusion("LogReg in-sample", gold, insample_preds)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_preds_logreg = cross_val_predict(clf_logreg, X, gold, cv=skf)
    print(f"\n 3.Logistic regression, 10-FOLD CROSS-VALIDATED (out-of-sample , the honest number)")
    report_confusion("LogReg 10-fold CV", gold, cv_preds_logreg)

    #Use linear SVM as a second classifier to improve robustness
    clf_svm = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", max_iter=5000),
    )

    cv_preds_svm = cross_val_predict(clf_svm, X, gold, cv=skf)
    print(f"\n Linear SVM, 10-FOLD CROSS-VALIDATED (second classifier, for robustness)")
    report_confusion("LinearSVC 10-fold CV", gold, cv_preds_svm)

    out = df[["item_id", "target_role", "gold_label"]].copy()
    for i in range(sims.shape[1]):
        out[f"sim_block_{i + 1}"] = sims[:, i]
    out["threshold_pred"] = thresh_preds
    out["logreg_insample_pred"] = insample_preds
    out["logreg_cv_pred"] = cv_preds_logreg
    out["svm_cv_pred"] = cv_preds_svm
    safe_name = model_name.split(" ")[0].replace("-", "_")
    fname = f"experiment_classifier_{safe_name}_results.csv"
    out.to_csv(fname, index=False)
    print(f"\n OK, Saved {fname}")


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items")

    for model_name, model_path in MODEL_CONFIGS.items():
        run_for_model(model_name, model_path, df)


if __name__ == "__main__":
    main()
