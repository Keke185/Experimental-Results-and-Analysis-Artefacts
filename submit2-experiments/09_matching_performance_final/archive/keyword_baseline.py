"""
Keyword-based baseline for comparison against the Sentence-BERT semantic matching
"""
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

LABELS_ORDER = ["Aligned", "Weakly Aligned", "Mismatched"]


CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]


def classify_score(score: float, aligned_th: float, weak_th: float) -> str:
    if score >= aligned_th:
        return "Aligned"
    elif score >= weak_th:
        return "Weakly Aligned"
    else:
        return "Mismatched"


def main():
    print("*" * 50)
    print("  Keyword Baseline (TF-IDF) - Local Matching")
    print("*" * 50)

    dataset_path = "dataset_items.csv"  # adjust path as needed (e.g. Kaggle input path)

    print(f"\n Loading evaluation dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} evaluation items")

    print("\n Fitting TF-IDF vectoriser over capability chunks + evaluation items")
    vectorizer = TfidfVectorizer(stop_words="english")
    corpus = CAPABILITY_CHUNKS + df["item_text"].tolist()
    X = vectorizer.fit_transform(corpus)
    chunk_vectors = X[: len(CAPABILITY_CHUNKS)]
    item_vectors = X[len(CAPABILITY_CHUNKS):]

    print(" Computing TF-IDF cosine similarity against role capability chunks "
          "(max-similarity strategy, same as the semantic pipeline)")
    sim_matrix = cosine_similarity(item_vectors, chunk_vectors)
    max_scores = sim_matrix.max(axis=1)
    best_block_idx = sim_matrix.argmax(axis=1)

    df["keyword_score"] = max_scores
    df["best_capability_block"] = best_block_idx

    print("\n Keyword score distribution by gold_label:")
    stats = df.groupby("gold_label")["keyword_score"].describe()
    print(stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])

    print("\n Grid-searching threshold pairs to maximise macro-F1 "
          "(same calibration procedure as the semantic pipeline, for a fair comparison)")
    y_true = df["gold_label"]
    candidates = np.round(np.arange(0.01, max_scores.max() + 0.02, 0.005), 3)

    best = None
    for weak_th, aligned_th in product(candidates, candidates):
        if aligned_th <= weak_th:
            continue
        y_pred = [classify_score(s, aligned_th, weak_th) for s in max_scores]
        acc = accuracy_score(y_true, y_pred)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS_ORDER, zero_division=0
        )
        macro_f1 = f1.mean()
        if best is None or macro_f1 > best[0]:
            best = (macro_f1, acc, weak_th, aligned_th)

    macro_f1, acc, weak_th, aligned_th = best
    print(f"\nOK, Best keyword thresholds: weakly_aligned={weak_th}, aligned={aligned_th}")
    print(f"     -> macro-F1 = {macro_f1:.4f}, accuracy = {acc:.4f}")

    df["predicted_label"] = [classify_score(s, aligned_th, weak_th) for s in max_scores]
    y_pred = df["predicted_label"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS_ORDER, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)

    print("\nPer-class metrics at best thresholds:")
    for lbl, p, r, f, s in zip(LABELS_ORDER, precision, recall, f1, support):
        print(f"  {lbl:15s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  n={s}")

    print(f"\n Confusion Matrix (rows=true, cols=pred), label order = {LABELS_ORDER}")
    print(cm)

    out_path = "keyword_baseline_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nOK, Full per-item results saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
