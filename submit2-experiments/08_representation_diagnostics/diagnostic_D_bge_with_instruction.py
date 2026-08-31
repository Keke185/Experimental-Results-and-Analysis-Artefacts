"""
Diagnostic D: BGE Search Command Prefix Test
A/B diagnostics revealed that the recall rate for BGE-large-en-v1.5 mismatch categories is zero,
with a large number of samples misclassified as alignment.
This script verifies whether this phenomenon is caused by BGE score calibration: Add the officially
recommended search command to the sample text on the query side, while keeping the capability block
document side without a prefix;
The remaining configuration is completely consistent with Diagnostic A, only adding the command prefix to
pinpoint the source of the problem.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
MODEL_PATH = "/kaggle/input/models/levantaokkz/bge-large-en-v1.5/transformers/default/1/bge-large-en-v1.5"

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]

# BAAI's officially recommended command prefixes for query clients
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]


def classify(score, weak_th, aligned_th):
    if score >= aligned_th:
        return "Aligned"
    elif score >= weak_th:
        return "Weakly Aligned"
    else:
        return "Mismatched"


def grid_search_best(scores, gold):
    best = None
    for weak_th in np.arange(0.05, 0.96, 0.02):
        for aligned_th in np.arange(weak_th, 0.96, 0.02):
            preds = [classify(s, weak_th, aligned_th) for s in scores]
            f1 = f1_score(gold, preds, labels=LABELS, average="macro", zero_division=0)
            if best is None or f1 > best[0]:
                best = (f1, weak_th, aligned_th, preds)
    return best


def main():
    print("*" * 70)
    print("  Diagnostic D: BGE-large-en-v1.5 WITH query instruction prefix (v1 profile)")
    print("*" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f" Loaded {len(df)} items")

    print(f"\n Loading {MODEL_PATH} ")
    model = SentenceTransformer(MODEL_PATH, local_files_only=True)

    prefixed_items = [QUERY_INSTRUCTION + t for t in df["item_text"].tolist()]

    print(" Encoding capability chunks (no prefix, unchanged)")
    cap_vectors = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True, show_progress_bar=True)
    print(" Encoding dataset items WITH instruction prefix")
    item_vectors = model.encode(prefixed_items, normalize_embeddings=True, show_progress_bar=True)

    sims = item_vectors @ cap_vectors.T
    scores = sims.max(axis=1)

    raw_preds = [classify(s, 0.30, 0.50) for s in scores]
    raw_acc = accuracy_score(df["gold_label"], raw_preds)
    print(f"\n [Raw, default thresholds 0.50/0.30] accuracy={raw_acc:.4f}")

    best_f1, weak_th, aligned_th, preds = grid_search_best(scores, df["gold_label"])
    acc = accuracy_score(df["gold_label"], preds)
    cm = confusion_matrix(df["gold_label"], preds, labels=LABELS)
    per_class_f1 = f1_score(df["gold_label"], preds, labels=LABELS, average=None, zero_division=0)

    print(f"\n Calibrated, best thresholds: weak={weak_th:.2f}, aligned={aligned_th:.2f}")
    print(f"accuracy={acc:.4f}  macro-F1={best_f1:.4f}")
    print(f"per-class F1 ({LABELS}): {per_class_f1}")
    print(f"\n Confusion matrix (rows=gold, cols=pred, labels={LABELS}):")
    print(cm)

    out = df[["item_id", "target_role", "gold_label"]].copy()
    out["bge_prefixed_score"] = scores
    out["predicted_label"] = preds
    out.to_csv("diagnostic_D_bge_prefixed_results.csv", index=False)
    print("\n OK, Saved diagnostic_D_bge_prefixed_results.csv")

if __name__ == "__main__":
    main()
