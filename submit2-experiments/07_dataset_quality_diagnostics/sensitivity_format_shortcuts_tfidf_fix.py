"""
Fix for a leak in sensitivity_format_shortcuts.py's run_tfidf_classifier().
"""
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET_PATH = "dataset_items_final.csv"
LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]

CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]


def strip_explain_and_normalize_periods(text: str) -> str:
    t = text.replace("Explain:", "")
    t = re.sub(r"\s+", " ", t).strip()
    if t.endswith("."):
        body = t[:-1]
    else:
        body = t
    body = body.replace(". ", "; ")
    t = body.rstrip(".;, ") + "."
    return t


def build_features(sims):
    mean_sim = sims.mean(axis=1, keepdims=True)
    std_sim = sims.std(axis=1, keepdims=True)
    max_sim = sims.max(axis=1, keepdims=True)
    margin = max_sim - mean_sim
    return np.hstack([sims, mean_sim, std_sim, margin])


def run_tfidf_classifier_leakfixed(texts, gold_labels, label, skf):

    texts = list(texts)
    n = len(texts)
    gold_arr = np.array(gold_labels)
    pred = np.empty(n, dtype=object)
    for train_idx, test_idx in skf.split(np.zeros(n), gold_arr):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        vec = TfidfVectorizer(stop_words="english")
        vec.fit(CAPABILITY_CHUNKS + train_texts)
        cap_tfidf = vec.transform(CAPABILITY_CHUNKS)
        train_sims = (vec.transform(train_texts) @ cap_tfidf.T).toarray()
        test_sims = (vec.transform(test_texts) @ cap_tfidf.T).toarray()

        X_train = build_features(train_sims)
        X_test = build_features(test_sims)
        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(X_train), gold_arr[train_idx])
        pred[test_idx] = clf.predict(scaler.transform(X_test))

    f1 = f1_score(gold_labels, pred, labels=LABELS, average="macro", zero_division=0)
    print(f"[{label}] TF-IDF 9-dim semantic classifier macro-F1 (leak-fixed, single-loop, matches authoritative pipeline) = {f1:.4f}")
    return f1


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    gold_labels = df["gold_label"].tolist()
    print(f"OK, Loaded {len(df)} items.")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    texts_before = df["item_text"].tolist()
    texts_after = [strip_explain_and_normalize_periods(t) for t in texts_before]

    print("\n    BEFORE (current frozen dataset text)    ")
    f1_before = run_tfidf_classifier_leakfixed(texts_before, gold_labels, "BEFORE", skf)

    print("\n    AFTER (Explain: stripped, periods normalized to 1/item, diagnostic-only)    ")
    f1_after = run_tfidf_classifier_leakfixed(texts_after, gold_labels, "AFTER", skf)

    print("\n    Summary (leak-fixed, comparable to the authoritative 0.6957 TF-IDF baseline)    ")
    print(f"  TF-IDF semantic macro-F1:  {f1_before:.4f} -> {f1_after:.4f}  (delta {f1_after - f1_before:+.4f})")
    print(f"  For reference, the ORIGINAL sensitivity_format_shortcuts.py (whole-corpus TF-IDF fit) reported 0.7043 -> 0.7043.")
    print(f"  Authoritative TF-IDF classifier baseline (Table 11/14/16/19/20 of the master report) = 0.6957.")


if __name__ == "__main__":
    main()
