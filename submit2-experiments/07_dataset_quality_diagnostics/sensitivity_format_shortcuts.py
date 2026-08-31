
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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


def format_features(texts):
    rows = []
    for t in texts:
        rows.append({
            "word_count": len(t.split()),
            "char_count": len(t),
            "bullet_count": t.count("•"),
            "newline_count": t.count("\n"),
            "has_Question": int("Question:" in t),
            "has_Explain": int("Explain:" in t),
            "period_count": t.count("."),
        })
    return pd.DataFrame(rows)


def run_format_classifier(df, gold_labels, label):
    X = format_features(df["item_text"]).to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    pred = cross_val_predict(clf, X_scaled, gold_labels, cv=skf)
    f1 = f1_score(gold_labels, pred, labels=LABELS, average="macro", zero_division=0)
    print(f"[{label}] format-shortcut classifier macro-F1 = {f1:.4f} "
          f"(majority-class baseline = 0.2222)")
    return f1


def run_tfidf_classifier(texts, gold_labels, label):
    corpus = CAPABILITY_CHUNKS + list(texts)
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(corpus)
    cap_vecs = tfidf[: len(CAPABILITY_CHUNKS)]
    item_vecs = tfidf[len(CAPABILITY_CHUNKS):]
    sims = (item_vecs @ cap_vecs.T).toarray()
    mean_sim = sims.mean(axis=1, keepdims=True)
    std_sim = sims.std(axis=1, keepdims=True)
    margin = sims.max(axis=1, keepdims=True) - mean_sim
    X = np.hstack([sims, mean_sim, std_sim, margin])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    pred = cross_val_predict(clf, X_scaled, gold_labels, cv=skf)
    f1 = f1_score(gold_labels, pred, labels=LABELS, average="macro", zero_division=0)
    print(f"[{label}] TF-IDF 9-dim semantic classifier macro-F1 = {f1:.4f}")
    return f1


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    gold_labels = df["gold_label"].tolist()
    print(f"OK, Loaded {len(df)} items.")

    n_explain = df["item_text"].str.contains("Explain:").sum()
    print(f"OK, {n_explain} items currently contain 'Explain:' "
          f"(by class: {df[df['item_text'].str.contains('Explain:')]['gold_label'].value_counts().to_dict()})")

    print("\n    BEFORE (current frozen dataset text)    ")
    f1_format_before = run_format_classifier(df, gold_labels, "BEFORE")
    f1_tfidf_before = run_tfidf_classifier(df["item_text"], gold_labels, "BEFORE")

    df_after = df.copy()
    df_after["item_text"] = df_after["item_text"].apply(strip_explain_and_normalize_periods)

    print("\n   AFTER (Explain: stripped, periods normalized to 1/item, diagnostic-only)    ")
    f1_format_after = run_format_classifier(df_after, gold_labels, "AFTER")
    f1_tfidf_after = run_tfidf_classifier(df_after["item_text"], gold_labels, "AFTER")

    print("\n    Summary    ")
    print(f"  Format-shortcut macro-F1:  {f1_format_before:.4f} -> {f1_format_after:.4f}  "
          f"(delta {f1_format_after - f1_format_before:+.4f})")
    print(f"  TF-IDF semantic macro-F1:  {f1_tfidf_before:.4f} -> {f1_tfidf_after:.4f}  "
          f"(delta {f1_tfidf_after - f1_tfidf_before:+.4f})")
    print("\n If the semantic classifier's macro-F1 barely moves while the format-shortcut")
    print("classifier drops further toward the 0.2222 baseline, that's direct evidence the")
    print("core semantic result does not depend on the residual Explain:/period signal.")


if __name__ == "__main__":
    main()
