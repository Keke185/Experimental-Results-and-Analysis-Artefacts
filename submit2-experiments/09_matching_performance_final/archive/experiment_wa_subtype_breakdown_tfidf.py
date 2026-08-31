
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "dataset_items_final.csv"

WA_D_IDS = {
    "Q_104","Q_113","Q_114","Q_115","Q_116","Q_119","Q_120","Q_121","Q_122",
    "Q_123","Q_124","Q_125","Q_126","Q_127","Q_128","Q_130","Q_136","Q_138",
    "Q_139","Q_143","Q_146","Q_147","Q_148","Q_149","Q_150",
}
WA_F_IDS = {
    "Q_101","Q_102","Q_103","Q_105","Q_106","Q_107","Q_108","Q_109","Q_110",
    "Q_111","Q_112","Q_117","Q_118","Q_129","Q_131","Q_132","Q_133","Q_134",
    "Q_135","Q_137","Q_140","Q_141","Q_142","Q_144","Q_145",
}

CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]


def build_features(sims):
    mean_sim = sims.mean(axis=1, keepdims=True)
    std_sim = sims.std(axis=1, keepdims=True)
    max_sim = sims.max(axis=1, keepdims=True)
    margin = max_sim - mean_sim
    return np.hstack([sims, mean_sim, std_sim, margin])


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"[OK] Loaded {len(df)} items.")

    corpus = CAPABILITY_CHUNKS + df["item_text"].tolist()
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(corpus)
    cap_vecs = tfidf[: len(CAPABILITY_CHUNKS)]
    item_vecs = tfidf[len(CAPABILITY_CHUNKS):]
    sims = (item_vecs @ cap_vecs.T).toarray()

    gold = df["gold_label"].to_numpy()
    ids = df["item_id"].to_numpy()
    X = build_features(sims)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    pred = cross_val_predict(clf, X_scaled, gold, cv=skf)

    print("\n    Overall classifier report (TF-IDF, 10-fold CV)    ")
    print(classification_report(gold, pred, digits=4))
    print("Confusion matrix (labels=[Aligned, Weakly Aligned, Mismatched]):")
    print(confusion_matrix(gold, pred, labels=["Aligned", "Weakly Aligned", "Mismatched"]))

    subtype = []
    for i in ids:
        if i in WA_D_IDS:
            subtype.append("WA-D")
        elif i in WA_F_IDS:
            subtype.append("WA-F")
        else:
            subtype.append("n/a")
    subtype = np.array(subtype)

    wa_mask = gold == "Weakly Aligned"
    print(f"\nOK, {wa_mask.sum()} Weakly Aligned items total "
          f"(WA-F={np.sum((subtype=='WA-F') & wa_mask)}, "
          f"WA-D={np.sum((subtype=='WA-D') & wa_mask)})")

    correct = pred == gold

    for st in ["WA-F", "WA-D"]:
        m = wa_mask & (subtype == st)
        n = m.sum()
        acc = correct[m].mean() if n else float("nan")
        print(f"\n--- {st} (Transferable Foundation Overlap)" if st == "WA-F"
              else f"\n--- {st} (Domain-Adjacent Engineering)")
        print(f"n={n}, recall (classified as Weakly Aligned)={acc:.4f}")
        mis_counts = pd.Series(pred[m]).value_counts()
        print("Predicted-label breakdown:")
        for label, cnt in mis_counts.items():
            print(f"  -> predicted '{label}': {cnt} ({cnt/n*100:.1f}%)")

    # Per-item detail table
    detail = pd.DataFrame({
        "item_id": ids[wa_mask],
        "subtype": subtype[wa_mask],
        "gold_label": gold[wa_mask],
        "predicted": pred[wa_mask],
        "correct": correct[wa_mask],
    }).sort_values(["subtype", "correct", "item_id"])
    print("\n    Per-item detail (Weakly Aligned only, TF-IDF classifier CV predictions)    ")
    print(detail.to_string(index=False))

    misclassified = detail[~detail["correct"]]
    print(f"\n SUMMARY, Misclassified Weakly Aligned items: {len(misclassified)}/{wa_mask.sum()}")
    print(f"  WA-F misclassified: {(misclassified['subtype']=='WA-F').sum()} / "
          f"{(subtype[wa_mask]=='WA-F').sum()}")
    print(f"  WA-D misclassified: {(misclassified['subtype']=='WA-D').sum()} / "
          f"{(subtype[wa_mask]=='WA-D').sum()}")

    detail.to_csv("wa_subtype_breakdown_tfidf.csv", index=False)
    print("\n OK, Saved wa_subtype_breakdown_tfidf.csv")


if __name__ == "__main__":
    main()
