"""
    Diagnosis C3: Investigate the actual composition of the two weakly aligned subclusters.
    Diagnosis C2 shows that the weakly aligned subclusters have the lowest intra-cluster cohesion among
    the three categories (mean 0.2389, aligned: 0.3304, mismatched: 0.5106); after forcing a binary split
    using KMeans, the intra-/inter-cluster differences are relatively small. This script analyzes the contents
    of the two subclusters to verify the hypothesis that the clustering splits correspond to different functional
    configuration modules, which is a reasonable inherent heterogeneity of the weakly aligned samples
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
MODEL_PATH = "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2"


RANDOM_STATE = 42

BLOCK_NAMES = [
    "Block1: Video Conferencing & Signaling",
    "Block2: Conference Mgmt & Business Control",
    "Block3: MCU & Media Engine",
    "Block4: Video Endpoints & Room Systems",
    "Block5: 4K UHD / Codecs / Multi-party Media",
    "Block6: CPaaS & Conversational AI",
]

CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]


def main():
    print("*" * 70)
    print("  Diagnostic C3: Weakly Aligned sub-cluster inspection")
    print("*" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items.")

    print(f"\n Loading {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH, local_files_only=True) if MODEL_PATH.startswith("/kaggle") else SentenceTransformer(MODEL_PATH)

    print(" Encoding capability chunks and all item texts")
    cap_vectors = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True, show_progress_bar=True)
    item_vectors = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    sims_to_blocks = item_vectors @ cap_vectors.T  # (n_items, 6)

    df["_best_block_idx"] = sims_to_blocks.argmax(axis=1)
    df["_best_block_score"] = sims_to_blocks.max(axis=1)

    wa_mask = df["gold_label"] == "Weakly Aligned"
    wa_df = df[wa_mask].copy()
    wa_vectors = item_vectors[wa_mask.to_numpy()]

    print(f"\n Re-running KMeans(k=2) on the {len(wa_df)} Weakly Aligned items "
          f"(identical settings to Diagnostic C2,should reproduce the 22/28 split)")
    km = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit(wa_vectors)
    wa_df["_cluster"] = km.labels_
    print(f"OK, Cluster sizes: cluster 0 = {(wa_df['_cluster'] == 0).sum()}, "
          f"cluster 1 = {(wa_df['_cluster'] == 1).sum()}")

    print("\n" + "*" * 70)
    print("  Cross-tab: cluster x best-matching capability block")
    print("*" * 70)
    wa_df["_best_block_name"] = wa_df["_best_block_idx"].apply(lambda i: BLOCK_NAMES[i])
    crosstab = pd.crosstab(wa_df["_cluster"], wa_df["_best_block_name"])
    print(crosstab.to_string())
    print("\nIf cluster 0 and cluster 1 concentrate on clearly different blocks, the split")
    print("reflects benign 'partial relevance to different profile blocks' heterogeneity.")
    print("If both clusters spread across the same blocks similarly, the split is not")
    print("block-driven and may reflect something else (e.g. writing-style drift).")

    cols = ["item_id", "_cluster", "_best_block_name", "_best_block_score", "target_role", "item_text"]
    cols = [c for c in cols if c in wa_df.columns]
    wa_df_sorted = wa_df.sort_values(["_cluster", "_best_block_idx"])
    wa_df_sorted[cols].to_csv("diagnostic_C3_weakly_aligned_clusters.csv", index=False)

    cols = ["item_id", "_cluster", "_best_block_name", "_best_block_score", "target_role", "item_text"]
    cols = [c for c in cols if c in wa_df.columns]
    wa_df_sorted = wa_df.sort_values(["_cluster", "_best_block_idx"])
    wa_df_sorted[cols].to_csv("diagnostic_C3_weakly_aligned_clusters.csv", index=False)

    for cl in [0, 1]:
        sub = wa_df_sorted[wa_df_sorted["_cluster"] == cl]
        print("\n" + "*" * 70)
        print(f"  Cluster {cl} ({len(sub)} items)")
        print("*" * 70)
        for i, row in sub.iterrows():
            print(f"\n  item_id={row['item_id']}  best_block={row['_best_block_name']}"
                  f"  score={row['_best_block_score']:.4f}  target_role={row.get('target_role', 'n/a')}")
            print(f"  text: {row['item_text']}")

    print("\n OK, Saved diagnostic_C3_weakly_aligned_clusters.csv")


if __name__ == "__main__":
    main()
