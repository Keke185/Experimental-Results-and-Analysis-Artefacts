import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items-final/dataset_items_final.csv"
MODEL_CONFIGS = {
    "MiniLM": "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2",
    "BGE-large-v1.5": "/kaggle/input/models/levantaokkz/bge-large-en-v1.5/transformers/default/1/bge-large-en-v1.5",
}

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]
LABEL_TO_CODE = {lbl: i for i, lbl in enumerate(LABELS)}

CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]

BLOCK_NAMES = [
    "block1_EnterpriseVC_Signaling", "block2_ConfMgmt_BusinessControl",
    "block3_MCU_MediaEngine", "block4_Endpoints_RoomSystems",
    "block5_4K_Codecs_MediaProcessing", "block6_CPaaS_ConversationalAI",
]

VARIANTS = {
    "6 raw blocks only":           [0, 1, 2, 3, 4, 5],
    "6 blocks + mean_sim only":    [0, 1, 2, 3, 4, 5, 6],
    "6 blocks + std_sim + margin": [0, 1, 2, 3, 4, 5, 7, 8],
    "full 9-dim (baseline)":       [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "full 9-dim minus block2":     [0, 2, 3, 4, 5, 6, 7, 8],
    "full 9-dim minus std_sim":    [0, 1, 2, 3, 4, 5, 6, 8],
    "full 9-dim minus margin":     [0, 1, 2, 3, 4, 5, 6, 7],
    "full 9-dim minus mean_sim":   [0, 1, 2, 3, 4, 5, 7, 8],
}


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


def run_ablation_for_model(model_name, sims, gold_labels, true_code):
    n = len(true_code)
    mean_ = sims.mean(axis=1, keepdims=True)
    std_ = sims.std(axis=1, keepdims=True)
    margin_ = sims.max(axis=1, keepdims=True) - mean_
    X_full = np.hstack([sims, mean_, std_, margin_])  # 0-5 raw, 6 mean, 7 std, 8 margin

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    gold_arr = np.array(gold_labels)

    print(f"\n    {model_name} ablation: 10-fold CV macro-F1 by feature subset    ")
    results = []
    for variant_name, cols in VARIANTS.items():
        fold_f1s = []
        for train_idx, test_idx in skf.split(X_full, gold_arr):
            X_train = X_full[train_idx][:, cols]
            X_test = X_full[test_idx][:, cols]
            scaler = StandardScaler().fit(X_train)
            clf = LogisticRegression(class_weight="balanced", max_iter=2000)
            clf.fit(scaler.transform(X_train), gold_arr[train_idx])
            pred_labels_test = clf.predict(scaler.transform(X_test))
            pred_code_test = np.array([LABEL_TO_CODE[l] for l in pred_labels_test])
            fold_f1s.append(fast_macro_f1(pred_code_test, true_code[test_idx]))
        fold_arr = np.array(fold_f1s)
        results.append((model_name, variant_name, len(cols), fold_arr.mean(), fold_arr.std(ddof=1)))
        print(f"  {variant_name:<32} n_features={len(cols):<3} "
              f"mean macro-F1={fold_arr.mean():.4f}  std={fold_arr.std(ddof=1):.4f}")

    baseline_f1 = [r[3] for r in results if r[1] == "full 9-dim (baseline)"][0]
    print(f"\n    {model_name} deltas vs full 9-dim baseline ({baseline_f1:.4f})    ")
    for _, name, nfeat, mean_f1, std_f1 in results:
        if name == "full 9-dim (baseline)":
            continue
        print(f"  {name:<32} delta = {mean_f1 - baseline_f1:+.4f}")

    return results


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items.")

    gold_labels = df["gold_label"].tolist()
    true_code = np.array([LABEL_TO_CODE[l] for l in gold_labels])

    all_results = []
    for model_name, model_path in MODEL_CONFIGS.items():
        print(f"\n Loading {model_name}")
        model = SentenceTransformer(model_path, local_files_only=True)
        cap_vectors = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True, show_progress_bar=True)
        item_vectors = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
        sims = item_vectors @ cap_vectors.T
        results = run_ablation_for_model(model_name, sims, gold_labels, true_code)
        all_results.extend(results)

    out = pd.DataFrame(all_results, columns=["model", "variant", "n_features", "mean_macro_f1", "std_macro_f1"])
    out.to_csv("ablation_semantic_results.csv", index=False)
    print("\nOK, Saved ablation_semantic_results.csv")


if __name__ == "__main__":
    main()
