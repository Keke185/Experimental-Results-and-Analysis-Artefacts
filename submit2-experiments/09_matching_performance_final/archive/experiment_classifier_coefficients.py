"""
Coefficient / feature-importance extraction for the classifier decision-logic
experiment (experiment_classifier_decision_logic.py).

Goal: confirm WHICH features the logistic regression actually relies on.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
MODEL_CONFIGS = {
    "MiniLM": "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2",
    "BGE-large-v1.5": "/kaggle/input/models/levantaokkz/bge-large-en-v1.5/transformers/default/1/bge-large-en-v1.5",
}

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]
BLOCK_NAMES = [
    "block1_EnterpriseVC_Signaling",
    "block2_ConfMgmt_BusinessControl",
    "block3_MCU_MediaEngine",
    "block4_Endpoints_RoomSystems",
    "block5_4K_Codecs_MediaProcessing",
    "block6_CPaaS_ConversationalAI",
]
FEATURE_NAMES = BLOCK_NAMES + ["mean_sim", "std_sim", "margin_max_minus_mean"]

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


def print_coef_table(title, classes, coefs):
    print(f"\n{title}")
    header = f"{'feature':<32}" + "".join(f"{c:>18}" for c in classes)
    print(header)
    print("-" * 50)
    for j, fname in enumerate(FEATURE_NAMES):
        row = f"{fname:<32}" + "".join(f"{coefs[i, j]:>18.3f}" for i in range(len(classes)))
        print(row)


def print_ranked_importance(title, classes, coefs):
    print(f"\n{title} (ranked by |coefficient|, per class)")
    for i, cls in enumerate(classes):
        order = np.argsort(-np.abs(coefs[i]))
        print(f"\n  class = {cls}:")
        for rank, j in enumerate(order, start=1):
            print(f"    {rank}. {FEATURE_NAMES[j]:<32} coef={coefs[i, j]:+.3f}")


def run_for_model(model_name, model_path, df):
    print("\n" + "*" * 70)
    print(f"  MODEL: {model_name}")
    print("*" * 70)

    print(f" Loading {model_path}")
    model = SentenceTransformer(model_path, local_files_only=True)

    print(" Encoding capability chunks and dataset items")
    cap_vectors = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True, show_progress_bar=True)
    item_vectors = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)

    sims = item_vectors @ cap_vectors.T
    gold = df["gold_label"].tolist()
    X = build_features(sims)

    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X_scaled, gold)
    classes = list(clf.classes_)
    print_coef_table(f"[{model_name}] Reference fit (all 200 items) -- standardized coefficients", classes, clf.coef_)
    print_ranked_importance(f"[{model_name}] Reference fit", classes, clf.coef_)

    # Stability check
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_coefs = []
    gold_arr = np.array(gold)
    for train_idx, _ in skf.split(X, gold_arr):
        X_train = X[train_idx]
        y_train = gold_arr[train_idx]
        sc = StandardScaler().fit(X_train)
        X_train_scaled = sc.transform(X_train)
        c = LogisticRegression(class_weight="balanced", max_iter=2000)
        c.fit(X_train_scaled, y_train)

        aligned = np.zeros_like(c.coef_)
        for i, cls in enumerate(classes):
            fold_class_idx = list(c.classes_).index(cls)
            aligned[i] = c.coef_[fold_class_idx]
        fold_coefs.append(aligned)
    fold_coefs = np.stack(fold_coefs, axis=0)
    mean_coefs = fold_coefs.mean(axis=0)
    std_coefs = fold_coefs.std(axis=0)

    print(f"\n[{model_name}] Stability check: mean coefficient across 10 CV-fold refits")
    print_coef_table(f"[{model_name}] Mean across 10 folds", classes, mean_coefs)
    print(f"\n[{model_name}] Stability check: std of coefficient across 10 CV-fold refits (lower = more stable)")
    print_coef_table(f"[{model_name}] Std across 10 folds", classes, std_coefs)

    # Save as a CSV file
    rows = []
    for i, cls in enumerate(classes):
        for j, fname in enumerate(FEATURE_NAMES):
            rows.append({
                "model": model_name,
                "class": cls,
                "feature": fname,
                "coef_full_fit": clf.coef_[i, j],
                "coef_cv_mean": mean_coefs[i, j],
                "coef_cv_std": std_coefs[i, j],
            })
    out = pd.DataFrame(rows)
    fname_out = f"experiment_coefficients_{model_name.replace(' ', '_').replace('-', '_')}.csv"
    out.to_csv(fname_out, index=False)
    print(f"\n OK, Saved {fname_out}")


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items")

    for model_name, model_path in MODEL_CONFIGS.items():
        run_for_model(model_name, model_path, df)

    print("\n" + "*" * 70)
    print("Look specifically at where 'margin_max_minus_mean' and 'std_sim' rank")
    print("for the Mismatched class (BGE) and Weakly Aligned class (MiniLM) ,")
    print("that's the direct test of the anisotropy/peakiness hypothesis.")
    print("Also check the CV-mean vs full-fit coefficients agree in sign ,")
    print("if a coefficient flips sign across folds, don't trust that feature's story")


if __name__ == "__main__":
    main()
