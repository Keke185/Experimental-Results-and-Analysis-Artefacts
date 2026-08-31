"""
Diagnosis H2: Examines whether length confusion contaminates the 9-feature classifier.
Diagnosis H proves that word count itself has predictive power, mainly distinguishing weak
alignment from mismatch, but the 9 similarity features are not directly input into word count.
This script uses two tests to determine whether there is a correlation between word count and
features: 1. Calculate the Pearson correlation coefficient between word count and the 9 features;
2. Residual test: Regress each feature against word count and take the residuals, then rerun the 10-fold CV
based on the residual features. If the residual set F1 is close to the original value, it indicates that the
model depends on the semantic signal and length confusion has no interference; a significant decrease in F1
indicates that the results are affected by word count confusion. The script is based on MiniLM and can be
replaced by BGE
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
MODEL_PATH = "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2"

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]
RANDOM_STATE = 42

CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]

FEATURE_NAMES = [f"sim_block_{i + 1}" for i in range(6)] + ["mean", "std", "margin"]


def build_features(sims):
    mean_sim = sims.mean(axis=1, keepdims=True)
    std_sim = sims.std(axis=1, keepdims=True)
    max_sim = sims.max(axis=1, keepdims=True)
    margin = max_sim - mean_sim
    return np.hstack([sims, mean_sim, std_sim, margin])


def cv_macro_f1(X, gold, tag):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, multi_class="multinomial"),
    )
    preds = cross_val_predict(clf, X, gold, cv=skf)
    f1 = f1_score(gold, preds, labels=LABELS, average="macro", zero_division=0)
    acc = accuracy_score(gold, preds)
    print(f"  [{tag}] macro-F1={f1:.4f}  accuracy={acc:.4f}")
    return f1, acc


def main():
    print("*" * 70)
    print("  Diagnostic H2: Word-count confound check on the real 9-feature classifier")
    print("*" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items.")

    word_count = df["item_text"].apply(lambda t: len(str(t).split())).to_numpy(dtype=float).reshape(-1, 1)
    gold = df["gold_label"].tolist()

    print(f"\nLoading {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH, local_files_only=True) if MODEL_PATH.startswith(
        "/kaggle") else SentenceTransformer(MODEL_PATH)

    print(" Encoding capability chunks and dataset items")
    cap_vectors = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True, show_progress_bar=True)
    item_vectors = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    sims = item_vectors @ cap_vectors.T
    X = build_features(sims)

    # Correlation between word frequency and each of the nine features
    print("\n" + "*" * 70)
    print("  Check 1: Pearson correlation")
    print("*" * 70)
    wc_flat = word_count.ravel()
    print(f"{'feature':<14}{'pearson r':>12}")
    print("-" * 50)
    corrs = {}
    for i, name in enumerate(FEATURE_NAMES):
        r = np.corrcoef(wc_flat, X[:, i])[0, 1]
        corrs[name] = r
        flag = "  <-- |r|>=0.3" if abs(r) >= 0.3 else ""
        print(f"{name:<14}{r:>12.4f}{flag}")

    # residualization test
    print("\n" + "*" * 70)
    print("  Check 2: residualization test (the decisive one)")
    print("=" * 70)

    print("\n a.Classifier on ORIGINAL 9 features (should reproduce the known ~0.85 for MiniLM):")
    f1_orig, acc_orig = cv_macro_f1(X, gold, "original 9 features")

    X_resid = np.zeros_like(X)
    for i, name in enumerate(FEATURE_NAMES):
        reg = LinearRegression().fit(word_count, X[:, i])
        pred = reg.predict(word_count)
        X_resid[:, i] = X[:, i] - pred  # residual = part NOT explained by word_count

    print("\n b.Classifier on RESIDUALIZED 9 features (word_count's linear effect removed from each):")
    f1_resid, acc_resid = cv_macro_f1(X_resid, gold, "residualized 9 features")

    drop = f1_orig - f1_resid
    out = pd.DataFrame({"feature": FEATURE_NAMES, "pearson_r_with_wordcount": [corrs[n] for n in FEATURE_NAMES]})
    out.loc[len(out)] = ["__cv_macro_f1_original__", f1_orig]
    out.loc[len(out)] = ["__cv_macro_f1_residualized__", f1_resid]
    out.to_csv("diagnostic_H2_wordcount_confound_results.csv", index=False)
    print("\nOK, Saved diagnostic_H2_wordcount_confound_results.csv")


if __name__ == "__main__":
    main()
