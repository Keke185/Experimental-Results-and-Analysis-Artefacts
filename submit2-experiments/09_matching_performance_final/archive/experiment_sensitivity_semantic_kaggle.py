import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items-final/dataset_items_final.csv"
MODEL_CONFIGS = {
    "MiniLM": "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2",
    "BGE-large-v1.5": "/kaggle/input/models/levantaokkz/bge-large-en-v1.5/transformers/default/1/bge-large-en-v1.5",
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


def strip_explain_and_normalize_periods(text):
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


def cv_classifier_macro_f1(X, gold_labels, skf):
    gold_arr = np.array(gold_labels)
    pred = np.empty(len(gold_arr), dtype=object)
    for train_idx, test_idx in skf.split(X, gold_arr):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(X[train_idx]), gold_arr[train_idx])
        pred[test_idx] = clf.predict(scaler.transform(X[test_idx]))
    return f1_score(gold_labels, pred, labels=LABELS, average="macro", zero_division=0)


def tfidf_sims_leakfixed(texts, gold_labels, true_idx_placeholder, skf):

    n = len(texts)
    sims_full = np.zeros((n, len(CAPABILITY_CHUNKS)))
    gold_arr = np.array(gold_labels)
    for train_idx, test_idx in skf.split(np.zeros(n), gold_arr):
        vec = TfidfVectorizer(stop_words="english")
        vec.fit(CAPABILITY_CHUNKS + [texts[i] for i in train_idx])
        cap_tfidf = vec.transform(CAPABILITY_CHUNKS)
        test_tfidf = vec.transform([texts[i] for i in test_idx])
        sims_full[test_idx] = (test_tfidf @ cap_tfidf.T).toarray()
    return sims_full


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    gold_labels = df["gold_label"].tolist()
    print(f"OK, Loaded {len(df)} items")

    texts_before = df["item_text"].tolist()
    texts_after = [strip_explain_and_normalize_periods(t) for t in texts_before]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results = []
    for model_name, model_path in MODEL_CONFIGS.items():
        print(f"\n Loading {model_name}")
        model = SentenceTransformer(model_path, local_files_only=True)
        cap_vecs = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True, show_progress_bar=True)

        print(f" Encoding BEFORE texts with {model_name}")
        item_vecs_before = model.encode(texts_before, normalize_embeddings=True, show_progress_bar=True)
        sims_before = item_vecs_before @ cap_vecs.T
        f1_before = cv_classifier_macro_f1(build_features(sims_before), gold_labels, skf)

        print(f" Encoding AFTER (Explain:/period-normalized) texts with {model_name}")
        item_vecs_after = model.encode(texts_after, normalize_embeddings=True, show_progress_bar=True)
        sims_after = item_vecs_after @ cap_vecs.T
        f1_after = cv_classifier_macro_f1(build_features(sims_after), gold_labels, skf)

        print(f"[{model_name}] BEFORE macro-F1={f1_before:.4f}  AFTER macro-F1={f1_after:.4f}  "
              f"delta={f1_after - f1_before:+.4f}")
        results.append((model_name, f1_before, f1_after, f1_after - f1_before))

        if model_name == "MiniLM":
            minilm_sims_before, minilm_sims_after = sims_before, sims_after

    print("\n Building leak-fixed TF-IDF sims for Hybrid (BEFORE)")
    kw_sims_before = tfidf_sims_leakfixed(texts_before, gold_labels, None, skf)
    print("Building leak-fixed TF-IDF sims for Hybrid (AFTER)")
    kw_sims_after = tfidf_sims_leakfixed(texts_after, gold_labels, None, skf)

    hybrid_X_before = np.hstack([build_features(minilm_sims_before), build_features(kw_sims_before)])
    hybrid_X_after = np.hstack([build_features(minilm_sims_after), build_features(kw_sims_after)])
    f1_hybrid_before = cv_classifier_macro_f1(hybrid_X_before, gold_labels, skf)
    f1_hybrid_after = cv_classifier_macro_f1(hybrid_X_after, gold_labels, skf)
    print(f"\nHybrid, BEFORE macro-F1={f1_hybrid_before:.4f}  AFTER macro-F1={f1_hybrid_after:.4f}  "
          f"delta={f1_hybrid_after - f1_hybrid_before:+.4f}")
    results.append(("Hybrid", f1_hybrid_before, f1_hybrid_after, f1_hybrid_after - f1_hybrid_before))

    print("\n" + "*" * 80)
    print("  SUMMARY: sensitivity to Explain:/period-normalization, all representations")
    print("*" * 80)
    print(f"{'Representation':<18}{'BEFORE':>10}{'AFTER':>10}{'delta':>10}")
    for name, before, after, delta in results:
        print(f"{name:<18}{before:>10.4f}{after:>10.4f}{delta:>+10.4f}")

    out = pd.DataFrame(results, columns=["representation", "macro_f1_before", "macro_f1_after", "delta"])
    out.to_csv("sensitivity_semantic_results.csv", index=False)
    print("\n OK, Saved sensitivity_semantic_results.csv")


if __name__ == "__main__":
    main()
