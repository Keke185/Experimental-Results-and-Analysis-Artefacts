import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items-final/dataset_items_final.csv"
MODEL_CONFIGS = {
    "MiniLM": "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2",
    "BGE-large-v1.5": "/kaggle/input/models/levantaokkz/bge-large-en-v1.5/transformers/default/1/bge-large-en-v1.5",
}

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
LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]

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


def subtype_of(item_id):
    if item_id in WA_D_IDS:
        return "WA-D"
    if item_id in WA_F_IDS:
        return "WA-F"
    return "n/a"


def report_subtype_breakdown(model_name, ids, gold, pred, subtype):
    wa_mask = gold == "Weakly Aligned"
    correct = pred == gold

    print(f"\n{'*'*70}")
    print(f"  MODEL: {model_name} -- WA-F vs WA-D breakdown (leak-fixed: per-fold scaler)")
    print(f"{'*'*70}")
    print(f"n Weakly Aligned = {wa_mask.sum()} "
          f"(WA-F={np.sum((subtype=='WA-F') & wa_mask)}, "
          f"WA-D={np.sum((subtype=='WA-D') & wa_mask)})")

    for st in ["WA-F", "WA-D"]:
        m = wa_mask & (subtype == st)
        n = m.sum()
        acc = correct[m].mean() if n else float("nan")
        print(f"\n--- {st} ---")
        print(f"n={n}, recall (correctly classified as Weakly Aligned)={acc:.4f}")
        counts = pd.Series(pred[m]).value_counts()
        for label, cnt in counts.items():
            print(f"  -> predicted '{label}': {cnt} ({cnt/n*100:.1f}%)")

    detail = pd.DataFrame({
        "item_id": ids[wa_mask],
        "subtype": subtype[wa_mask],
        "gold_label": gold[wa_mask],
        "predicted": pred[wa_mask],
        "correct": correct[wa_mask],
    }).sort_values(["subtype", "correct", "item_id"])
    fname = f"wa_subtype_breakdown_v2_{model_name.replace(' ', '_').replace('-', '_')}.csv"
    detail.to_csv(fname, index=False)
    print(f"\nOK, Saved {fname}")


def cv_predict_leakfree(X, gold, skf):
    n = len(gold)
    pred = np.empty(n, dtype=object)
    gold_arr = np.asarray(gold)
    for train_idx, test_idx in skf.split(X, gold_arr):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(X[train_idx]), gold_arr[train_idx])
        pred[test_idx] = clf.predict(scaler.transform(X[test_idx]))
    return pred


def run_semantic(model_name, model_path, df, ids, gold, subtype, skf):
    print(f"\n Loading {model_name}")
    model = SentenceTransformer(model_path, local_files_only=True)
    cap_vecs = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True, show_progress_bar=True)
    item_vecs = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    sims = item_vecs @ cap_vecs.T
    X = build_features(sims)

    pred = cv_predict_leakfree(X, gold, skf)

    print(f"\n    Overall classifier report ({model_name}, 10-fold CV, leak-fixed)    ")
    print(classification_report(gold, pred, digits=4))
    print("Confusion matrix (labels=[Aligned, Weakly Aligned, Mismatched]):")
    print(confusion_matrix(gold, pred, labels=LABELS))

    report_subtype_breakdown(model_name, ids, gold, pred, subtype)
    return sims


def run_hybrid(df, ids, gold, subtype, minilm_sims, skf):
    texts = df["item_text"].tolist()
    n = len(df)
    gold_arr = np.asarray(gold)
    sem_X_full = build_features(minilm_sims)

    pred = np.empty(n, dtype=object)
    for train_idx, test_idx in skf.split(np.zeros(n), gold_arr):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        vec = TfidfVectorizer(stop_words="english")
        vec.fit(CAPABILITY_CHUNKS + train_texts)
        cap_tfidf = vec.transform(CAPABILITY_CHUNKS)
        kw_train_sims = (vec.transform(train_texts) @ cap_tfidf.T).toarray()
        kw_test_sims = (vec.transform(test_texts) @ cap_tfidf.T).toarray()
        kw_X_train = build_features(kw_train_sims)
        kw_X_test = build_features(kw_test_sims)

        X_train = np.hstack([sem_X_full[train_idx], kw_X_train])
        X_test = np.hstack([sem_X_full[test_idx], kw_X_test])
        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(X_train), gold_arr[train_idx])
        pred[test_idx] = clf.predict(scaler.transform(X_test))

    print(f"\n    Overall classifier report (Hybrid concat features, 10-fold CV, leak-fixed)   ")
    print(classification_report(gold, pred, digits=4))
    print("Confusion matrix (labels=[Aligned, Weakly Aligned, Mismatched]):")
    print(confusion_matrix(gold, pred, labels=LABELS))

    report_subtype_breakdown("Hybrid", ids, gold, pred, subtype)


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items")

    ids = df["item_id"].to_numpy()
    gold = df["gold_label"].to_numpy()
    subtype = np.array([subtype_of(i) for i in ids])
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    minilm_sims = None
    for model_name, model_path in MODEL_CONFIGS.items():
        sims = run_semantic(model_name, model_path, df, ids, gold, subtype, skf)
        if model_name == "MiniLM":
            minilm_sims = sims

    run_hybrid(df, ids, gold, subtype, minilm_sims, skf)


if __name__ == "__main__":
    main()
