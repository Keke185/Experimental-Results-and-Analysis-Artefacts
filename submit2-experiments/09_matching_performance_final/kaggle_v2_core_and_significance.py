import re
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support,

)
from scipy import stats
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items-final-v2/dataset_items_final_v2.csv"
SEMANTIC_MODELS = {
    "MiniLM": "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2",
    "BGE-large-v1.5": "/kaggle/input/models/levantaokkz/bge-large-en-v1.5/transformers/default/1/bge-large-en-v1.5",
}

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]
LABEL_TO_CODE = {lbl: i for i, lbl in enumerate(LABELS)}
CODE_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_CODE.items()}

WA_D_IDS = {
    "Q_104", "Q_113", "Q_114", "Q_115", "Q_116", "Q_119", "Q_120", "Q_121", "Q_122",
    "Q_123", "Q_124", "Q_125", "Q_126", "Q_127", "Q_128", "Q_130", "Q_136", "Q_138",
    "Q_139", "Q_143", "Q_146", "Q_147", "Q_148", "Q_149", "Q_150",
}
WA_F_IDS = {
    "Q_101", "Q_102", "Q_103", "Q_105", "Q_106", "Q_107", "Q_108", "Q_109", "Q_110",
    "Q_111", "Q_112", "Q_117", "Q_118", "Q_129", "Q_131", "Q_132", "Q_133", "Q_134",
    "Q_135", "Q_137", "Q_140", "Q_141", "Q_142", "Q_144", "Q_145",
}

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
FEATURE_NAMES = BLOCK_NAMES + ["mean_sim", "std_sim", "margin"]

N_REPEATS = 10
th_candidates = np.arange(0.05, 0.96, 0.03)
alphas = np.round(np.arange(0.0, 1.01, 0.1), 2)


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


def code_classify(scores, weak_th, aligned_th):
    return np.where(scores >= aligned_th, 0, np.where(scores >= weak_th, 1, 2))


def best_thresholds_on(scores, true_code, cands):
    best = None
    for weak_th in cands:
        for aligned_th in cands:
            if aligned_th <= weak_th:
                continue
            pred_code = code_classify(scores, weak_th, aligned_th)
            f1 = fast_macro_f1(pred_code, true_code)
            if best is None or f1 > best[0]:
                best = (f1, weak_th, aligned_th)
    return best


def build_features(sims):
    mean_sim = sims.mean(axis=1, keepdims=True)
    std_sim = sims.std(axis=1, keepdims=True)
    max_sim = sims.max(axis=1, keepdims=True)
    margin = max_sim - mean_sim
    return np.hstack([sims, mean_sim, std_sim, margin])


def tfidf_sims_single_fold(texts, train_idx, test_idx):
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    vec = TfidfVectorizer(stop_words="english")
    vec.fit(CAPABILITY_CHUNKS + train_texts)
    cap_tfidf = vec.transform(CAPABILITY_CHUNKS)
    train_sims = (vec.transform(train_texts) @ cap_tfidf.T).toarray()
    test_sims = (vec.transform(test_texts) @ cap_tfidf.T).toarray()
    return train_sims, test_sims


def report_confusion(name, gold, preds):
    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, labels=LABELS, average="macro", zero_division=0)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        gold, preds, labels=LABELS, zero_division=0)
    cm = confusion_matrix(gold, preds, labels=LABELS)
    print(f"\n[{name}] accuracy={acc:.4f}  macro-F1={f1:.4f}")
    for lbl, p, r, f, s in zip(LABELS, precision, recall, per_class_f1, support):
        print(f"  {lbl:15s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  n={s}")
    print(f"  confusion matrix (rows=gold, cols=pred, labels={LABELS}):\n{cm}")
    mismatched_idx, aligned_idx = LABELS.index("Mismatched"), LABELS.index("Aligned")
    print(f"  Mismatched -> Aligned misclassifications: {cm[mismatched_idx, aligned_idx]} / {cm[mismatched_idx].sum()}")
    return f1, cm


def wa_subtype_report(name, ids, gold, pred):
    subtype = np.array(["WA-D" if i in WA_D_IDS else ("WA-F" if i in WA_F_IDS else "n/a") for i in ids])
    wa_mask = np.array(gold) == "Weakly Aligned"
    correct = np.array(pred) == np.array(gold)
    print(f"\n[{name}] WA-F/WA-D subtype breakdown:")
    for st in ["WA-F", "WA-D"]:
        m = wa_mask & (subtype == st)
        n = m.sum()
        acc = correct[m].mean() if n else float("nan")
        print(f"  {st}: n={n}, recall={acc:.4f}")
        mis_counts = pd.Series(np.array(pred)[m]).value_counts()
        for label, cnt in mis_counts.items():
            print(f"    -> predicted '{label}': {cnt} ({cnt / n * 100:.1f}%)")


def run_one_cv(name, X_or_texts, gold_labels, true_code, skf, is_tfidf=False, sims_precomp=None):
    n = len(gold_labels)
    th_oof = np.full(n, -1, dtype=int)
    clf_oof = np.full(n, -1, dtype=int)
    th_fold_f1, clf_fold_f1 = [], []

    for train_idx, test_idx in skf.split(np.zeros(n), true_code):
        if is_tfidf:
            train_sims, test_sims = tfidf_sims_single_fold(X_or_texts, train_idx, test_idx)
        else:
            train_sims, test_sims = sims_precomp[train_idx], sims_precomp[test_idx]

        train_max = train_sims.max(axis=1)
        test_max = test_sims.max(axis=1)
        _, weak_th, aligned_th = best_thresholds_on(train_max, true_code[train_idx], th_candidates)
        pred_code_test = code_classify(test_max, weak_th, aligned_th)
        th_oof[test_idx] = pred_code_test
        th_fold_f1.append(fast_macro_f1(pred_code_test, true_code[test_idx]))

        X_train = build_features(train_sims)
        X_test = build_features(test_sims)
        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(X_train), [gold_labels[i] for i in train_idx])
        pred_labels_test = clf.predict(scaler.transform(X_test))
        pred_code_test_clf = np.array([LABEL_TO_CODE[l] for l in pred_labels_test])
        clf_oof[test_idx] = pred_code_test_clf
        clf_fold_f1.append(fast_macro_f1(pred_code_test_clf, true_code[test_idx]))

    th_labels = [CODE_TO_LABEL[c] for c in th_oof]
    clf_labels = [CODE_TO_LABEL[c] for c in clf_oof]
    print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")

    print(f"  threshold per-fold: mean={np.mean(th_fold_f1):.4f} std={np.std(th_fold_f1, ddof=1):.4f}")
    print(f"  classifier per-fold: mean={np.mean(clf_fold_f1):.4f} std={np.std(clf_fold_f1, ddof=1):.4f}")
    return th_labels, clf_labels, th_fold_f1, clf_fold_f1


def run_hybrid_cv(name, texts, minilm_sims, gold_labels, true_code, skf):
    # Hybrid: threshold logic uses alpha-fusion
    n = len(gold_labels)
    gold_arr = np.array(gold_labels)
    sem_max = minilm_sims.max(axis=1)
    sem_X_full = build_features(minilm_sims)

    th_oof = np.full(n, -1, dtype=int)
    clf_oof = np.full(n, -1, dtype=int)
    th_fold_f1, clf_fold_f1 = [], []

    for train_idx, test_idx in skf.split(np.zeros(n), true_code):
        train_sims, test_sims = tfidf_sims_single_fold(texts, train_idx, test_idx)
        kw_train_max, kw_test_max = train_sims.max(axis=1), test_sims.max(axis=1)
        sem_train_max, sem_test_max = sem_max[train_idx], sem_max[test_idx]

        sem_lo, sem_hi = sem_train_max.min(), sem_train_max.max()
        kw_lo, kw_hi = kw_train_max.min(), kw_train_max.max()
        sem_norm_tr = (sem_train_max - sem_lo) / max(sem_hi - sem_lo, 1e-12)
        kw_norm_tr = (kw_train_max - kw_lo) / max(kw_hi - kw_lo, 1e-12)

        best_overall = None
        for alpha in alphas:
            hybrid_tr = alpha * sem_norm_tr + (1 - alpha) * kw_norm_tr
            f1, weak_th, aligned_th = best_thresholds_on(hybrid_tr, true_code[train_idx], th_candidates)
            if best_overall is None or f1 > best_overall[0]:
                best_overall = (f1, alpha, weak_th, aligned_th)
        _, alpha, weak_th, aligned_th = best_overall

        sem_te = (sem_test_max - sem_lo) / max(sem_hi - sem_lo, 1e-12)
        kw_te = (kw_test_max - kw_lo) / max(kw_hi - kw_lo, 1e-12)
        hybrid_te = alpha * sem_te + (1 - alpha) * kw_te
        pred_code_test = code_classify(hybrid_te, weak_th, aligned_th)
        th_oof[test_idx] = pred_code_test
        th_fold_f1.append(fast_macro_f1(pred_code_test, true_code[test_idx]))

        X_train = np.hstack([sem_X_full[train_idx], build_features(train_sims)])
        X_test = np.hstack([sem_X_full[test_idx], build_features(test_sims)])
        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(X_train), gold_arr[train_idx])
        pred_labels_test = clf.predict(scaler.transform(X_test))
        pred_code_test_clf = np.array([LABEL_TO_CODE[l] for l in pred_labels_test])
        clf_oof[test_idx] = pred_code_test_clf
        clf_fold_f1.append(fast_macro_f1(pred_code_test_clf, true_code[test_idx]))

    th_labels = [CODE_TO_LABEL[c] for c in th_oof]
    clf_labels = [CODE_TO_LABEL[c] for c in clf_oof]
    print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
    report_confusion(f"{name} -- threshold (alpha-fusion)", gold_labels, th_labels)
    report_confusion(f"{name} -- classifier (18-dim concat)", gold_labels, clf_labels)
    print(f"  threshold per-fold: mean={np.mean(th_fold_f1):.4f} std={np.std(th_fold_f1, ddof=1):.4f}")
    print(f"  classifier per-fold: mean={np.mean(clf_fold_f1):.4f} std={np.std(clf_fold_f1, ddof=1):.4f}")
    return th_labels, clf_labels, th_fold_f1, clf_fold_f1


def repeated_cv_hybrid_threshold_f1(texts, minilm_sims, gold_labels, true_code, n_repeats=N_REPEATS, n_splits=10,
                                    seed_offset=0):
    sem_max = minilm_sims.max(axis=1)
    all_f1 = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1000 + seed_offset + rep)
        for train_idx, test_idx in skf.split(np.zeros(len(true_code)), true_code):
            train_sims, test_sims = tfidf_sims_single_fold(texts, train_idx, test_idx)
            kw_train_max, kw_test_max = train_sims.max(axis=1), test_sims.max(axis=1)
            sem_train_max, sem_test_max = sem_max[train_idx], sem_max[test_idx]
            sem_lo, sem_hi = sem_train_max.min(), sem_train_max.max()
            kw_lo, kw_hi = kw_train_max.min(), kw_train_max.max()
            sem_norm_tr = (sem_train_max - sem_lo) / max(sem_hi - sem_lo, 1e-12)
            kw_norm_tr = (kw_train_max - kw_lo) / max(kw_hi - kw_lo, 1e-12)
            best_overall = None
            for alpha in alphas:
                hybrid_tr = alpha * sem_norm_tr + (1 - alpha) * kw_norm_tr
                f1, weak_th, aligned_th = best_thresholds_on(hybrid_tr, true_code[train_idx], th_candidates)
                if best_overall is None or f1 > best_overall[0]:
                    best_overall = (f1, alpha, weak_th, aligned_th)
            _, alpha, weak_th, aligned_th = best_overall
            sem_te = (sem_test_max - sem_lo) / max(sem_hi - sem_lo, 1e-12)
            kw_te = (kw_test_max - kw_lo) / max(kw_hi - kw_lo, 1e-12)
            hybrid_te = alpha * sem_te + (1 - alpha) * kw_te
            pred_code_test = code_classify(hybrid_te, weak_th, aligned_th)
            all_f1.append(fast_macro_f1(pred_code_test, true_code[test_idx]))
    return np.array(all_f1)


def repeated_cv_classifier_f1(build_X_fn, gold_labels, n_repeats=N_REPEATS, n_splits=10, seed_offset=0):
    gold_arr = np.array(gold_labels)
    all_f1 = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1000 + seed_offset + rep)
        for train_idx, test_idx in skf.split(np.zeros(len(gold_arr)), gold_arr):
            X_train, X_test = build_X_fn(train_idx, test_idx)
            scaler = StandardScaler().fit(X_train)
            clf = LogisticRegression(class_weight="balanced", max_iter=2000)
            clf.fit(scaler.transform(X_train), gold_arr[train_idx])
            pred = clf.predict(scaler.transform(X_test))
            pred_code = np.array([LABEL_TO_CODE[l] for l in pred])
            true_code_test = np.array([LABEL_TO_CODE[l] for l in gold_arr[test_idx]])
            all_f1.append(fast_macro_f1(pred_code, true_code_test))
    return np.array(all_f1)


def repeated_cv_threshold_f1(build_scores_fn, gold_labels, true_code, n_repeats=N_REPEATS, n_splits=10, seed_offset=0):
    all_f1 = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1000 + seed_offset + rep)
        for train_idx, test_idx in skf.split(np.zeros(len(true_code)), true_code):
            train_scores, test_scores = build_scores_fn(train_idx, test_idx)
            _, weak_th, aligned_th = best_thresholds_on(train_scores, true_code[train_idx], th_candidates)
            pred_code_test = code_classify(test_scores, weak_th, aligned_th)
            all_f1.append(fast_macro_f1(pred_code_test, true_code[test_idx]))
    return np.array(all_f1)


def nadeau_bengio_paired_test(diffs, n_train, n_test, n_splits=10):
    diffs = np.asarray(diffs)
    n = len(diffs)
    mean_d = diffs.mean()
    var_d = diffs.var(ddof=1)
    corrected_var = var_d * (1.0 / n + n_test / n_train)
    t_stat = mean_d / np.sqrt(corrected_var)
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    t_crit = stats.t.ppf(0.975, df=df)
    ci_low = mean_d - t_crit * np.sqrt(corrected_var)
    ci_high = mean_d + t_crit * np.sqrt(corrected_var)
    return t_stat, p_value, ci_low, ci_high, mean_d


def main():
    t0 = time.time()
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} rows from {DATASET_PATH}")
    assert df["item_text"].str.contains("Explain:").sum() == 0, "v2 dataset should have no 'Explain:' left!"

    texts = df["item_text"].tolist()
    gold_labels = df["gold_label"].tolist()
    ids = df["item_id"].tolist()
    true_code = np.array([LABEL_TO_CODE[l] for l in gold_labels])
    n = len(df)
    skf_main = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    semantic_sims = {}
    for model_name, model_path in SEMANTIC_MODELS.items():
        print(f"\n Loading {model_name}")
        model = SentenceTransformer(model_path, local_files_only=True)
        cap_vecs = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True, show_progress_bar=True)
        item_vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        semantic_sims[model_name] = item_vecs @ cap_vecs.T

    minilm_sims = semantic_sims["MiniLM"]
    bge_sims = semantic_sims["BGE-large-v1.5"]

    results = {}

    results["TF-IDF"] = run_one_cv("TF-IDF (v2, leak-fixed)", texts, gold_labels, true_code, skf_main, is_tfidf=True)
    results["MiniLM"] = run_one_cv("MiniLM (v2)", None, gold_labels, true_code, skf_main, sims_precomp=minilm_sims)
    results["BGE"] = run_one_cv("BGE-large-v1.5 (v2)", None, gold_labels, true_code, skf_main, sims_precomp=bge_sims)
    results["Hybrid"] = run_hybrid_cv("Hybrid (v2, alpha-fusion threshold + 18-dim classifier, single-loop leak-fixed)",
                                      texts, minilm_sims, gold_labels, true_code, skf_main)

    for name in ["TF-IDF", "MiniLM", "BGE", "Hybrid"]:
        th_labels, clf_labels, th_f1s, clf_f1s = results[name]
        wa_subtype_report(f"{name} -- classifier", ids, gold_labels, clf_labels)
        pd.DataFrame({"fold": range(1, 11), "threshold_f1": th_f1s, "classifier_f1": clf_f1s}
                     ).to_csv(f"{name.replace('-', '_')}_v2_perfold_f1.csv", index=False)

    variants = {
        "6 raw blocks only": [0, 1, 2, 3, 4, 5],
        "6 blocks + mean_sim only": [0, 1, 2, 3, 4, 5, 6],
        "6 blocks + std_sim + margin": [0, 1, 2, 3, 4, 5, 7, 8],
        "full 9-dim (baseline)": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "full 9-dim minus block2": [0, 2, 3, 4, 5, 6, 7, 8],
        "full 9-dim minus std_sim": [0, 1, 2, 3, 4, 5, 6, 8],
        "full 9-dim minus margin": [0, 1, 2, 3, 4, 5, 6, 7],
        "full 9-dim minus mean_sim": [0, 1, 2, 3, 4, 5, 7, 8],
    }
    gold_arr = np.array(gold_labels)
    for model_name, sims in [("MiniLM", minilm_sims), ("BGE-large-v1.5", bge_sims)]:
        print(f"\n{'*' * 70}\n  ABLATION: {model_name} (v2)\n{'*' * 70}")
        X_full = build_features(sims)
        ablation_rows = []
        for variant_name, cols in variants.items():
            fold_f1s = []
            for train_idx, test_idx in skf_main.split(np.zeros(n), true_code):
                X_train, X_test = X_full[train_idx][:, cols], X_full[test_idx][:, cols]
                scaler = StandardScaler().fit(X_train)
                clf = LogisticRegression(class_weight="balanced", max_iter=2000)
                clf.fit(scaler.transform(X_train), gold_arr[train_idx])
                pred = clf.predict(scaler.transform(X_test))
                pred_code = np.array([LABEL_TO_CODE[l] for l in pred])
                fold_f1s.append(fast_macro_f1(pred_code, true_code[test_idx]))
            fold_arr = np.array(fold_f1s)
            ablation_rows.append((variant_name, len(cols), fold_arr.mean(), fold_arr.std(ddof=1)))
            print(f"  {variant_name:<32} n={len(cols):<3} mean={fold_arr.mean():.4f} std={fold_arr.std(ddof=1):.4f}")
        baseline = [r[2] for r in ablation_rows if r[0] == "full 9-dim (baseline)"][0]
        print(f"  Deltas vs baseline ({baseline:.4f}):")
        for nm, nf, mf, sf in ablation_rows:
            if nm != "full 9-dim (baseline)":
                print(f"    {nm:<32} delta={mf - baseline:+.4f}")
        pd.DataFrame(ablation_rows, columns=["variant", "n_features", "mean_macro_f1", "std_macro_f1"]
                     ).to_csv(f"ablation_{model_name.replace('-', '_')}_v2_results.csv", index=False)

    for model_name, sims in [("MiniLM", minilm_sims), ("BGE-large-v1.5", bge_sims)]:
        print(f"\n{'*' * 70}\n  COEFFICIENT STABILITY: {model_name} (v2)\n{'*' * 70}")
        X = build_features(sims)
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(scaler.transform(X), gold_arr)
        classes = list(clf.classes_)
        fold_coefs = []
        for train_idx, _ in skf_main.split(X, gold_arr):
            sc = StandardScaler().fit(X[train_idx])
            c = LogisticRegression(class_weight="balanced", max_iter=2000).fit(sc.transform(X[train_idx]),
                                                                               gold_arr[train_idx])
            aligned = np.zeros_like(c.coef_)
            for i, cls in enumerate(classes):
                aligned[i] = c.coef_[list(c.classes_).index(cls)]
            fold_coefs.append(aligned)
        fold_coefs = np.stack(fold_coefs, axis=0)
        mean_coefs, std_coefs = fold_coefs.mean(axis=0), fold_coefs.std(axis=0)
        for i, cls in enumerate(classes):
            print(f"\n  class={cls}")
            order = np.argsort(-np.abs(mean_coefs[i]))
            for j in order:
                sign_flips = np.sum(np.sign(fold_coefs[:, i, j]) != np.sign(mean_coefs[i, j]))
                print(f"    {FEATURE_NAMES[j]:<32} full-fit={clf.coef_[i, j]:+.3f}  "
                      f"cv-mean={mean_coefs[i, j]:+.3f}  cv-std={std_coefs[i, j]:.3f}  "
                      f"sign-flips-across-10-folds={sign_flips}")

    print(f"\n{'*' * 70}\n  REPEATED CV ({N_REPEATS}x10-fold) + NADEAU-BENGIO CORRECTED TESTS (v2)\n{'*' * 70}")

    def tfidf_build_X(train_idx, test_idx):
        train_sims, test_sims = tfidf_sims_single_fold(texts, train_idx, test_idx)
        return build_features(train_sims), build_features(test_sims)

    def hybrid_build_X(train_idx, test_idx):
        train_sims, test_sims = tfidf_sims_single_fold(texts, train_idx, test_idx)
        return (np.hstack([minilm_sims[train_idx], build_features(train_sims)]),
                np.hstack([minilm_sims[test_idx], build_features(test_sims)]))

    def make_sem_build_X(sims):
        def _fn(train_idx, test_idx):
            return build_features(sims[train_idx]), build_features(sims[test_idx])

        return _fn

    def tfidf_build_scores(train_idx, test_idx):
        train_sims, test_sims = tfidf_sims_single_fold(texts, train_idx, test_idx)
        return train_sims.max(axis=1), test_sims.max(axis=1)

    def make_sem_build_scores(sims):
        def _fn(train_idx, test_idx):
            return sims[train_idx].max(axis=1), sims[test_idx].max(axis=1)

        return _fn

    repeated_clf = {}
    repeated_th = {}
    repeated_clf["TF-IDF"] = repeated_cv_classifier_f1(tfidf_build_X, gold_labels, seed_offset=0)
    repeated_th["TF-IDF"] = repeated_cv_threshold_f1(tfidf_build_scores, gold_labels, true_code, seed_offset=0)
    repeated_clf["MiniLM"] = repeated_cv_classifier_f1(make_sem_build_X(minilm_sims), gold_labels, seed_offset=0)
    repeated_th["MiniLM"] = repeated_cv_threshold_f1(make_sem_build_scores(minilm_sims), gold_labels, true_code,
                                                     seed_offset=0)
    repeated_clf["BGE"] = repeated_cv_classifier_f1(make_sem_build_X(bge_sims), gold_labels, seed_offset=0)
    repeated_th["BGE"] = repeated_cv_threshold_f1(make_sem_build_scores(bge_sims), gold_labels, true_code,
                                                  seed_offset=0)
    repeated_clf["Hybrid"] = repeated_cv_classifier_f1(hybrid_build_X, gold_labels, seed_offset=0)
    repeated_th["Hybrid"] = repeated_cv_hybrid_threshold_f1(texts, minilm_sims, gold_labels, true_code, seed_offset=0)

    n_test = n // 10
    n_train = n - n_test

    print("\n    A. Classifier vs Threshold, within each representation (Nadeau-Bengio corrected)    ")
    for name in ["TF-IDF", "MiniLM", "BGE", "Hybrid"]:
        diffs = repeated_clf[name] - repeated_th[name]
        t_stat, p_val, ci_lo, ci_hi, mean_d = nadeau_bengio_paired_test(diffs, n_train, n_test)
        print(f"  {name}: classifier-threshold mean diff={mean_d:+.4f}  t={t_stat:.3f}  p={p_val:.5f}  "
              f"95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  n_diffs={len(diffs)}")

    print("\n    B. Classifier vs Classifier, across representations (Nadeau-Bengio + Holm)    ")
    pairs = [("MiniLM", "BGE"), ("MiniLM", "Hybrid"), ("BGE", "Hybrid"),
             ("TF-IDF", "MiniLM"), ("TF-IDF", "BGE"), ("TF-IDF", "Hybrid")]
    raw_p = []
    labels_out = []
    for a, b in pairs:
        diffs = repeated_clf[a] - repeated_clf[b]
        t_stat, p_val, ci_lo, ci_hi, mean_d = nadeau_bengio_paired_test(diffs, n_train, n_test)
        print(
            f"  {a} vs {b}: mean diff={mean_d:+.4f}  t={t_stat:.3f}  p={p_val:.5f}  95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}]")
        raw_p.append(p_val)
        labels_out.append(f"{a} vs {b}")
    order = np.argsort(raw_p)
    m = len(raw_p)
    holm_reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = 0.05 / (m - rank)
        if raw_p[idx] < threshold:
            holm_reject[idx] = True
        else:
            break
    print("\n  Holm-Bonferroni correction:")
    for i, lbl in enumerate(labels_out):
        print(f"    {lbl:<20} raw p={raw_p[i]:.5f}  Holm-significant={holm_reject[i]}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
